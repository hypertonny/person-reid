"""
gallery.py
----------
Global Gallery — persistent memory of all observed persons across cameras.

Responsibilities:
  - Maintain a dictionary of global IDs → embedding prototypes
  - New ID creation and EMA prototype updates
  - Cosine similarity matching (exact numpy for small galleries, FAISS for large)
  - Optimal batch assignment via Hungarian algorithm (scipy linear_sum_assignment)
  - Gallery pruning (remove stale entries beyond TTL)
  - Thread-safe operations via RLock

Data structure per entry:
  {
      "ID-001": {
          "embedding":        np.array(512,),   # current prototype (L2-normalised, EMA updated)
          "initial_embedding": np.array(512,),  # frozen first-seen embedding (anti-drift anchor)
          "first_seen":       float,             # timestamp (seconds)
          "last_seen":        float,             # timestamp (seconds)
          "last_camera":      str,               # most recent camera ID
          "cameras":          set,               # cameras this person has been seen in
          "n_obs":            int,               # total observation count
      }
  }
"""

import logging
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class GlobalGallery:
    """
    Thread-safe global identity gallery for multi-camera person ReID.

    Parameters
    ----------
    config : dict
        The full loaded config dict. Uses 'matching', 'gallery' sections.
    """

    def __init__(self, config: dict):
        matching_cfg = config.get("matching", {})
        gallery_cfg  = config.get("gallery", {})

        self.cosine_threshold = matching_cfg.get("cosine_threshold", 0.70)
        # Same-camera re-entry threshold: slightly relaxed for occlusion recovery
        # but NOT so low that it lets two different people merge (must stay >= 0.65)
        self.same_camera_threshold = matching_cfg.get("same_camera_threshold", 0.65)
        # Reduce EMA aggressiveness to prevent prototype drift
        self.ema_alpha        = matching_cfg.get("ema_alpha", 0.03)
        self.ema_max_updates  = matching_cfg.get("ema_max_updates", 500)
        self.use_hungarian    = matching_cfg.get("use_hungarian", True)
        self.faiss_threshold  = matching_cfg.get("faiss_threshold", 1000)
        self.ttl_sec          = gallery_cfg.get("ttl_sec", 3600)
        self.persistence_path = Path(gallery_cfg.get("persistence_path", "outputs/gallery_state/"))
        self.save_on_exit     = gallery_cfg.get("save_on_exit", True)

        self._lock: threading.RLock = threading.RLock()
        self._gallery: Dict[str, dict] = {}
        self._id_counter: int = 0
        self._faiss_index = None   # Lazily built when gallery exceeds faiss_threshold

        self.persistence_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"[Gallery] Initialised — threshold={self.cosine_threshold}, "
            f"same_cam_threshold={self.same_camera_threshold}, "
            f"ema_alpha={self.ema_alpha}, ttl={self.ttl_sec}s"
        )

    # ------------------------------------------------------------------
    # Public API — single track matching
    # ------------------------------------------------------------------

    def match_or_create(
        self,
        embedding: np.ndarray,
        camera_id: str,
        timestamp: float,
        st_model=None,
        active_ids: set = None,
    ) -> str:
        """
        Match a single embedding to the gallery, or create a new global ID.

        Uses dual-embedding matching:
          - Compare against both EMA prototype AND frozen initial embedding
          - Use max(sim_ema, sim_initial) for robust re-identification

        Parameters
        ----------
        embedding  : L2-normalised float32 array of shape (embedding_dim,)
        camera_id  : ID string of the camera that observed this person
        timestamp  : observation time in seconds
        st_model   : optional SpatioTemporalModel for feasibility gating
        active_ids : set of global IDs already assigned to tracked persons
                     this frame — EXCLUDED from candidates to prevent two
                     different people from sharing one ID.

        Returns
        -------
        str : assigned global ID (e.g. "ID-001")
        """
        with self._lock:
            candidates = self._get_feasible_candidates(
                camera_id, timestamp, st_model, exclude_ids=active_ids
            )

            if not candidates:
                return self._create_new_id(embedding, camera_id, timestamp)

            best_gid, best_sim = self._dual_match(embedding, candidates, camera_id)
            threshold = self._get_threshold(best_gid, camera_id)

            if best_sim >= threshold:
                self._update_entry(best_gid, embedding, camera_id, timestamp)
                logger.debug(
                    f"[Gallery] Re-identified {best_gid} in {camera_id} "
                    f"(sim={best_sim:.3f}, thr={threshold:.2f})"
                )
                return best_gid
            else:
                new_gid = self._create_new_id(embedding, camera_id, timestamp)
                logger.debug(
                    f"[Gallery] New ID {new_gid} in {camera_id} "
                    f"(best_sim={best_sim:.3f} < threshold={threshold:.2f})"
                )
                return new_gid

    # ------------------------------------------------------------------
    # Public API — batch matching (Hungarian algorithm)
    # ------------------------------------------------------------------

    def batch_match_or_create(
        self,
        embeddings: np.ndarray,
        camera_id: str,
        timestamp: float,
        st_model=None,
        active_ids: set = None,
    ) -> List[str]:
        """
        Optimal one-to-one assignment for a batch of new observations.
        Uses the Hungarian algorithm to minimise total assignment cost.
        Uses dual-embedding matching (EMA + initial) for robust similarity.

        Parameters
        ----------
        embeddings : (N, D) float32 array of L2-normalised embeddings
        camera_id  : camera that observed all these tracks
        timestamp  : shared observation timestamp
        active_ids : set of global IDs already assigned to live tracks —
                     EXCLUDED from candidates so two people can't share one ID.

        Returns
        -------
        list of str : global ID assigned to each embedding row
        """
        n = len(embeddings)
        if n == 0:
            return []

        with self._lock:
            candidates = self._get_feasible_candidates(
                camera_id, timestamp, st_model, exclude_ids=active_ids
            )

            if not candidates or not self.use_hungarian:
                # Fall back to greedy per-track matching
                assigned = []
                # Track which IDs we've already assigned in this batch
                # so two rows can't claim the same gallery entry
                batch_used = set(active_ids) if active_ids else set()
                for emb in embeddings:
                    gid = self.match_or_create(
                        emb, camera_id, timestamp, st_model, active_ids=batch_used
                    )
                    batch_used.add(gid)
                    assigned.append(gid)
                return assigned

            # Build bank-based similarity matrix (N, M)
            # We compute the max similarity across ALL stored appearances in the bank
            sim_best = np.zeros((n, len(candidates)), dtype=np.float32)
            for j, gid in enumerate(candidates):
                bank = self._gallery[gid].get("embedding_bank", [])
                if not bank:
                    # Fallback for corrupted/old gallery entries
                    bank = [self._gallery[gid]["embedding"], self._gallery[gid]["initial_embedding"]]
                bank_mat = np.stack(bank)  # (K, D)
                sims = embeddings @ bank_mat.T  # (N, K)
                sim_best[:, j] = np.max(sims, axis=1)

            cost_matrix = 1.0 - sim_best
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_ids = [None] * n
            matched_gallery = set()  # prevents two rows in this batch stealing same ID

            for r, c in zip(row_ind, col_ind):
                sim = 1.0 - cost_matrix[r, c]
                gid = candidates[c]
                threshold = self._get_threshold(gid, camera_id)

                if sim >= threshold and gid not in matched_gallery:
                    self._update_entry(gid, embeddings[r], camera_id, timestamp)
                    assigned_ids[r] = gid
                    matched_gallery.add(gid)
                    logger.info(
                        f"[Gallery] MATCHED {gid} in {camera_id} "
                        f"(sim={sim:.3f} >= thr={threshold:.2f})"
                    )
                else:
                    logger.info(
                        f"[Gallery] REJECTED match for {gid} in {camera_id} "
                        f"(sim={sim:.3f} < thr={threshold:.2f} or already matched)"
                    )

            # Any unassigned tracks get new IDs
            for i in range(n):
                if assigned_ids[i] is None:
                    new_gid = self._create_new_id(embeddings[i], camera_id, timestamp)
                    assigned_ids[i] = new_gid
                    logger.info(f"[Gallery] Created NEW ID {new_gid} in {camera_id} because best candidate sim was too low")

            return assigned_ids

    # ------------------------------------------------------------------
    # Gallery state queries
    # ------------------------------------------------------------------

    def size(self) -> int:
        with self._lock:
            return len(self._gallery)

    def get_entry(self, global_id: str) -> Optional[dict]:
        with self._lock:
            return dict(self._gallery.get(global_id, {}))

    def all_ids(self) -> List[str]:
        with self._lock:
            return list(self._gallery.keys())

    def stats(self) -> dict:
        with self._lock:
            n = len(self._gallery)
            cameras_seen = set()
            for entry in self._gallery.values():
                cameras_seen.update(entry["cameras"])
            return {
                "total_ids": n,
                "cameras_active": list(cameras_seen),
                "id_counter": self._id_counter,
            }

    # ------------------------------------------------------------------
    # Public API — direct EMA update for known tracks
    # ------------------------------------------------------------------

    def update_entry(
        self,
        global_id: str,
        new_embedding: np.ndarray,
        camera_id: str,
        timestamp: float,
    ) -> None:
        """
        Update an existing gallery entry's embedding via EMA.
        Called by CameraPipeline for tracks that are already cached
        (i.e., the local track ID → global ID mapping is known).
        This avoids re-querying the gallery every frame.

        Parameters
        ----------
        global_id     : the known global ID to update
        new_embedding : new L2-normalised embedding observation
        camera_id     : camera this observation came from
        timestamp     : observation timestamp
        """
        with self._lock:
            if global_id in self._gallery:
                self._update_entry(global_id, new_embedding, camera_id, timestamp)

    # ------------------------------------------------------------------
    # Gallery maintenance
    # ------------------------------------------------------------------

    def prune(self, current_time: float) -> int:
        """Remove entries older than TTL. Returns count of pruned entries."""
        with self._lock:
            stale = [
                gid for gid, entry in self._gallery.items()
                if (current_time - entry["last_seen"]) > self.ttl_sec
            ]
            for gid in stale:
                del self._gallery[gid]
            if stale:
                logger.info(f"[Gallery] Pruned {len(stale)} stale entries: {stale}")
            return len(stale)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, suffix: str = "") -> None:
        """Pickle the current gallery state to disk."""
        path = self.persistence_path / f"gallery{suffix}.pkl"
        with self._lock:
            data = {
                "gallery": self._gallery,
                "id_counter": self._id_counter,
                "saved_at": time.time(),
            }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"[Gallery] Saved to {path}")

    def load(self, suffix: str = "") -> bool:
        """Load a previously saved gallery from disk."""
        path = self.persistence_path / f"gallery{suffix}.pkl"
        if not path.exists():
            logger.warning(f"[Gallery] No saved gallery found at {path}")
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        with self._lock:
            self._gallery = data["gallery"]
            self._id_counter = data["id_counter"]
        logger.info(
            f"[Gallery] Loaded {len(self._gallery)} entries from {path} "
            f"(saved at {data['saved_at']:.0f})"
        )
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_id(self) -> str:
        self._id_counter += 1
        return f"ID-{self._id_counter:04d}"

    def _create_new_id(
        self, embedding: np.ndarray, camera_id: str, timestamp: float
    ) -> str:
        gid = self._generate_id()
        self._gallery[gid] = {
            "embedding":         embedding.copy(),
            "initial_embedding": embedding.copy(),   # FIX 1: frozen anchor
            "first_seen":        timestamp,
            "last_seen":         timestamp,
            "last_camera":       camera_id,
            "cameras":           {camera_id},
            "n_obs":             1,
            "embedding_bank":    [embedding.copy()],  # Bank of diverse appearances
        }
        logger.info(f"[Gallery] ✨ New person {gid} first seen in {camera_id} @ t={timestamp:.2f}s")
        return gid

    def _update_entry(
        self, gid: str, new_embedding: np.ndarray, camera_id: str, timestamp: float
    ) -> None:
        """EMA update of the prototype embedding + metadata refresh."""
        entry = self._gallery[gid]

        # FIX 2: Cap EMA updates to prevent runaway drift
        if entry["n_obs"] < self.ema_max_updates:
            updated = (
                self.ema_alpha * new_embedding
                + (1.0 - self.ema_alpha) * entry["embedding"]
            )
            norm = np.linalg.norm(updated)
            entry["embedding"] = updated / (norm + 1e-8)

        entry["last_seen"] = timestamp
        entry["last_camera"] = camera_id
        entry["cameras"].add(camera_id)
        entry["n_obs"] += 1

        # FIX: Update Embedding Bank with diverse appearances
        bank = entry.setdefault("embedding_bank", [entry["initial_embedding"].copy()])
        
        # If this new embedding is sufficiently different from what we've stored
        # (e.g. person turned around, different lighting), add it to the bank.
        # Max size of bank limits memory growth.
        bank_mat = np.stack(bank)
        sims = new_embedding @ bank_mat.T
        max_sim = float(np.max(sims))
        
        if max_sim < 0.85 and len(bank) < 20:
            bank.append(new_embedding.copy())
            logger.debug(f"[Gallery] Added new appearance mode to {gid} bank (size={len(bank)})")

    def _dual_match(
        self, embedding: np.ndarray, candidates: List[str], camera_id: str
    ) -> Tuple[str, float]:
        """
        FIX 1: Dual-embedding matching.
        Compare query against both EMA prototypes AND frozen initial embeddings.
        Return the best (gid, similarity) using max(sim_ema, sim_initial).
        """
        best_gid = None
        best_sim = -1.0

        for gid in candidates:
            entry = self._gallery[gid]
            bank = entry.get("embedding_bank", [])
            if not bank:
                bank = [entry["embedding"], entry["initial_embedding"]]
            
            bank_mat = np.stack(bank)
            sims = embedding @ bank_mat.T
            sim = float(np.max(sims))
            
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        return best_gid, best_sim

    def _get_threshold(self, gid: str, camera_id: str) -> float:
        """
        FIX 3: Adaptive threshold based on camera context.
        If the gallery entry was last seen in the SAME camera,
        use a relaxed threshold (occlusion recovery).
        Otherwise, use the standard cross-camera threshold.
        """
        if gid is None:
            return self.cosine_threshold

        entry = self._gallery.get(gid)
        if entry and entry.get("last_camera") == camera_id:
            return self.same_camera_threshold
        return self.cosine_threshold

    def _get_feasible_candidates(
        self, camera_id: str, timestamp: float, st_model,
        exclude_ids: set = None
    ) -> List[str]:
        """
        Return gallery IDs that pass the spatio-temporal feasibility gate.
        IDs in `exclude_ids` are always skipped — they are already assigned
        to an actively-tracked person this frame, so a NEW track must not
        claim them (this prevents two people sharing one global ID).
        """
        excluded = exclude_ids or set()

        if not self._gallery:
            return []

        candidates = []
        for gid, entry in self._gallery.items():
            if gid in excluded:
                continue   # already live — off-limits for new tracks
            if st_model is None:
                candidates.append(gid)
            else:
                last_camera = entry.get("last_camera", "")
                last_time   = entry["last_seen"]
                if st_model.is_feasible(last_camera, camera_id, last_time, timestamp):
                    candidates.append(gid)
        return candidates
