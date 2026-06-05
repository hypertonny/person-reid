"""
camera_pipeline.py
------------------
Single-camera processing pipeline for one video source (file or RTSP stream).

Phase 1 (offline):
  - Reads from a video file (e.g., data/clips/cctv1.mp4)
  - Timestamps derived from: start_offset_sec + frame_index / fps
  - Writes annotated output video to outputs/annotated/

Phase 2 (live):
  - Reads from RTSP stream (mode switch in config.yaml)
  - Timestamps from time.time() (wall clock)
  - Auto-reconnects on stream drop

Responsibilities:
  1. Frame capture loop
  2. YOLO detection + ByteTrack tracking (local IDs)
  3. Crop extraction per tracked person
  4. Feature extraction via ReidExtractor (batch call)
  5. Gallery matching (batch_match_or_create)
  6. Frame annotation with global IDs + bounding boxes
  7. Output video writing (Phase 1) or display (Phase 2)
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Colour palette for bounding box annotation (BGR)
_ID_COLOURS = [
    (0, 200, 255), (0, 255, 100), (255, 80, 0),  (150, 0, 255),
    (255, 200, 0), (0, 150, 255), (200, 255, 0),  (255, 0, 150),
    (0, 255, 200), (255, 150, 0), (100, 255, 0),  (0, 80, 255),
]


def _id_colour(global_id: str) -> tuple:
    """Deterministic colour per global ID string."""
    idx = int(global_id.split("-")[-1]) % len(_ID_COLOURS)
    return _ID_COLOURS[idx]


class CameraPipeline:
    """
    Runs the full per-camera processing loop in its own thread.

    Parameters
    ----------
    camera_cfg   : dict from config['cameras'][i]
    full_config  : the full loaded config dict
    extractor    : shared ReidExtractor instance
    gallery      : shared GlobalGallery instance
    st_model     : shared SpatioTemporalModel instance (or None)
    stop_event   : threading.Event — set to request graceful shutdown
    """

    def __init__(
        self,
        camera_cfg: dict,
        full_config: dict,
        extractor,
        gallery,
        st_model,
        stop_event: threading.Event,
    ):
        self.camera_id       = camera_cfg["id"]
        self.source          = camera_cfg["source"]
        self.start_offset    = float(camera_cfg.get("start_offset_sec", 0.0))
        self.full_config     = full_config
        self.extractor       = extractor
        self.gallery         = gallery
        self.st_model        = st_model
        self.stop_event      = stop_event

        # Runtime mode
        runtime_cfg = full_config.get("runtime", {})
        self.mode        = runtime_cfg.get("mode", "offline")
        self.frame_skip  = int(runtime_cfg.get("frame_skip", 1))
        self.display     = runtime_cfg.get("display", False)

        # Detection config
        det_cfg = full_config.get("detection", {})
        self.yolo_model  = det_cfg.get("yolo_model", "yolov8n.pt")
        self.confidence  = float(det_cfg.get("confidence", 0.40))
        self.iou         = float(det_cfg.get("iou", 0.50))
        self.tracker     = det_cfg.get("tracker", "bytetrack.yaml")
        self.classes     = det_cfg.get("classes", [0])

        # Output config
        out_cfg = full_config.get("output", {})
        self.save_video     = out_cfg.get("save_annotated_video", True)
        self.annotated_path = Path(out_cfg.get("annotated_path", "outputs/annotated/"))
        self.stats_every    = int(out_cfg.get("print_stats_every_n_frames", 100))
        self.min_crop_h     = full_config.get("reid", {}).get("min_crop_height_px", 30)

        self.annotated_path.mkdir(parents=True, exist_ok=True)

        self._yolo = None        # Lazy-loaded
        self._writer = None      # cv2.VideoWriter for Phase 1
        self._frame_count = 0
        self._total_reid = 0

        # Track cache: local_track_id (int) → global_id (str)
        # This is THE key data structure that prevents duplicate ID creation.
        # Once ByteTrack assigns local ID T1 and we match it to ID-001,
        # every subsequent frame with T1 reuses ID-001 without re-querying the gallery.
        self._track_to_global: dict = {}

        # FIX 4: Cooldown buffer for recently disappeared tracks
        # When ByteTrack drops a track, we keep it here for _cooldown_sec seconds.
        # If a new track appears with high visual similarity to a cooled-down entry,
        # we re-link it to the same global ID without a full gallery query.
        # Structure: { global_id: { "embedding": np.array, "disappeared_at": float } }
        self._cooldown_buffer: dict = {}
        self._cooldown_sec = float(full_config.get("matching", {}).get("track_cooldown_sec", 10.0))
        # Must be HIGH enough to avoid merging different people (>= cross-camera threshold)
        self._cooldown_threshold = float(full_config.get("matching", {}).get("cooldown_cosine_threshold", 0.72))

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main processing loop. Called in a thread by the orchestrator."""
        logger.info(f"[{self.camera_id}] Starting pipeline — source: {self.source}")

        self._yolo = self._load_yolo()
        cap = self._open_source()
        if cap is None:
            logger.error(f"[{self.camera_id}] Cannot open source — aborting")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            f"[{self.camera_id}] Opened: fps={fps:.1f}, "
            f"frames={self.total_frames if self.total_frames > 0 else 'live'}"
        )

        if self.save_video and self.mode == "offline":
            self._writer = self._create_writer(cap)

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if self.mode == "offline":
                        logger.info(f"[{self.camera_id}] Video ended.")
                        break
                    else:
                        logger.warning(f"[{self.camera_id}] Stream lost — reconnecting…")
                        cap.release()
                        time.sleep(2.0)
                        cap = self._open_source()
                        if cap is None:
                            break
                        continue

                self._frame_count += 1

                # Frame skipping
                if self._frame_count % self.frame_skip != 0:
                    continue

                # Compute timestamp
                if self.mode == "offline":
                    timestamp = self.start_offset + (self._frame_count / fps)
                else:
                    timestamp = time.time()

                # Process frame
                annotated = self._process_frame(frame, timestamp)

                # Output
                if self._writer is not None:
                    self._writer.write(annotated)

                if self.display:
                    cv2.imshow(self.camera_id, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop_event.set()
                        break

                # Periodic stats
                if self._frame_count % self.stats_every == 0:
                    self._log_stats(timestamp)

        finally:
            cap.release()
            if self._writer:
                self._writer.release()
            if self.display:
                cv2.destroyWindow(self.camera_id)
            logger.info(
                f"[{self.camera_id}] Pipeline finished — "
                f"frames={self._frame_count}, re-ids={self._total_reid}"
            )

    # ------------------------------------------------------------------
    # Core per-frame logic
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Run detection + tracking + ReID + annotation on a single frame.
        
        Key logic:
          - ByteTrack assigns local track IDs (T1, T2, ...) that persist across frames
            within one camera.
          - We maintain _track_to_global: {local_track_id → global_id}
          - KNOWN tracks (already in cache): just update the gallery embedding (EMA)
          - NEW tracks (first appearance): check cooldown buffer first, then query gallery
          - STALE tracks (disappeared): moved to cooldown buffer instead of deleted
        
        Returns the annotated frame.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # 1. Detect + track with YOLO + ByteTrack
        results = self._yolo.track(
            frame,
            persist=True,
            classes=self.classes,
            conf=self.confidence,
            iou=self.iou,
            tracker=self.tracker,
            verbose=False,
        )

        if results is None or len(results) == 0:
            return annotated

        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            return annotated

        boxes     = result.boxes.xyxy.cpu().numpy().astype(int)   # (N, 4)
        track_ids = result.boxes.id.cpu().numpy().astype(int)     # (N,)
        confs     = result.boxes.conf.cpu().numpy()                # (N,)

        if len(boxes) == 0:
            return annotated

        # 2. Extract person crops
        crops = []
        valid_mask = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop_h = y2 - y1
            if crop_h < self.min_crop_h or x2 <= x1 or y2 <= y1:
                crops.append(None)
                valid_mask.append(False)
            else:
                crop = frame[y1:y2, x1:x2]
                crops.append(crop)
                valid_mask.append(True)

        # 3. Batch ReID feature extraction (for ALL valid detections)
        valid_crops = [c for c, v in zip(crops, valid_mask) if v]
        if valid_crops:
            all_embeddings = self.extractor.extract_batch(valid_crops)
        else:
            all_embeddings = np.empty((0, self.extractor.embedding_dim), dtype=np.float32)

        # Reconstruct full embedding list (zeros for invalid crops)
        embeddings = []
        valid_iter = iter(all_embeddings)
        for v in valid_mask:
            if v:
                embeddings.append(next(valid_iter))
            else:
                embeddings.append(np.zeros(self.extractor.embedding_dim, dtype=np.float32))

        # 4. Identify current valid tracks and handle STALE tracks first
        current_track_set = set()
        for i, tid in enumerate(track_ids):
            if valid_mask[i]:
                current_track_set.add(int(tid))

        # Move stale tracks to cooldown buffer BEFORE processing new tracks
        # This ensures that if a track breaks in this exact frame, its global ID
        # is freed and placed in the cooldown buffer for immediate re-linking.
        stale_tids = set(self._track_to_global.keys()) - current_track_set
        for tid in stale_tids:
            gid = self._track_to_global.pop(tid)
            entry = self.gallery.get_entry(gid)
            if entry and "embedding" in entry:
                self._cooldown_buffer[gid] = {
                    "embedding": entry["embedding"].copy(),
                    "disappeared_at": timestamp,
                }
            logger.debug(
                f"[{self.camera_id}] Track T{tid} ({gid}) → cooldown buffer "
                f"(buffer size: {len(self._cooldown_buffer)})"
            )

        # Expire old cooldown entries
        self._purge_cooldown(timestamp)

        # 5. Separate tracks into KNOWN (cached) vs NEW (need gallery query)
        global_ids = ["??"] * len(track_ids)

        # Active IDs = global IDs already assigned to a known live track this frame.
        # Since we already popped stale tracks, _track_to_global only has truly active ones.
        active_global_ids = set(self._track_to_global.values())

        new_track_indices = []       # indices of tracks that need gallery matching
        new_track_embeddings = []    # their embeddings

        for i, tid in enumerate(track_ids):
            if not valid_mask[i]:
                continue
            current_track_set.add(int(tid))

            if int(tid) in self._track_to_global:
                # KNOWN track — reuse cached global ID + EMA update
                gid = self._track_to_global[int(tid)]
                global_ids[i] = gid
                self.gallery.update_entry(gid, embeddings[i], self.camera_id, timestamp)
            else:
                # NEW track — check cooldown buffer first (re-entry after occlusion)
                cooldown_gid = self._check_cooldown(embeddings[i], timestamp, active_global_ids)
                if cooldown_gid is not None:
                    self._track_to_global[int(tid)] = cooldown_gid
                    global_ids[i] = cooldown_gid
                    active_global_ids.add(cooldown_gid)  # mark as live immediately
                    self.gallery.update_entry(cooldown_gid, embeddings[i], self.camera_id, timestamp)
                    logger.info(
                        f"[{self.camera_id}] Cooldown recovery: T{tid} → {cooldown_gid} "
                        f"(cache size: {len(self._track_to_global)})"
                    )
                else:
                    new_track_indices.append(i)
                    new_track_embeddings.append(embeddings[i])

        # 6. Gallery matching for NEW tracks only
        if new_track_embeddings:
            new_embs = np.array(new_track_embeddings)
            matched_gids = self.gallery.batch_match_or_create(
                new_embs, self.camera_id, timestamp, self.st_model,
                active_ids=active_global_ids   # exclude already-live IDs
            )
            for idx, gid in zip(new_track_indices, matched_gids):
                tid = int(track_ids[idx])
                self._track_to_global[tid] = gid
                global_ids[idx] = gid
                active_global_ids.add(gid)  # mark as live
                self._total_reid += 1
                logger.info(
                    f"[{self.camera_id}] New track T{tid} → {gid} "
                    f"(cache size: {len(self._track_to_global)})"
                )

        # 7. Annotate frame
        for i, (box, tid, gid, conf) in enumerate(zip(boxes, track_ids, global_ids, confs)):
            if not valid_mask[i]:
                continue
            x1, y1, x2, y2 = box
            colour = _id_colour(gid)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

            # Label: GlobalID + local track ID + confidence
            label = f"{gid} | T{tid} | {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 4, y1), colour, -1)
            cv2.putText(
                annotated, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA
            )

        # HUD overlay (top-left)
        self._draw_hud(annotated, timestamp, len(valid_crops))
        return annotated

    # ------------------------------------------------------------------
    # FIX 4: Cooldown buffer operations
    # ------------------------------------------------------------------

    def _check_cooldown(
        self, embedding: np.ndarray, timestamp: float,
        active_global_ids: set = None
    ) -> Optional[str]:
        """
        Check if a new track matches any recently disappeared track in the
        cooldown buffer. Skips IDs that are already live (active_global_ids)
        so a still-tracked person's ID can never be stolen by a new track.
        Returns the global ID if found, None otherwise.
        """
        if not self._cooldown_buffer:
            return None

        excluded = active_global_ids or set()
        best_gid = None
        best_sim = -1.0

        for gid, data in self._cooldown_buffer.items():
            if gid in excluded:
                continue  # already live — don't give this ID to another person
            age = timestamp - data["disappeared_at"]
            if age > self._cooldown_sec:
                continue
            sim = float(data["embedding"] @ embedding)
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        if best_gid is not None and best_sim >= self._cooldown_threshold:
            del self._cooldown_buffer[best_gid]
            logger.debug(
                f"[{self.camera_id}] Cooldown match: {best_gid} (sim={best_sim:.3f})"
            )
            return best_gid

        return None

    def _purge_cooldown(self, timestamp: float) -> None:
        """Remove entries that have been in cooldown longer than the timeout."""
        expired = [
            gid for gid, data in self._cooldown_buffer.items()
            if (timestamp - data["disappeared_at"]) > self._cooldown_sec
        ]
        for gid in expired:
            del self._cooldown_buffer[gid]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray, timestamp: float, n_people: int) -> None:
        """Draw camera ID, timestamp, person count on frame."""
        lines = [
            f"{self.camera_id}",
            f"t={timestamp:.1f}s",
            f"People: {n_people}",
            f"Gallery: {self.gallery.size()}",
        ]
        y = 20
        for line in lines:
            cv2.putText(
                frame, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
            )
            y += 22

    def _log_stats(self, timestamp: float) -> None:
        stats = self.gallery.stats()
        logger.info(
            f"[{self.camera_id}] Frame {self._frame_count} | t={timestamp:.1f}s | "
            f"Gallery size={stats['total_ids']} | Total re-ids={self._total_reid}"
        )

    def _load_yolo(self):
        """Load YOLO model (auto-downloads on first run)."""
        try:
            from ultralytics import YOLO
            model = YOLO(self.yolo_model)
            logger.info(f"[{self.camera_id}] Loaded YOLO: {self.yolo_model}")
            return model
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    def _open_source(self) -> Optional[cv2.VideoCapture]:
        """Open video source (file or RTSP)."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error(f"[{self.camera_id}] Failed to open source: {self.source}")
            return None
        return cap

    def _create_writer(self, cap: cv2.VideoCapture) -> Optional[cv2.VideoWriter]:
        """Create VideoWriter for annotated output (Phase 1 only)."""
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_name = f"{self.camera_id.replace(' ', '_').lower()}_annotated.mp4"
        out_path = str(self.annotated_path / out_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        logger.info(f"[{self.camera_id}] Writing annotated output to: {out_path}")
        return writer
