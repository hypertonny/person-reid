"""
st_model.py
-----------
Spatio-Temporal Travel-Time Model for cross-camera person ReID.

Purpose:
  Filter impossible cross-camera matches before computing appearance similarity.
  A person who left CCTV-1 at t=0s cannot appear on CCTV-2 at t=1s if the
  minimum walk time is 15s — this module enforces that constraint.

Strategy:
  - For each camera pair (A → B), maintain a Gaussian distribution N(μ, σ²)
    over observed travel times Δt = t_appear_B - t_disappear_A.
  - Use μ ± k*σ as the acceptance window (default k=3).
  - Phase 1: μ, σ are configured manually in config.yaml.
  - Phase 2 / future: learn μ, σ online from observed transitions.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CameraPairModel:
    """
    Gaussian travel-time model for a single directed camera pair (src → dst).

    Parameters
    ----------
    mu    : mean travel time in seconds
    sigma : standard deviation in seconds
    min_travel_sec : hard lower bound (physics constraint — reject anything faster)
    max_travel_sec : hard upper bound (reject if too old — person probably gone)
    k_sigma : acceptance window width in units of sigma
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        min_travel_sec: float,
        max_travel_sec: float,
        k_sigma: float = 3.0,
    ):
        self.mu = mu
        self.sigma = sigma
        self.min_travel = min_travel_sec
        self.max_travel = max_travel_sec
        self.k_sigma = k_sigma
        # Precompute acceptance bounds
        self._low  = max(min_travel_sec, mu - k_sigma * sigma)
        self._high = min(max_travel_sec, mu + k_sigma * sigma)

    def is_feasible(self, delta_t: float) -> bool:
        """
        Return True if the travel time delta_t falls within the acceptance window.

        Parameters
        ----------
        delta_t : t_appear_in_dst - t_disappear_from_src (seconds)
        """
        return self._low <= delta_t <= self._high

    def probability(self, delta_t: float) -> float:
        """Gaussian probability density (unnormalised) at delta_t."""
        if self.sigma <= 0:
            return 1.0 if abs(delta_t - self.mu) < 1e-6 else 0.0
        return float(np.exp(-0.5 * ((delta_t - self.mu) / self.sigma) ** 2))

    def update(self, delta_t: float, learning_rate: float = 0.05) -> None:
        """
        Online update of mu and sigma using exponential moving average.
        Used in Phase 2 to learn the model from real traffic.

        Parameters
        ----------
        delta_t       : observed travel time sample (seconds)
        learning_rate : EMA learning rate (small = stable, large = reactive)
        """
        old_mu = self.mu
        self.mu    = (1 - learning_rate) * self.mu    + learning_rate * delta_t
        self.sigma = (1 - learning_rate) * self.sigma + learning_rate * abs(delta_t - old_mu)
        # Recompute acceptance bounds
        self._low  = max(self.min_travel, self.mu - self.k_sigma * self.sigma)
        self._high = min(self.max_travel, self.mu + self.k_sigma * self.sigma)
        logger.debug(
            f"[ST] Updated model: μ={self.mu:.1f}s σ={self.sigma:.1f}s "
            f"window=[{self._low:.1f}, {self._high:.1f}]s"
        )

    def __repr__(self) -> str:
        return (
            f"CameraPairModel(μ={self.mu:.1f}s, σ={self.sigma:.1f}s, "
            f"window=[{self._low:.1f}, {self._high:.1f}]s)"
        )


class SpatioTemporalModel:
    """
    Manages spatio-temporal travel-time models for all camera pairs.

    Parameters
    ----------
    config : dict
        The full loaded config dict. Uses 'spatio_temporal' section.
    """

    def __init__(self, config: dict):
        st_cfg = config.get("spatio_temporal", {})
        self.enabled = st_cfg.get("enabled", True)
        self.k_sigma = st_cfg.get("k_sigma", 3.0)

        # Dict[Tuple[str,str], CameraPairModel]
        self._models: Dict[Tuple[str, str], CameraPairModel] = {}

        if self.enabled:
            self._build_models(st_cfg.get("camera_pairs", []))
        else:
            logger.warning("[ST] Spatio-temporal gating is DISABLED — all matches are candidates")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_feasible(
        self,
        src_camera: str,
        dst_camera: str,
        t_disappear: float,
        t_appear: float,
    ) -> bool:
        """
        Check whether a person could plausibly travel from src_camera to dst_camera
        given the time elapsed.

        Parameters
        ----------
        src_camera  : camera the person was last seen in
        dst_camera  : camera the person just appeared in
        t_disappear : timestamp when person left src_camera (seconds)
        t_appear    : timestamp when person appeared in dst_camera (seconds)

        Returns
        -------
        bool : True = feasible (keep as candidate), False = reject
        """
        if not self.enabled:
            return True

        # Same camera — always feasible (re-appearing after occlusion)
        if src_camera == dst_camera:
            return True

        delta_t = t_appear - t_disappear

        # In offline mode, cameras process in parallel — the same person can
        # be detected on CCTV-2 at video_time=0.96s while CCTV-1 already logged
        # last_seen=1.83s, producing a negative delta. This doesn't mean
        # "teleportation"; it means concurrent processing.
        #
        # Allow small negative deltas (overlap tolerance) and also allow
        # delta_t == 0 (simultaneous multi-cam views).
        overlap_tolerance_sec = 5.0  # allow up to 5s of "overlap"

        if delta_t < -overlap_tolerance_sec:
            logger.debug(
                f"[ST] Rejected {src_camera}→{dst_camera}: "
                f"Δt={delta_t:.1f}s exceeds overlap tolerance"
            )
            return False

        # For small/negative deltas (concurrent views), always feasible
        if delta_t < 0:
            logger.debug(
                f"[ST] Allowing {src_camera}→{dst_camera}: "
                f"Δt={delta_t:.1f}s within overlap tolerance"
            )
            return True

        model = self._get_model(src_camera, dst_camera)
        if model is None:
            # No model for this pair — allow by default
            logger.debug(
                f"[ST] No model for {src_camera}→{dst_camera} — defaulting to feasible"
            )
            return True

        feasible = model.is_feasible(delta_t)
        if not feasible:
            logger.debug(
                f"[ST] Rejected {src_camera}→{dst_camera}: "
                f"Δt={delta_t:.1f}s outside [{model._low:.1f}, {model._high:.1f}]s"
            )
        return feasible

    def travel_probability(
        self,
        src_camera: str,
        dst_camera: str,
        t_disappear: float,
        t_appear: float,
    ) -> float:
        """
        Return the Gaussian probability of the observed travel time.
        Useful as a soft weight to scale similarity scores.
        Returns 1.0 when no model is available (neutral).
        """
        if not self.enabled or src_camera == dst_camera:
            return 1.0
        model = self._get_model(src_camera, dst_camera)
        if model is None:
            return 1.0
        delta_t = t_appear - t_disappear
        return model.probability(delta_t)

    def observe_transition(
        self,
        src_camera: str,
        dst_camera: str,
        delta_t: float,
        learning_rate: float = 0.05,
    ) -> None:
        """
        Update the model with an observed transition (Phase 2 online learning).
        Call this when a high-confidence match is made across cameras.

        Parameters
        ----------
        src_camera    : camera person left
        dst_camera    : camera person appeared in
        delta_t       : t_appear - t_disappear (seconds)
        learning_rate : EMA learning rate
        """
        model = self._get_model(src_camera, dst_camera)
        if model is not None:
            model.update(delta_t, learning_rate)
        else:
            logger.debug(
                f"[ST] No model to update for {src_camera}→{dst_camera}"
            )

    def describe(self) -> str:
        """Human-readable summary of all models."""
        lines = ["[SpatioTemporalModel] Configured pairs:"]
        for (src, dst), model in self._models.items():
            lines.append(f"  {src} → {dst}: {model}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_models(self, pairs_config: list) -> None:
        for pair_cfg in pairs_config:
            cameras = pair_cfg.get("pair", [])
            if len(cameras) != 2:
                logger.warning(f"[ST] Invalid camera pair config: {pair_cfg}")
                continue

            cam_a, cam_b = cameras
            mu  = float(pair_cfg.get("mu_sec", 20.0))
            sig = float(pair_cfg.get("sigma_sec", 10.0))
            lo  = float(pair_cfg.get("min_travel_sec", 3.0))
            hi  = float(pair_cfg.get("max_travel_sec", 120.0))

            model = CameraPairModel(mu, sig, lo, hi, self.k_sigma)
            # Register both directions
            self._models[(cam_a, cam_b)] = model
            self._models[(cam_b, cam_a)] = model   # symmetric for now
            logger.info(f"[ST] Registered pair {cam_a} ↔ {cam_b}: {model}")

    def _get_model(
        self, src: str, dst: str
    ) -> Optional[CameraPairModel]:
        return self._models.get((src, dst))
