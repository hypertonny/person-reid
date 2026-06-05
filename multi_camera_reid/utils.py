"""
utils.py
--------
Utility functions for the Multi-Camera Person ReID system.

Includes:
  - Crop quality assessment (blur, size, aspect ratio)
  - HSV colour histogram (optional supplementary feature)
  - Horizontal body-part split helper
  - Logging setup re-export
  - Timer context manager for profiling
"""

import logging
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Crop Quality Assessment
# ============================================================

def is_quality_crop(
    crop: np.ndarray,
    min_height: int = 30,
    min_width: int = 15,
    min_aspect: float = 0.5,   # width/height — too wide = not a person
    max_aspect: float = 1.5,
    blur_threshold: float = 50.0,  # Laplacian variance — below = blurry
) -> bool:
    """
    Return True if the crop is suitable for ReID feature extraction.

    Parameters
    ----------
    crop          : BGR image crop (H, W, 3)
    min_height    : minimum crop height in pixels
    min_width     : minimum crop width in pixels
    min_aspect    : minimum width/height ratio
    max_aspect    : maximum width/height ratio
    blur_threshold: reject crops blurrier than this (Laplacian variance)
    """
    if crop is None or crop.size == 0:
        return False

    h, w = crop.shape[:2]

    if h < min_height or w < min_width:
        return False

    aspect = w / h
    if not (min_aspect <= aspect <= max_aspect):
        return False

    # Blur detection via Laplacian variance
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < blur_threshold:
        return False

    return True


# ============================================================
# HSV Colour Histogram (Supplementary Feature)
# ============================================================

def hsv_histogram(
    crop: np.ndarray,
    h_bins: int = 16,
    s_bins: int = 8,
    v_bins: int = 4,
    split_body: bool = True,
) -> np.ndarray:
    """
    Compute HSV histogram of a person crop.
    Optionally split the crop horizontally into upper / lower body halves
    and concatenate their histograms for better discriminability.

    Parameters
    ----------
    crop       : BGR image crop (H, W, 3)
    h_bins     : number of hue bins
    s_bins     : number of saturation bins
    v_bins     : number of value bins
    split_body : if True, compute separate histograms for upper and lower body

    Returns
    -------
    np.ndarray : L2-normalised float32 histogram vector.
                 Length = (h_bins * s_bins * v_bins) * (2 if split_body else 1)
    """
    def _hist_region(region: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2],
            None,
            [h_bins, s_bins, v_bins],
            [0, 180, 0, 256, 0, 256],
        )
        hist = hist.flatten().astype(np.float32)
        norm = np.linalg.norm(hist)
        return hist / (norm + 1e-8)

    if not split_body:
        return _hist_region(crop)

    h = crop.shape[0]
    mid = h // 2
    upper = crop[:mid, :]
    lower = crop[mid:, :]

    feat_upper = _hist_region(upper) if upper.size > 0 else np.zeros(h_bins * s_bins * v_bins)
    feat_lower = _hist_region(lower) if lower.size > 0 else np.zeros(h_bins * s_bins * v_bins)
    combined = np.concatenate([feat_upper, feat_lower])
    norm = np.linalg.norm(combined)
    return combined / (norm + 1e-8)


# ============================================================
# Horizontal Body-Part Zones (for part-based ReID)
# ============================================================

def split_body_parts(crop: np.ndarray, n_zones: int = 4) -> List[np.ndarray]:
    """
    Split a person crop into N horizontal zones (top → bottom).
    Used for part-based feature extraction to handle occlusion.

    Parameters
    ----------
    crop    : BGR image crop (H, W, 3)
    n_zones : number of horizontal zones (default 4: head, upper body, lower body, legs)

    Returns
    -------
    list of np.ndarray : list of zone crops (each may be empty if crop is too small)
    """
    h = crop.shape[0]
    zone_h = max(1, h // n_zones)
    zones = []
    for i in range(n_zones):
        y1 = i * zone_h
        y2 = min(h, y1 + zone_h) if i < n_zones - 1 else h
        zones.append(crop[y1:y2, :])
    return zones


# ============================================================
# ReID Score Combination
# ============================================================

def fuse_scores(
    appearance_sim: float,
    st_prob: float,
    appearance_weight: float = 0.85,
    st_weight: float = 0.15,
) -> float:
    """
    Combine appearance cosine similarity with spatio-temporal probability.
    Returns a fused score in [0, 1].

    Parameters
    ----------
    appearance_sim     : cosine similarity from ReID (0–1)
    st_prob            : Gaussian probability from SpatioTemporalModel (0–1)
    appearance_weight  : weight for appearance (default 0.85)
    st_weight          : weight for spatio-temporal (default 0.15)
    """
    return appearance_weight * appearance_sim + st_weight * st_prob


# ============================================================
# Profiling Timer
# ============================================================

@contextmanager
def timer(name: str = "Block", enabled: bool = True):
    """
    Context manager for simple wall-clock profiling.

    Usage:
        with timer("ReID batch"):
            embeddings = extractor.extract_batch(crops)
    """
    if not enabled:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"[Timer] {name}: {elapsed:.1f}ms")


# ============================================================
# Video Utilities
# ============================================================

def get_video_info(source: str) -> dict:
    """
    Open a video source and return basic metadata.

    Parameters
    ----------
    source : file path or RTSP URL

    Returns
    -------
    dict with keys: fps, width, height, total_frames, is_live
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return {}
    info = {
        "fps":          cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "is_live":      int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 0,
    }
    cap.release()
    return info


def make_summary_grid(
    frames: List[np.ndarray],
    labels: Optional[List[str]] = None,
    grid_w: int = 2,
    thumb_size: Tuple[int, int] = (320, 240),
) -> np.ndarray:
    """
    Arrange a list of annotated frames into a single grid image.
    Useful for side-by-side comparison of multiple cameras.

    Parameters
    ----------
    frames     : list of BGR frames (one per camera)
    labels     : optional list of text labels for each frame
    grid_w     : number of columns in the grid
    thumb_size : (width, height) of each thumbnail

    Returns
    -------
    np.ndarray : BGR grid image
    """
    import math
    n = len(frames)
    if n == 0:
        return np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)

    grid_h_count = math.ceil(n / grid_w)
    tw, th = thumb_size

    canvas = np.zeros((th * grid_h_count, tw * grid_w, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        r, c = divmod(i, grid_w)
        thumb = cv2.resize(frame, (tw, th))
        if labels and i < len(labels):
            cv2.putText(
                thumb, labels[i], (8, th - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
        canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = thumb

    return canvas
