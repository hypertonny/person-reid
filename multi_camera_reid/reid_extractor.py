"""
reid_extractor.py
-----------------
OSNet-x1.0 wrapper for person re-identification feature extraction.

Responsibilities:
  - Load pretrained OSNet model (auto-download on first run)
  - Accept BGR person crops (numpy arrays from OpenCV)
  - Return L2-normalised 512-d float32 embeddings
  - GPU/CPU auto-detection with graceful fallback
  - Thread-safe: single model instance shared across camera threads

Usage:
    extractor = ReidExtractor(config)
    embedding = extractor.extract(crop_bgr)   # shape: (512,)
    embeddings = extractor.extract_batch([crop1, crop2, ...])  # shape: (N, 512)
"""

import logging
import os
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — map config name → torchreid model name
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "osnet_x1_0": "osnet_x1_0",
    "osnet_x0_75": "osnet_x0_75",
    "osnet_x0_5": "osnet_x0_5",
    "osnet_ibn_x1_0": "osnet_ibn_x1_0",
}


class ReidExtractor:
    """
    Thread-safe OSNet-based person re-identification feature extractor.

    Parameters
    ----------
    config : dict
        The 'reid' section from config.yaml, e.g.:
        {
            'model': 'osnet_x1_0',
            'weights': 'models/osnet_x1_0_market.pth',
            'device': 'auto',
            'embedding_dim': 512,
            'input_size': [256, 128],
            'min_crop_height_px': 30
        }
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("model", "osnet_x1_0")
        self.weights_path = config.get("weights", "models/osnet_x1_0_market.pth")
        self.embedding_dim = config.get("embedding_dim", 512)
        self.input_h, self.input_w = config.get("input_size", [256, 128])
        self.min_crop_h = config.get("min_crop_height_px", 30)

        # Device selection
        self.device = self._resolve_device(config.get("device", "auto"))
        logger.info(f"[ReidExtractor] Using device: {self.device}")

        # Build and load model
        self.model = self._load_model()
        self.model.eval()

        # Image preprocessing pipeline (ImageNet stats — OSNet trained with these)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_h, self.input_w)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        logger.info(
            f"[ReidExtractor] Ready — model={self.model_name}, "
            f"input={self.input_h}x{self.input_w}, embed_dim={self.embedding_dim}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Extract a single L2-normalised embedding from one person crop.

        Parameters
        ----------
        crop_bgr : np.ndarray
            BGR image crop from OpenCV. Shape: (H, W, 3).
            Returns zero vector if crop is too small or invalid.

        Returns
        -------
        np.ndarray
            L2-normalised float32 embedding of shape (embedding_dim,).
        """
        if crop_bgr is None or crop_bgr.size == 0:
            logger.debug("[ReidExtractor] Empty crop — returning zero embedding")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        h, w = crop_bgr.shape[:2]
        if h < self.min_crop_h:
            logger.debug(f"[ReidExtractor] Crop too small ({h}px) — skipping")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        batch = self._preprocess([crop_bgr])  # (1, 3, H, W)
        with torch.no_grad():
            feat = self.model(batch)           # (1, embedding_dim)
            feat = F.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy()[0]           # (embedding_dim,)

    def extract_batch(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for a batch of person crops in one forward pass.

        Parameters
        ----------
        crops_bgr : list of np.ndarray
            List of BGR person crops.

        Returns
        -------
        np.ndarray
            Float32 array of shape (N, embedding_dim). L2-normalised rows.
        """
        if not crops_bgr:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        valid_crops = []
        valid_indices = []
        for i, crop in enumerate(crops_bgr):
            if crop is not None and crop.size > 0 and crop.shape[0] >= self.min_crop_h:
                valid_crops.append(crop)
                valid_indices.append(i)

        result = np.zeros((len(crops_bgr), self.embedding_dim), dtype=np.float32)
        if not valid_crops:
            return result

        batch = self._preprocess(valid_crops)  # (N_valid, 3, H, W)
        with torch.no_grad():
            feats = self.model(batch)           # (N_valid, embedding_dim)
            feats = F.normalize(feats, p=2, dim=1)

        feats_np = feats.cpu().numpy()
        for out_idx, src_idx in enumerate(valid_indices):
            result[src_idx] = feats_np[out_idx]

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                dev = torch.device("cuda")
                logger.info(f"[ReidExtractor] CUDA available — {torch.cuda.get_device_name(0)}")
            else:
                dev = torch.device("cpu")
                logger.warning("[ReidExtractor] CUDA not available — falling back to CPU (slower)")
        else:
            dev = torch.device(device_str)
        return dev

    def _load_model(self) -> torch.nn.Module:
        """Load OSNet via torchreid. Auto-downloads weights if needed."""
        try:
            import torchreid
        except ImportError:
            raise ImportError(
                "torchreid not installed. Run:\n"
                "  pip install git+https://github.com/KaiyangZhou/deep-person-reid.git\n"
                "or pin: pip install torchreid==1.4.0"
            )

        torchreid_name = MODEL_REGISTRY.get(self.model_name)
        if torchreid_name is None:
            raise ValueError(
                f"Unknown ReID model '{self.model_name}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        weights_path = Path(self.weights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        # Build model
        model = torchreid.models.build_model(
            name=torchreid_name,
            num_classes=751,   # Market-1501 class count (only used for loading; overridden below)
            pretrained=not weights_path.exists(),  # Download if weights not cached
        )

        # Load custom weights if provided and exist
        if weights_path.exists():
            logger.info(f"[ReidExtractor] Loading weights from {weights_path}")
            state = torch.load(str(weights_path), map_location="cpu")
            # Handle various checkpoint formats
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
        else:
            logger.info(
                f"[ReidExtractor] Using default pretrained weights "
                f"(Market-1501). Will cache to {weights_path} on next explicit save."
            )

        model = model.to(self.device)
        return model

    def _preprocess(self, crops_bgr: List[np.ndarray]) -> torch.Tensor:
        """Convert a list of BGR crops to a normalised torch tensor batch."""
        tensors = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))
        batch = torch.stack(tensors).to(self.device)  # (N, 3, H, W)
        return batch
