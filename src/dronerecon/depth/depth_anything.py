"""Depth Anything V2 (Small) monocular depth network, run through Hugging Face transformers.

The network predicts relative inverse depth (disparity) up to an unknown scale and
shift, so its maps are returned as kind "inverse_depth" and aligned to the metric
sparse points by `align_depth` like any other backend.

Weights load from a local folder or the local Hugging Face cache only; this
backend never downloads. It was trained on ground-level and synthetic images, so
steep nadir views of flat ground carry little depth signal for it; the alignment
and cross-view checks decide whether each map is usable.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dronerecon.config import DepthConfig
from dronerecon.depth.mono import DepthPrediction, FrameObservations
from dronerecon.types import DataError

WEIGHT_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")


def _local_dir(model: str) -> Path | None:
    """The folder holding the model files, from a path or the local HF cache; None if absent."""
    p = Path(model)
    if p.is_dir():
        return p if all((p / f).is_file() for f in WEIGHT_FILES) else None
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    found = [try_to_load_from_cache(model, f) for f in WEIGHT_FILES]
    if all(isinstance(f, str) for f in found):
        return Path(found[0]).parent
    return None


def weights_cached(cfg: DepthConfig | None = None) -> bool:
    """True when the model can run without any download."""
    return _local_dir((cfg or DepthConfig()).model) is not None


class DepthAnythingV2:
    is_mock = False

    def __init__(self, cfg: DepthConfig | None = None) -> None:
        cfg = cfg or DepthConfig()
        folder = _local_dir(cfg.model)
        if folder is None:
            raise DataError(f"Depth Anything V2 weights ('{cfg.model}') are not on disk; put the model folder there "
                            "or use depth.backend 'mock'")
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise DataError(f"Depth Anything V2 needs torch and transformers: {e}") from e
        self._torch = torch
        self.processor = AutoImageProcessor.from_pretrained(folder, local_files_only=True)
        self.net = AutoModelForDepthEstimation.from_pretrained(folder, local_files_only=True).eval()
        self.name = f"depth-anything-v2:{Path(cfg.model).name}"

    def predict(self, image: np.ndarray, obs: FrameObservations, full_width: int) -> DepthPrediction:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        torch = self._torch
        with torch.no_grad():
            inputs = self.processor(images=rgb, return_tensors="pt")
            out = self.net(**inputs).predicted_depth  # 1xhxw relative inverse depth
            disp = torch.nn.functional.interpolate(out[:, None], size=(h, w), mode="bicubic", align_corners=False)
        values = disp[0, 0].numpy().astype(np.float32)
        values[~np.isfinite(values)] = np.nan
        return DepthPrediction(values, "inverse_depth", self.name, False)
