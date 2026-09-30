"""DeepLabV3-ResNet50 (torchvision, VOC labels) for people, vehicles and animals.

A learned model for the classes that matter most for masking: moving objects.
It knows 20 everyday classes (trained on ground-level photos), so it is mapped
to our label set as person / vehicle / animal. Vegetation and water still come
from the colour rules of the placeholder, and roads and buildings are not
detected yet; the backend name says so.

Weights load from the local torch cache only; this backend never downloads.
Small objects seen from high altitude (a person is a few pixels at 100 m) will
often be missed; that is a known limit of a ground-level model.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dronerecon.config import SegmentationConfig
from dronerecon.segmentation.base import Label, MockSegmenter, SegmentationResult
from dronerecon.types import DataError

VOC_TO_LABEL = {
    "person": Label.PERSON,
    "bicycle": Label.VEHICLE, "bus": Label.VEHICLE, "car": Label.VEHICLE, "motorbike": Label.VEHICLE,
    "train": Label.VEHICLE, "boat": Label.VEHICLE, "aeroplane": Label.VEHICLE,
    "bird": Label.ANIMAL, "cat": Label.ANIMAL, "cow": Label.ANIMAL, "dog": Label.ANIMAL,
    "horse": Label.ANIMAL, "sheep": Label.ANIMAL,
}


def _weights():
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

    return DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1


def weights_cached() -> bool:
    """True when the model can run without any download."""
    try:
        import torch

        name = Path(_weights().url).name
        return (Path(torch.hub.get_dir()) / "checkpoints" / name).is_file()
    except Exception:  # torchvision missing or unusual install
        return False


class DeepLabSegmenter:
    name = "deeplabv3-voc+colour-rules"
    is_mock = False

    def __init__(self, cfg: SegmentationConfig | None = None, min_confidence: float = 0.5) -> None:
        if not weights_cached():
            raise DataError("DeepLabV3 weights are not in the local torch cache; use segmentation.backend 'mock'")
        import torch
        from torchvision.models.segmentation import deeplabv3_resnet50

        self.cfg = cfg or SegmentationConfig()
        self.min_conf = min_confidence
        self._torch = torch
        w = _weights()
        self._model = deeplabv3_resnet50(weights=w, weights_backbone=None).eval()
        self._mean = torch.tensor(w.transforms().mean).view(3, 1, 1)
        self._std = torch.tensor(w.transforms().std).view(3, 1, 1)
        cats = w.meta["categories"]
        self._lut = np.full(len(cats), int(Label.OTHER), np.uint8)
        for i, c in enumerate(cats):
            if c in VOC_TO_LABEL:
                self._lut[i] = int(VOC_TO_LABEL[c])
        self._colour = MockSegmenter(cfg)

    def segment(self, image: np.ndarray) -> SegmentationResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise DataError("segment: expected a BGR colour image")
        h, w = image.shape[:2]
        s = min(1.0, self.cfg.work_width / w)
        small = cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else image
        t = self._torch
        x = t.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)
        x = ((x - self._mean) / self._std)[None]
        with t.inference_mode():
            prob = t.softmax(self._model(x)["out"][0], dim=0).numpy()
        cls, conf = prob.argmax(0), prob.max(0)
        learned = self._lut[cls]
        keep = (learned != Label.OTHER) & (conf >= self.min_conf)

        base = self._colour.segment(small)  # vegetation / water from colour rules
        labels = np.where(keep, learned, base.labels).astype(np.uint8)
        confidence = np.where(keep, conf, base.confidence).astype(np.float32)
        if s < 1:
            labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
            confidence = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR)
        return SegmentationResult(labels, confidence, self.name, False)
