"""One segmenter, two jobs: exclusion masks for matching and semantic classes for 3D voting."""

from aerometric.segmentation.base import (
    DYNAMIC,
    EXCLUDED_FROM_MATCHING,
    Label,
    Segmenter,
    SegmentationResult,
    load_labels,
    make_segmenter,
    save_labels,
)

__all__ = ["DYNAMIC", "EXCLUDED_FROM_MATCHING", "Label", "Segmenter", "SegmentationResult",
           "load_labels", "make_segmenter", "save_labels"]
