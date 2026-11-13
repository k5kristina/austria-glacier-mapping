"""Training loop and losses."""

from glacier_monitoring.training.losses import SegmentationCriterion
from glacier_monitoring.training.trainer import (
    TrainingResult,
    set_global_seed,
    train_segmentation_model,
)

__all__ = ["SegmentationCriterion", "TrainingResult", "set_global_seed", "train_segmentation_model"]
