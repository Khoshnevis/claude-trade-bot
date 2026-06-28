"""Meta-labeling ML for the gold (XAUUSD) strategy."""
from .model import MetaLabeler
from .pipeline import build_dataset, build_features, primary_signal

__all__ = ["MetaLabeler", "build_dataset", "build_features", "primary_signal"]
