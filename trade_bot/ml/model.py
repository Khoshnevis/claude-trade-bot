"""Meta-label classifier wrapper (scikit-learn).

Kept deliberately simple and regularized -- on low signal-to-noise data a
complex model just overfits. Gradient boosting with shallow trees is the
default; the out-of-sample evaluation is the safeguard that catches overfit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import FEATURE_COLUMNS


class MetaLabeler:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.columns = FEATURE_COLUMNS + ["side"]
        # A regularized LINEAR model: on low signal-to-noise data it generalizes
        # far better than gradient boosting (which memorized the training set,
        # AUC ~1.0, and was worse than random out-of-sample).
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.3, max_iter=2000, class_weight="balanced",
            )),
        ])
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X[self.columns], y)
        self._fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.columns])[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        from sklearn.metrics import accuracy_score, roc_auc_score
        p = self.predict_proba(X)
        out = {"n": len(y), "base_rate": float(y.mean())}
        try:
            out["auc"] = float(roc_auc_score(y, p))
        except ValueError:
            out["auc"] = float("nan")  # single-class fold
        out["accuracy"] = float(accuracy_score(y, (p >= 0.5).astype(int)))
        return out

    def save(self, path: str | Path) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "columns": self.columns}, path)

    @classmethod
    def load(cls, path: str | Path) -> "MetaLabeler":
        import joblib
        obj = cls.__new__(cls)
        data = joblib.load(path)
        obj.model = data["model"]
        obj.columns = data["columns"]
        obj._fitted = True
        return obj
