"""
Inference utilities for the frozen production model.

The dashboard and any future API should use this module rather than directly
calling training code or hardcoded model filenames.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.artifacts import load_artifact_registry, load_artifact_model
from src.config import (
    ALL_FEATURES,
    CLASS_NAMES,
    ENCODER_PATH,
    ORIGINAL_FEATURES,
    SCALER_PATH,
)
from src.preprocessing import engineer_features


@dataclass
class PredictionResult:
    label_index: int
    label_name: str
    confidence: float
    probabilities: np.ndarray
    inference_ms: float


class ProductionPredictor:
    """
    Frozen inference wrapper for one selected production model.

    It handles raw feature engineering, scaling, prediction, and batch scoring.
    """

    def __init__(self, model_name: str | None = None):
        self.registry = load_artifact_registry()
        self.scaler = joblib.load(SCALER_PATH)
        self.encoder = joblib.load(ENCODER_PATH)
        self.model = load_artifact_model(model_name)
        self.model_name = model_name or self.registry.get("production", {}).get("name", "production")
        self.class_names = list(self.encoder.classes_) if hasattr(self.encoder, "classes_") else CLASS_NAMES
        self.is_torch = not hasattr(self.model, "predict_proba")

        # Determine how many features the model expects (torch models store n_features in checkpoint)
        self._n_features = self._resolve_n_features(model_name)

    def _resolve_n_features(self, model_name: str | None) -> int:
        """Return the feature count the loaded model was trained on."""
        if not self.is_torch:
            return len(ALL_FEATURES)
        candidates = self.registry.get("candidates", [])
        name = model_name or self.registry.get("production", {}).get("name")
        for c in candidates:
            if c.get("name") == name:
                import torch
                try:
                    p = torch.load(c["model_path"], map_location="cpu", weights_only=False)
                    return int(p.get("n_features", len(ALL_FEATURES)))
                except Exception:
                    pass
        return len(ALL_FEATURES)

    @property
    def feature_names(self) -> list[str]:
        return list(ALL_FEATURES[:self._n_features])

    def validate_raw_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in ORIGINAL_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        out = df.copy()
        for col in ORIGINAL_FEATURES:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[ORIGINAL_FEATURES].isna().any().any():
            bad = [c for c in ORIGINAL_FEATURES if out[c].isna().any()]
            raise ValueError(f"Non-numeric or missing values found in: {bad}")
        return out

    def transform_raw(self, df_raw: pd.DataFrame) -> np.ndarray:
        df = self.validate_raw_frame(df_raw)
        df = engineer_features(df)
        X = df[ALL_FEATURES].to_numpy(dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        # Slice to the feature count the model was trained on
        return X_scaled[:, :self._n_features]

    def transform_row(self, row: dict[str, Any]) -> np.ndarray:
        return self.transform_raw(pd.DataFrame([row]))

    def _predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.is_torch:
            import torch
            xt = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                logits = self.model(xt)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
            return proba
        return self.model.predict_proba(X_scaled)

    def predict(self, X_scaled: np.ndarray) -> PredictionResult:
        t0 = time.perf_counter()
        proba = self._predict_proba(X_scaled)
        pred_idx = int(np.argmax(proba[0]))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return PredictionResult(
            label_index=pred_idx,
            label_name=self.class_names[pred_idx],
            confidence=float(proba[0, pred_idx]),
            probabilities=proba[0],
            inference_ms=float(elapsed_ms),
        )

    def predict_row(self, row: dict[str, Any]) -> PredictionResult:
        return self.predict(self.transform_row(row))

    def predict_frame(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self.transform_raw(df_raw)
        proba = self._predict_proba(X_scaled)
        preds = proba.argmax(axis=1)
        result = df_raw.copy().reset_index(drop=True)
        result["predicted_class"] = [self.class_names[i] for i in preds]
        result["predicted_index"] = preds
        result["confidence"] = proba.max(axis=1)
        result["inference_ms"] = np.nan
        return result
