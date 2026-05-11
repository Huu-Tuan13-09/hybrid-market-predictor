"""
backend/ml_models/predictor.py
================================
Market Predictor — XGBoost inference engine.

Responsibilities:
  1. Auto-discover the latest serialized model from data/models/.
  2. Load model + feature config via joblib.
  3. Accept today's OHLCV DataFrame, engineer features, run predict_proba().
  4. Return a structured dict: {p_up, p_down, direction, confidence, label}.

This output is consumed directly by the Quant Agent in LangGraph.

Design Principles:
  - OOP, Type Hinting, try/except on IO and inference.
  - Lazy loading: model is loaded once and cached for the lifetime of the object.
  - Loguru for structured logging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from loguru import logger
from xgboost import XGBClassifier

from backend.ml_models.feature_engineer import FeatureEngineer, FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Confidence Thresholds
# ---------------------------------------------------------------------------

# Maps P(UP) probability to direction + confidence label
_THRESHOLDS = [
    (0.65, "TĂNG",     "HIGH"),
    (0.55, "TĂNG",     "MEDIUM"),
    (0.45, "ĐI NGANG", "LOW"),
    (0.35, "GIẢM",     "MEDIUM"),
    (0.00, "GIẢM",     "HIGH"),
]

DEFAULT_MODEL_DIR = Path("data/models")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _classify(p_up: float) -> tuple[str, str]:
    """Map P(UP) to (direction, confidence) using threshold table."""
    for threshold, direction, confidence in _THRESHOLDS:
        if p_up >= threshold:
            return direction, confidence
    return "GIẢM", "HIGH"   # Fallback (p_up < 0)


def _find_latest_model(model_dir: Path) -> Optional[Path]:
    """
    Scan model_dir for .pkl files and return the most recently modified one.
    This avoids hardcoding model filenames and supports model rotation.
    """
    pkl_files = sorted(
        model_dir.glob("xgboost_vnindex_*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return pkl_files[0] if pkl_files else None


# ---------------------------------------------------------------------------
# Main Predictor
# ---------------------------------------------------------------------------

class MarketPredictor:
    """
    Loads the latest serialized XGBoost model and runs inference on today's
    OHLCV data to predict tomorrow's VN-Index direction.

    Usage:
        predictor = MarketPredictor()
        result    = predictor.predict(ohlcv_df)
        # result → {
        #     "p_up":       0.72,
        #     "p_down":     0.28,
        #     "direction":  "TĂNG",
        #     "confidence": "HIGH",
        #     "model_path": "data/models/xgboost_vnindex_20260506.pkl",
        # }
    """

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir      = Path(model_dir)
        self._model:        Optional[XGBClassifier] = None
        self._feature_cols: list[str]               = FEATURE_COLUMNS
        self._model_path:   Optional[Path]          = None
        logger.debug(f"MarketPredictor init | model_dir={self.model_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, ohlcv_df: pd.DataFrame) -> dict:
        """
        Run inference on the provided OHLCV DataFrame.

        Args:
            ohlcv_df: Cleaned OHLCV DataFrame from MarketDataFetcher.
                      Must contain at least 60 rows to compute all indicators.

        Returns:
            dict with keys: p_up, p_down, direction, confidence, model_path, features_snapshot.

        Raises:
            RuntimeError: If no trained model found or feature engineering fails.
        """
        self._ensure_model_loaded()

        fe         = FeatureEngineer()
        X_today    = fe.get_inference_row(ohlcv_df)

        logger.info(f"Running inference | model={self._model_path.name}")

        try:
            proba  = self._model.predict_proba(X_today)[0]   # shape (2,)
        except Exception as exc:
            logger.error(f"predict_proba failed: {exc}")
            raise RuntimeError(f"Inference failed: {exc}") from exc

        p_up       = float(round(proba[1], 4))
        p_down     = float(round(proba[0], 4))
        direction, confidence = _classify(p_up)

        result = {
            "p_up":              p_up,
            "p_down":            p_down,
            "direction":         direction,
            "confidence":        confidence,
            "model_path":        str(self._model_path),
            "features_snapshot": self._snapshot_features(X_today),
        }
        logger.success(
            f"Prediction → {direction} | P(UP)={p_up:.2%} | confidence={confidence}"
        )
        return result

    def reload_model(self) -> None:
        """Force reload the latest model from disk (useful after re-training)."""
        self._model      = None
        self._model_path = None
        self._ensure_model_loaded()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazy-load model on first call; subsequent calls are no-ops."""
        if self._model is not None:
            return

        model_path = _find_latest_model(self.model_dir)
        if model_path is None:
            raise RuntimeError(
                f"No trained model found in '{self.model_dir}'. "
                "Run 'python backend/ml_models/trainer.py' first."
            )

        logger.info(f"Loading model: {model_path}")
        try:
            self._model      = joblib.load(model_path)
            self._model_path = model_path
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from {model_path}: {exc}") from exc

        # Optionally load feature config to verify column alignment
        feat_cfg_path = model_path.parent / model_path.name.replace(
            "xgboost_vnindex_", "feature_config_"
        ).replace(".pkl", ".json")
        if feat_cfg_path.exists():
            try:
                cfg = json.loads(feat_cfg_path.read_text(encoding="utf-8"))
                self._feature_cols = cfg.get("features", FEATURE_COLUMNS)
                logger.debug(f"Feature config loaded: {len(self._feature_cols)} features")
            except Exception as exc:
                logger.warning(f"Could not load feature config: {exc} — using defaults")

        logger.success(f"Model loaded: {model_path.name}")

    def _snapshot_features(self, X_today: pd.DataFrame) -> dict:
        """
        Return a human-readable snapshot of today's key indicator values
        for display in the Streamlit UI and agent context.
        """
        row = X_today.iloc[0]
        return {
            "rsi_14":     round(float(row.get("rsi_14", 0)),    2),
            "macd":       round(float(row.get("macd", 0)),      4),
            "macd_diff":  round(float(row.get("macd_diff", 0)), 4),
            "bb_width":   round(float(row.get("bb_width", 0)),  4),
            "atr_14":     round(float(row.get("atr_14", 0)),    2),
            "adx":        round(float(row.get("adx", 0)),       2),
            "return_1d":  round(float(row.get("return_1d", 0)), 4),
            "return_5d":  round(float(row.get("return_5d", 0)), 4),
            "volume_ratio": round(float(row.get("volume_ratio", 0)), 2),
        }
