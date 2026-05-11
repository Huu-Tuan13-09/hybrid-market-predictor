"""
backend/ml_models/trainer.py
==============================
XGBoost Model Trainer — trains and serializes the VN-Index direction classifier.

Pipeline:
  1. Pull OHLCV data via MarketDataFetcher.
  2. Engineer features via FeatureEngineer.
  3. TimeSeriesSplit cross-validation (5 folds, no shuffle).
  4. Train XGBClassifier with early stopping on the final fold.
  5. Evaluate on hold-out set → log Accuracy, AUC-ROC, Classification Report.
  6. Serialize model (.pkl) + metadata (.json) to data/models/.

Can be run as a standalone script:
    python backend/ml_models/trainer.py

Design Principles:
  - OOP, Type Hinting, try/except on IO and training steps.
  - Loguru for structured output.
  - Reproducible via fixed random_state=42.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from backend.scraper.market_data import MarketDataFetcher
from backend.ml_models.feature_engineer import FeatureEngineer, FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR  = Path("data/models")
# DEFAULT_TICKER     = "^VNINDEX"
DEFAULT_TICKER     = "VNM"
DEFAULT_LOOKBACK   = 1500         # ~6 years trading days
N_SPLITS           = 5            # TimeSeriesSplit folds
RANDOM_STATE       = 42


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

XGBOOST_PARAMS: dict = {
    "n_estimators":      1000,
    "max_depth":         5,
    "learning_rate":     0.01,
    "subsample":         0.7,
    "colsample_bytree":  0.7,
    "min_child_weight":  5,
    "gamma":             0.1,
    "eval_metric":       "logloss",
    "random_state":      RANDOM_STATE,
    "tree_method":       "hist",   # Fast histogram-based algorithm
    "n_jobs":            -1,
}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Orchestrates the full XGBoost training pipeline.

    Usage:
        trainer = ModelTrainer(ticker="^VNINDEX", lookback_days=504)
        metadata = trainer.train()
        print(metadata)  # {"model_path": "...", "accuracy": 0.58, ...}
    """

    def __init__(
        self,
        ticker:        str  = DEFAULT_TICKER,
        lookback_days: int  = DEFAULT_LOOKBACK,
        model_dir:     Path = DEFAULT_MODEL_DIR,
    ) -> None:
        self.ticker        = ticker
        self.lookback_days = lookback_days
        self.model_dir     = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ModelTrainer init | ticker={ticker} | lookback={lookback_days}d")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """
        Execute the full training pipeline.

        Returns:
            dict with keys: model_path, feature_config_path, metadata_path,
                            accuracy, auc_roc, n_training_rows, trained_at.
        """
        # Step 1 — Data
        X, y = self._load_data()

        # Step 2 — Cross-validation report
        cv_scores = self._cross_validate(X, y)

        # Step 3 — Final training on all data (minus last 20% for evaluation)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = self._fit_model(X_train, y_train, X_test, y_test)

        # Step 4 — Final evaluation
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        accuracy    = accuracy_score(y_test, y_pred)
        auc         = roc_auc_score(y_test, y_pred_prob)

        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['DOWN', 'UP'])}")
        logger.success(f"Final Accuracy={accuracy:.4f} | AUC-ROC={auc:.4f}")

        # Step 5 — Serialize
        paths = self._save_artifacts(model, accuracy, auc, cv_scores, len(X))
        return paths

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Fetch OHLCV and build feature matrix."""
        logger.info("Loading market data…")
        fetcher = MarketDataFetcher(ticker=self.ticker, lookback_days=self.lookback_days)
        result  = fetcher.fetch()

        logger.info("Engineering features…")
        fe    = FeatureEngineer()
        X, y  = fe.get_clean_training_data(result.df)

        if len(X) < 100:
            raise RuntimeError(
                f"Insufficient training data: only {len(X)} rows after feature engineering. "
                "Increase lookback_days or check ticker symbol."
            )
        return X, y

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> list[float]:
        """TimeSeriesSplit CV — returns per-fold accuracy scores."""
        logger.info(f"Running {N_SPLITS}-fold TimeSeriesSplit CV…")
        tscv   = TimeSeriesSplit(n_splits=N_SPLITS)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Compute class weight for this fold
            pos_ratio          = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            params             = XGBOOST_PARAMS.copy()
            params["scale_pos_weight"] = pos_ratio

            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, verbose=False)
            acc = accuracy_score(y_val, model.predict(X_val))
            scores.append(acc)
            logger.debug(f"  Fold {fold}: accuracy={acc:.4f}")

        mean_acc = float(np.mean(scores))
        std_acc  = float(np.std(scores))
        logger.info(f"CV Result: mean_accuracy={mean_acc:.4f} ± {std_acc:.4f}")
        return scores

    def _fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
    ) -> XGBClassifier:
        """Fit final XGBClassifier with early stopping on validation set."""
        pos_ratio          = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        params             = XGBOOST_PARAMS.copy()
        params["scale_pos_weight"] = pos_ratio
        params["early_stopping_rounds"] = 50

        model = XGBClassifier(**params)
        logger.info("Training final XGBoost model with early stopping…")

        try:
            model.fit(
                X_train, y_train,
                eval_set = [(X_val, y_val)],
                verbose  = False,
            )
        except Exception as exc:
            logger.error(f"XGBoost training failed: {exc}")
            raise

        best_iter = model.best_iteration if hasattr(model, "best_iteration") else "N/A"
        logger.success(f"Training complete | best_iteration={best_iter}")
        return model

    def _save_artifacts(
        self,
        model:        XGBClassifier,
        accuracy:     float,
        auc:          float,
        cv_scores:    list[float],
        n_rows:       int,
    ) -> dict:
        """Serialize model + metadata to disk."""
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_fname = f"xgboost_vnindex_{timestamp}.pkl"
        feat_fname  = f"feature_config_{timestamp}.json"
        meta_fname  = f"training_metadata_{timestamp}.json"

        model_path  = self.model_dir / model_fname
        feat_path   = self.model_dir / feat_fname
        meta_path   = self.model_dir / meta_fname

        # Save model
        try:
            joblib.dump(model, model_path)
            logger.success(f"Model saved: {model_path}")
        except Exception as exc:
            raise RuntimeError(f"Failed to save model: {exc}") from exc

        # Save feature config
        feature_config = {"features": FEATURE_COLUMNS, "version": timestamp}
        feat_path.write_text(json.dumps(feature_config, indent=2), encoding="utf-8")

        # Save training metadata
        metadata = {
            "ticker":           self.ticker,
            "lookback_days":    self.lookback_days,
            "n_training_rows":  n_rows,
            "accuracy":         round(accuracy, 4),
            "auc_roc":          round(auc, 4),
            "cv_scores":        [round(s, 4) for s in cv_scores],
            "cv_mean":          round(float(np.mean(cv_scores)), 4),
            "xgboost_params":   XGBOOST_PARAMS,
            "trained_at":       datetime.now().isoformat(),
            "model_path":       str(model_path),
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.success(f"Metadata saved: {meta_path}")

        return {
            "model_path":           str(model_path),
            "feature_config_path":  str(feat_path),
            "metadata_path":        str(meta_path),
            "accuracy":             round(accuracy, 4),
            "auc_roc":              round(auc, 4),
            "trained_at":           metadata["trained_at"],
        }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logger.add(sys.stdout, level="INFO")

    ticker        = os.getenv("VNINDEX_TICKER", DEFAULT_TICKER)
    lookback_days = int(os.getenv("LOOKBACK_DAYS", DEFAULT_LOOKBACK))
    model_dir     = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_DIR)))

    trainer  = ModelTrainer(ticker=ticker, lookback_days=lookback_days, model_dir=model_dir)
    result   = trainer.train()
    logger.info(f"Training pipeline complete:\n{json.dumps(result, indent=2)}")
