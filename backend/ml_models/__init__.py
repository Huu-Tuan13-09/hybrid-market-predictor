"""
backend/ml_models/__init__.py
ML Models package — XGBoost Time-Series Forecasting Pipeline.

Exports:
  - FeatureEngineer : Technical indicator feature creator (ta library)
  - ModelTrainer    : XGBoost training with TimeSeriesSplit
  - MarketPredictor : Inference engine returning P(up) / P(down)
"""

from .feature_engineer import FeatureEngineer
from .trainer import ModelTrainer
from .predictor import MarketPredictor

__all__ = ["FeatureEngineer", "ModelTrainer", "MarketPredictor"]
