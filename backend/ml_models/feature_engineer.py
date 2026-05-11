"""
backend/ml_models/feature_engineer.py
=======================================
Feature Engineering — converts raw OHLCV DataFrame into an ML-ready feature matrix.

Uses the `ta` (Technical Analysis) library to compute 23+ indicators across
five groups: Momentum, Trend, Volatility, Volume, and Price Action.

Design Principles:
  - Zero data leakage: all indicators computed on past data only; target is
    shift(-1) so features at day T predict label at day T+1.
  - The last row (today) has features but no label — this is the inference row.
  - OOP with full Type Hinting.
  - try/except around every indicator group so one failure doesn't break all.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger

import ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    # Momentum
    "rsi_14", "stoch_k", "stoch_d",
    # Trend
    "macd", "macd_signal", "macd_diff",
    "ema_9", "ema_21", "sma_50", "adx",
    # Volatility
    "bb_upper", "bb_lower", "bb_width", "atr_14",
    # Volume
    "obv", "volume_ratio",
    # Price Action
    "return_1d", "return_5d", "return_20d", "high_low_pct",
    # Calendar
    "day_of_week", "month",
]

TARGET_COLUMN = "target"  # 1 = next day UP, 0 = next day DOWN or flat


# ---------------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Transforms a cleaned OHLCV DataFrame (from MarketDataFetcher)
    into a feature matrix + target vector suitable for XGBoost training.

    Usage (training):
        fe       = FeatureEngineer()
        feat_df  = fe.build_features(ohlcv_df)        # adds features + target
        X_train  = feat_df[FEATURE_COLUMNS].dropna()
        y_train  = feat_df[TARGET_COLUMN].dropna()

    Usage (inference — today's features):
        feat_df  = fe.build_features(ohlcv_df, add_target=False)
        X_today  = feat_df[FEATURE_COLUMNS].iloc[[-1]]  # last row only
    """

    def __init__(self) -> None:
        logger.debug("FeatureEngineer initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_features(
        self,
        df: pd.DataFrame,
        add_target: bool = True,
    ) -> pd.DataFrame:
        """
        Add all technical indicators and (optionally) the target column to df.

        Args:
            df:         Cleaned OHLCV DataFrame from MarketDataFetcher.
                        Must have columns: open, high, low, close, volume.
            add_target: If True, compute binary target (1=up, 0=down/flat).
                        Set False for inference where tomorrow is unknown.

        Returns:
            DataFrame with original OHLCV columns + feature columns (+ target).
        """
        df = df.copy()

        df = self._add_momentum(df)
        df = self._add_trend(df)
        df = self._add_volatility(df)
        df = self._add_volume(df)
        df = self._add_price_action(df)
        df = self._add_calendar(df)

        if add_target:
            df = self._add_target(df)

        logger.info(
            f"FeatureEngineer: built {len(FEATURE_COLUMNS)} features | "
            f"rows={len(df)} | NaN rows will be dropped by caller"
        )
        return df

    def get_clean_training_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Convenience method: build features, drop NaN rows, return (X, y).

        Returns:
            (X, y) tuple where X is the feature DataFrame and y is the target Series.
        """
        featured = self.build_features(df, add_target=True)
        featured = featured.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
        X = featured[FEATURE_COLUMNS]
        y = featured[TARGET_COLUMN].astype(int)
        logger.info(f"Training data: X={X.shape}, y distribution={y.value_counts().to_dict()}")
        return X, y

    def get_inference_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a single-row DataFrame with today's features (no target).
        This is the input for MarketPredictor.predict().
        """
        featured = self.build_features(df, add_target=False)
        latest   = featured[FEATURE_COLUMNS].iloc[[-1]]
        if latest.isnull().any().any():
            logger.warning("Inference row contains NaN — model may produce unreliable output")
        return latest

    # ------------------------------------------------------------------
    # Private — Indicator Groups
    # ------------------------------------------------------------------

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI and Stochastic Oscillator."""
        try:
            rsi              = RSIIndicator(close=df["close"], window=14)
            df["rsi_14"]     = rsi.rsi()

            stoch            = StochasticOscillator(
                high=df["high"], low=df["low"], close=df["close"],
                window=14, smooth_window=3
            )
            df["stoch_k"]    = stoch.stoch()
            df["stoch_d"]    = stoch.stoch_signal()
        except Exception as exc:
            logger.error(f"Momentum indicators failed: {exc}")
            for col in ["rsi_14", "stoch_k", "stoch_d"]:
                df[col] = np.nan
        return df

    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD, EMA, SMA, ADX."""
        try:
            macd_ind         = MACD(
                close=df["close"], window_slow=26, window_fast=12, window_sign=9
            )
            df["macd"]       = macd_ind.macd()
            df["macd_signal"]= macd_ind.macd_signal()
            df["macd_diff"]  = macd_ind.macd_diff()

            df["ema_9"]      = EMAIndicator(close=df["close"], window=9).ema_indicator()
            df["ema_21"]     = EMAIndicator(close=df["close"], window=21).ema_indicator()
            df["sma_50"]     = SMAIndicator(close=df["close"], window=50).sma_indicator()

            adx_ind          = ADXIndicator(
                high=df["high"], low=df["low"], close=df["close"], window=14
            )
            df["adx"]        = adx_ind.adx()
        except Exception as exc:
            logger.error(f"Trend indicators failed: {exc}")
            for col in ["macd", "macd_signal", "macd_diff", "ema_9", "ema_21", "sma_50", "adx"]:
                df[col] = np.nan
        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands and ATR."""
        try:
            bb               = BollingerBands(close=df["close"], window=20, window_dev=2)
            df["bb_upper"]   = bb.bollinger_hband()
            df["bb_lower"]   = bb.bollinger_lband()
            bb_mid           = bb.bollinger_mavg()
            df["bb_width"]   = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)

            atr              = AverageTrueRange(
                high=df["high"], low=df["low"], close=df["close"], window=14
            )
            df["atr_14"]     = atr.average_true_range()
        except Exception as exc:
            logger.error(f"Volatility indicators failed: {exc}")
            for col in ["bb_upper", "bb_lower", "bb_width", "atr_14"]:
                df[col] = np.nan
        return df

    def _add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBV and Volume Ratio."""
        try:
            obv              = OnBalanceVolumeIndicator(
                close=df["close"], volume=df["volume"]
            )
            df["obv"]        = obv.on_balance_volume()

            vma_20           = df["volume"].rolling(window=20).mean()
            df["volume_ratio"] = df["volume"] / vma_20.replace(0, np.nan)
        except Exception as exc:
            logger.error(f"Volume indicators failed: {exc}")
            for col in ["obv", "volume_ratio"]:
                df[col] = np.nan
        return df

    def _add_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return series and High-Low range."""
        try:
            df["return_1d"]    = df["close"].pct_change(1)
            df["return_5d"]    = df["close"].pct_change(5)
            df["return_20d"]   = df["close"].pct_change(20)
            df["high_low_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        except Exception as exc:
            logger.error(f"Price action features failed: {exc}")
            for col in ["return_1d", "return_5d", "return_20d", "high_low_pct"]:
                df[col] = np.nan
        return df

    def _add_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Day of week and month as categorical numeric features."""
        try:
            df["day_of_week"] = df.index.dayofweek.astype(float)   # 0=Mon … 4=Fri
            df["month"]       = df.index.month.astype(float)        # 1-12
        except Exception as exc:
            logger.error(f"Calendar features failed: {exc}")
            df["day_of_week"] = np.nan
            df["month"]       = np.nan
        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Binary target: 1 if next day close > today close, else 0.
        The last row will have NaN target (that's the inference day).
        """
        df[TARGET_COLUMN] = (df["close"].shift(-1) > df["close"]).astype(float)
        df.loc[df.index[-1], TARGET_COLUMN] = np.nan  # Explicit NaN for last row
        return df
