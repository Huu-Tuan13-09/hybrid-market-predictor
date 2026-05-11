"""
backend/scraper/market_data.py
================================
Market Data Fetcher — multi-source OHLCV downloader with 4-layer fallback.

Fallback Strategy (VN indices):
  1. vnstock3   — pip install vnstock3  (preferred, actively maintained)
  2. vnstock    — pip install vnstock   (legacy, different API)
  3. SSI public REST API — no auth needed, works from Docker
  4. yfinance + curl_cffi Chrome TLS fingerprint — last resort

Design Principles:
  - OOP with full Type Hinting.
  - try/except on every external call with informative error messages.
  - Loguru for structured logging (no bare print statements).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

# ---------------------------------------------------------------------------
# Try curl_cffi for TLS-fingerprint spoofing (bypasses Docker IP blocks on Yahoo)
# ---------------------------------------------------------------------------
try:
    from curl_cffi import requests as cffi_requests
    _HAS_CURL_CFFI = True
    logger.debug("curl_cffi available — will use Chrome TLS fingerprint for yfinance")
except ImportError:
    import requests as cffi_requests  # type: ignore[assignment]
    _HAS_CURL_CFFI = False
    logger.warning(
        "curl_cffi not installed — falling back to plain requests. "
        "Yahoo Finance may block Docker IPs. Install: pip install curl_cffi"
    )

# ---------------------------------------------------------------------------
# Try vnstock / vnstock3 (actively maintained, preferred)
# Note: As of late 2024, vnstock3 merged back into 'vnstock' 4.x
# ---------------------------------------------------------------------------
_HAS_VNSTOCK3 = False
try:
    # Try the new unified API first
    from vnstock import Vnstock as Vnstock3
    _HAS_VNSTOCK3 = True
    logger.debug("vnstock (unified) available — will use as Layer 1")
except (ImportError, AttributeError):
    try:
        # Fallback to older vnstock3 package
        from vnstock3 import Vnstock as Vnstock3
        _HAS_VNSTOCK3 = True
        logger.debug("vnstock3 available — will use as Layer 1")
    except ImportError:
        logger.debug("vnstock (unified/v3) not installed")

# ---------------------------------------------------------------------------
# Try vnstock legacy (different API surface)
# ---------------------------------------------------------------------------
_HAS_VNSTOCK_LEGACY = False
try:
    # Legacy vnstock uses a module-level function
    from vnstock import stock_historical_data as _vnstock_legacy_fetch  # type: ignore
    _HAS_VNSTOCK_LEGACY = True
    logger.debug("vnstock (legacy) available — will use as Layer 2")
except (ImportError, AttributeError):
    logger.debug("vnstock (legacy) not installed or incompatible")



def _make_yf_session():
    """
    Create a session that mimics a real Chrome browser at the TLS level.
    curl_cffi impersonates Chrome's TLS fingerprint, bypassing Yahoo Finance's
    bot detection that blocks standard data-center / Docker container IPs.
    """
    if _HAS_CURL_CFFI:
        # impersonate="chrome" sets the TLS fingerprint + HTTP/2 headers
        session = cffi_requests.Session(impersonate="chrome")
    else:
        import requests as _requests
        session = _requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })
    return session


# VN tickers → vnstock symbol mapping (indices only)
_VN_INDEX_MAP: dict[str, str] = {
    "^VNINDEX": "VNINDEX",
    "^VN30":    "VN30",
}

# Ticker fallback aliases — yfinance sometimes uses different symbols
_TICKER_ALIASES: dict[str, list[str]] = {
    "^VNINDEX": ["^VNINDEX", "VNINDEX", "^VNI"],
    "^VN30":    ["^VN30", "VN30"],
}

# SSI public API endpoints for VN index OHLCV (no auth required)
# Ref: https://iboard-query.ssi.com.vn/
_SSI_INDEX_CODE: dict[str, str] = {
    "^VNINDEX": "VNINDEX",
    "^VN30":    "VN30",
}


def _is_vn_stock_ticker(ticker: str) -> bool:
    """
    Returns True if `ticker` looks like a bare Vietnamese stock code
    (2-4 uppercase letters, no caret, no dot suffix).
    Examples: VNM, VIC, HPG, SSI, FPT, TCB, VHM.
    Excludes index tickers like ^VNINDEX and already-suffixed tickers like VNM.VN.
    """
    import re
    return bool(re.fullmatch(r"[A-Z0-9]{2,5}", ticker)) and ticker not in _VN_INDEX_MAP.values()


def _yf_aliases_for_vn_stock(ticker: str) -> list[str]:
    """
    Build yfinance symbol candidates for a bare VN stock ticker.
    HoSE stocks trade as 'VNM.VN', HNX as 'VNM.HN' on Yahoo Finance.
    We try both suffixes plus the bare symbol as a last-ditch attempt.
    """
    return [f"{ticker}.VN", f"{ticker}.HN", ticker]


# ---------------------------------------------------------------------------
# Data class for raw fetch results (acts as a lightweight DTO)
# ---------------------------------------------------------------------------

@dataclass
class OHLCVResult:
    """Lightweight Data Transfer Object returned by MarketDataFetcher."""

    ticker: str
    df: pd.DataFrame                        # Cleaned OHLCV DataFrame
    start_date: str
    end_date: str
    num_rows: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_rows = len(self.df)

    def to_summary_dict(self) -> dict:
        """Return a compact dict with latest bar statistics for downstream agents."""
        if self.df.empty:
            return {}
        latest = self.df.iloc[-1]
        prev   = self.df.iloc[-2] if len(self.df) > 1 else latest
        return {
            "ticker":       self.ticker,
            "date":         str(self.df.index[-1].date()),
            "close":        round(float(latest["close"]), 2),
            "open":         round(float(latest["open"]),  2),
            "high":         round(float(latest["high"]),  2),
            "low":          round(float(latest["low"]),   2),
            "volume":       int(latest["volume"]),
            "return_1d":    round((float(latest["close"]) - float(prev["close"])) / float(prev["close"]) * 100, 4),
            "num_rows":     self.num_rows,
        }


# ---------------------------------------------------------------------------
# Main Fetcher Class
# ---------------------------------------------------------------------------

class MarketDataFetcher:
    """
    Downloads and cleans OHLCV market data from Yahoo Finance via yfinance.

    Usage:
        fetcher = MarketDataFetcher(ticker="^VNINDEX", lookback_days=504)
        result  = fetcher.fetch()
        df      = result.df
    """

    # Expected column mapping from yfinance multi-level columns to flat names
    _COLUMN_MAP: dict[str, str] = {
        "Open":     "open",
        "High":     "high",
        "Low":      "low",
        "Close":    "close",
        "Volume":   "volume",
    }

    def __init__(
        self,
        ticker:        str = "^VNINDEX",
        lookback_days: int = 504,
        auto_adjust:   bool = True,
    ) -> None:
        """
        Args:
            ticker:        Yahoo Finance ticker symbol.
            lookback_days: Number of TRADING days to look back.
                           Internally converted to calendar days (×1.4) to ensure
                           enough calendar days cover the requested trading days.
                           504 trading days ≈ 730 calendar days (~2 years).
            auto_adjust:   Whether yfinance should auto-adjust prices for splits/dividends.
        """
        self.ticker        = ticker
        # Convert trading days → calendar days (trading days ≈ 71% of calendar days)
        self.lookback_days = int(lookback_days * 1.45)
        self.auto_adjust   = auto_adjust
        self._session      = _make_yf_session()
        logger.debug(f"MarketDataFetcher init | ticker={ticker} | lookback={lookback_days}td → {self.lookback_days}cd")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self) -> OHLCVResult:
        """
        Execute the full fetch-clean-validate pipeline.

        Returns:
            OHLCVResult with a cleaned DataFrame and metadata.

        Raises:
            RuntimeError: If the download returns an empty DataFrame.
        """
        end_date   = pd.Timestamp.now(tz="UTC")
        start_date = end_date - pd.Timedelta(days=self.lookback_days)

        logger.info(f"Fetching {self.ticker} | {start_date.date()} → {end_date.date()}")

        raw_df = self._download(start_date, end_date)
        clean_df = self._clean(raw_df)

        result = OHLCVResult(
            ticker     = self.ticker,
            df         = clean_df,
            start_date = str(start_date.date()),
            end_date   = str(end_date.date()),
        )
        logger.success(
            f"Fetched {result.num_rows} rows for {self.ticker} "
            f"({result.start_date} → {result.end_date})"
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Download OHLCV data with a 4-layer fallback strategy:
          1. vnstock3  — pip install vnstock3 (preferred, actively maintained)
                         handles both VN indices AND individual stocks (HoSE/HNX)
          2. vnstock   — pip install vnstock  (legacy API, different method names)
                         handles both VN indices AND individual stocks
          3. SSI public REST API — no auth, works from Docker
                         handles indices (VNINDEX/VN30) AND individual stocks
          4. yfinance + curl_cffi Chrome TLS fingerprint (last resort)
                         for VN stocks: tries 'VNM.VN' then 'VNM.HN' suffixes
        """
        start_str = start.strftime("%Y-%m-%d")
        end_str   = end.strftime("%Y-%m-%d")
        last_exc: Exception = RuntimeError("No layers attempted")

        is_vn_index = self.ticker in _VN_INDEX_MAP
        is_vn_stock = _is_vn_stock_ticker(self.ticker)

        # ── Layer 1: vnstock3 (preferred — handles indices AND stocks) ─────────
        if _HAS_VNSTOCK3 and (is_vn_index or is_vn_stock):
            vn_symbol = _VN_INDEX_MAP.get(self.ticker, self.ticker)
            try:
                logger.debug(f"[Layer 1] Trying vnstock3 for {vn_symbol}")
                stock = Vnstock3().stock(symbol=vn_symbol, source="KBS")
                df_vn = stock.quote.history(
                    start=start_str,
                    end=end_str,
                    interval="1D",
                )
                if df_vn is not None and not df_vn.empty:
                    logger.debug(f"[Layer 1] vnstock3 success: {len(df_vn)} rows")
                    return self._normalize_vnstock_df(df_vn)
                logger.warning(f"[Layer 1] vnstock3 returned empty for {vn_symbol}")
            except Exception as exc:
                last_exc = exc
                logger.warning(f"[Layer 1] vnstock3 failed for {vn_symbol}: {exc}")

        # ── Layer 2: vnstock legacy (handles indices AND stocks) ───────────────
        if _HAS_VNSTOCK_LEGACY and (is_vn_index or is_vn_stock):
            vn_symbol = _VN_INDEX_MAP.get(self.ticker, self.ticker)
            data_type = "index" if is_vn_index else "stock"
            try:
                logger.debug(f"[Layer 2] Trying vnstock (legacy) for {vn_symbol} type={data_type}")
                df_vn = _vnstock_legacy_fetch(
                    symbol=vn_symbol,
                    start_date=start_str,
                    end_date=end_str,
                    resolution="1D",
                    type=data_type,
                    beautify=True,
                    decor=False,
                    source="VNDIRECT",
                )
                if df_vn is not None and not df_vn.empty:
                    logger.debug(f"[Layer 2] vnstock legacy success: {len(df_vn)} rows")
                    return self._normalize_vnstock_df(df_vn)
                logger.warning(f"[Layer 2] vnstock legacy returned empty for {vn_symbol}")
            except Exception as exc:
                last_exc = exc
                logger.warning(f"[Layer 2] vnstock legacy failed for {vn_symbol}: {exc}")

        # ── Layer 3: SSI public REST API (no auth, Docker-friendly) ───────────
        # Works for indices AND individual stocks
        ssi_symbol = _SSI_INDEX_CODE.get(self.ticker, self.ticker if is_vn_stock else None)
        if ssi_symbol is not None:
            try:
                logger.debug(f"[Layer 3] Trying SSI API for {ssi_symbol}")
                df_ssi = self._fetch_from_ssi(ssi_symbol, start_str, end_str)
                if df_ssi is not None and not df_ssi.empty:
                    logger.debug(f"[Layer 3] SSI API success: {len(df_ssi)} rows")
                    return df_ssi
                logger.warning("[Layer 3] SSI API returned empty data")
            except Exception as exc:
                last_exc = exc
                logger.warning(f"[Layer 3] SSI API failed: {exc}")

        # ── Layer 4: yfinance with curl_cffi TLS fingerprint ──────────────────
        # For bare VN stock tickers (e.g. VNM), try Yahoo Finance suffixes .VN / .HN
        if is_vn_stock and self.ticker not in _TICKER_ALIASES:
            aliases = _yf_aliases_for_vn_stock(self.ticker)
        else:
            aliases = _TICKER_ALIASES.get(self.ticker, [self.ticker])

        for ticker_attempt in aliases:
            try:
                logger.debug(f"[Layer 4] Trying yfinance ticker: {ticker_attempt}")
                df = yf.download(
                    tickers     = ticker_attempt,
                    start       = start_str,
                    end         = end_str,
                    auto_adjust = self.auto_adjust,
                    progress    = False,
                    threads     = False,
                    session     = self._session,
                )
                if df is not None and not df.empty:
                    logger.debug(f"[Layer 4] yfinance success with alias: {ticker_attempt}")
                    return df
                logger.warning(f"[Layer 4] yfinance empty for alias: {ticker_attempt}")
            except Exception as exc:
                last_exc = exc
                logger.warning(f"[Layer 4] yfinance failed for {ticker_attempt}: {exc}")

        # All layers failed
        raise RuntimeError(
            f"All 4 data sources failed for ticker '{self.ticker}'.\n"
            f"Last error: {last_exc}\n"
            "Checklist:\n"
            "  1. pip install vnstock3       — best source for VN stocks & indices\n"
            "  2. pip install curl_cffi      — bypasses Yahoo Finance Docker IP blocks\n"
            "  3. SSI API unreachable?       — check Docker DNS / network\n"
            "  4. Ticker symbol valid?       — VN stocks: use bare code (VNM, VIC)\n"
            "                                  VN indices: use '^VNINDEX' or 'VNINDEX'\n"
            "                                  Yahoo Finance: VNM.VN (HoSE) / VNM.HN (HNX)"
        )

    # ------------------------------------------------------------------
    # Internal helpers for data normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_vnstock_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize a DataFrame returned by vnstock/vnstock3 into the same
        column format as yfinance (Open, High, Low, Close, Volume).

        vnstock may return columns named in various ways:
          - lowercase: 'open', 'high', 'low', 'close', 'volume'
          - with date col: 'time', 'date', 'tradingDate', 'TradingDate'
        """
        df = df.copy()
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(str(c) for c in col).strip("_") for col in df.columns]

        # Lowercase all column names for consistent matching
        df.columns = [c.lower() for c in df.columns]

        # Identify and set the date/time column as index
        date_candidates = ("time", "date", "tradingdate", "trading_date")
        date_col = next((c for c in df.columns if c in date_candidates), None)
        if date_col and date_col != df.index.name:
            df = df.set_index(date_col)
        df.index = pd.to_datetime(df.index)

        # Rename lowercase → Title-case (yfinance convention)
        rename_map = {
            "open":   "Open",
            "high":   "High",
            "low":    "Low",
            "close":  "Close",
            "volume": "Volume",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return df

    @staticmethod
    def _fetch_from_ssi(
        index_code: str,
        start_str: str,
        end_str: str,
    ) -> pd.DataFrame:
        """
        Fetch index OHLCV from SSI's public iBoard API.
        No authentication required — works reliably from Docker.

        Endpoint: GET https://iboard-query.ssi.com.vn/v2/stock/bars-long-term
        """
        import requests
        from datetime import datetime

        url = "https://iboard-query.ssi.com.vn/v2/stock/bars-long-term"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Origin":  "https://iboard.ssi.com.vn",
            "Referer": "https://iboard.ssi.com.vn/",
        }

        # Convert date strings to Unix timestamps (seconds)
        start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp())
        end_ts   = int(datetime.strptime(end_str,   "%Y-%m-%d").timestamp())

        params = {
            "symbol":     index_code,
            "resolution": "1D",
            "from":       start_ts,
            "to":         end_ts,
        }

        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        # SSI returns {"data": {"t": [...], "o": [...], "h": [...], "l": [...], "c": [...], "v": [...]}}
        bars = data.get("data", {})
        timestamps = bars.get("t", [])
        if not timestamps:
            return pd.DataFrame()

        df = pd.DataFrame({
            "Open":   bars.get("o", []),
            "High":   bars.get("h", []),
            "Low":    bars.get("l", []),
            "Close":  bars.get("c", []),
            "Volume": bars.get("v", []),
        }, index=pd.to_datetime(timestamps, unit="s", utc=True))

        df.index.name = "Date"
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize and validate the raw DataFrame.

        Steps:
          1. Flatten multi-level columns if present (yfinance sometimes returns them).
          2. Rename to lowercase standard column names.
          3. Ensure DatetimeIndex is UTC-aware.
          4. Drop rows with close == 0 or volume == 0 (exchange holidays / bad data).
          5. Forward-fill remaining NaN gaps (e.g., missing days).
          6. Drop any residual NaN rows that cannot be filled.
          7. Sort ascending by date.
        """
        # Step 1 — Flatten multi-level columns (e.g., ("Close", "^VNINDEX") → "Close")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Step 2 — Rename to lowercase standard names
        df = df.rename(columns=self._COLUMN_MAP)
        required_cols = list(self._COLUMN_MAP.values())
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Downloaded DataFrame missing expected columns: {missing}")
        df = df[required_cols].copy()

        # Step 3 — Normalize DatetimeIndex timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Step 4 — Drop clearly invalid rows
        before = len(df)
        df = df[(df["close"] > 0) & (df["volume"] > 0)]
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with close=0 or volume=0")

        # Step 5 — Forward-fill gaps (holiday / missing session)
        df = df.ffill()

        # Step 6 — Drop residual NaN
        df = df.dropna()

        # Step 7 — Sort ascending
        df = df.sort_index()

        if df.empty:
            raise RuntimeError("DataFrame is empty after cleaning. Check source data quality.")

        logger.debug(f"Cleaned DataFrame: {len(df)} rows, columns={list(df.columns)}")
        return df

    # ------------------------------------------------------------------
    # Convenience method — latest bar only (for quick health checks)
    # ------------------------------------------------------------------

    def fetch_latest(self) -> Optional[dict]:
        """
        Fetch and return only the latest trading bar as a dict.
        Useful for lightweight health checks and UI status banners.
        """
        try:
            result = self.fetch()
            return result.to_summary_dict()
        except RuntimeError as exc:
            logger.error(f"fetch_latest failed: {exc}")
            return None
