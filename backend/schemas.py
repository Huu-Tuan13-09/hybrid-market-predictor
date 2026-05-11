"""
backend/schemas.py
====================
Pydantic v2 request/response models for the FastAPI API.

All models use strict typing and Field() for documentation
and validation constraints.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for POST /predict"""
    ticker:        str = Field(default="^VNINDEX", description="Yahoo Finance ticker symbol")
    lookback_days: int = Field(default=504, ge=60, le=1260, description="Days of historical data (60–1260)")
    analysis_date: Optional[str] = Field(default=None, description="Override analysis date (YYYY-MM-DD)")


class TrainRequest(BaseModel):
    """Request body for POST /train"""
    ticker:        str = Field(default="^VNINDEX")
    lookback_days: int = Field(default=504, ge=100, le=1260)


class RagIndexRequest(BaseModel):
    """Request body for POST /rag/index — index raw text into ChromaDB"""
    text:       str   = Field(..., min_length=50, description="Text to index")
    source:     str   = Field(..., description="Source identifier (filename or URL)")
    collection: Literal["macro_reports", "market_news", "company_filings"] = "macro_reports"
    doc_date:   Optional[str] = Field(default=None, description="Document date YYYY-MM-DD")


# ---------------------------------------------------------------------------
# Response Sub-models
# ---------------------------------------------------------------------------

class AgentReportResponse(BaseModel):
    """Single agent's report summary for API response."""
    agent_name:      str
    recommendation:  str
    confidence:      int
    summary:         str
    raw_report:      dict[str, Any]   # Full JSON from the agent


class WeightsUsed(BaseModel):
    """Dynamic weights applied by CIO Agent."""
    quant:     float
    sentiment: float
    macro:     float


# ---------------------------------------------------------------------------
# Primary Response Models
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Response for POST /predict"""
    ticker:          str
    prediction_date: str
    direction:       Literal["TĂNG", "GIẢM", "ĐI NGANG"]
    confidence_score: float   = Field(ge=0.0, le=1.0)
    action:          str      = Field(description="MUA | BÁN | CHỜ ĐỢI")

    # ML output
    ml_probability:  dict[str, float]   # {p_up, p_down}
    ml_direction:    str
    ml_confidence:   str

    # Agent outputs
    weight_regime:   str
    weights_used:    WeightsUsed
    weighted_score:  float
    consensus:       str
    key_signals:     list[str]
    risk_factors:    list[str]
    cio_reasoning:   str
    stop_loss_note:  str
    agent_reports:   list[AgentReportResponse]

    # Meta
    articles_scraped:   int
    rag_context_chars:  int
    error_messages:     list[str]
    processing_time_ms: int


class TrainResponse(BaseModel):
    """Response for POST /train"""
    status:       str
    model_path:   str
    accuracy:     float
    auc_roc:      float
    trained_at:   str
    message:      str


class MarketDataResponse(BaseModel):
    """Response for GET /market-data"""
    ticker:       str
    date:         str
    close:        float
    open:         float
    high:         float
    low:          float
    volume:       int
    return_1d:    float
    indicators:   dict[str, float]
    ohlcv_history: list[dict]


class NewsResponse(BaseModel):
    """Response for GET /news"""
    articles:     list[dict]
    total:        int
    scraped_at:   str


class RagQueryResponse(BaseModel):
    """Response for GET /rag/query"""
    query:      str
    collection: str
    results:    list[dict]
    total:      int


class HealthResponse(BaseModel):
    """Response for GET /health"""
    status:           str
    model_loaded:     bool
    chromadb_counts:  dict[str, int]
    version:          str = "1.0.0"


class ErrorResponse(BaseModel):
    """Standard error response envelope."""
    error:   str
    detail:  Optional[str] = None
    code:    int
