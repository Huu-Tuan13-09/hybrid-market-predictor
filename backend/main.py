"""
backend/main.py
================
FastAPI application entry point — registers all routes, middleware,
and startup/shutdown lifecycle events.

Endpoints:
  GET  /health          → System health check
  POST /predict         → Full Hybrid AI analysis pipeline
  POST /train           → Re-train XGBoost model
  GET  /market-data     → Latest OHLCV + indicators
  GET  /news            → Crawl and return latest news
  GET  /rag/query       → Direct ChromaDB semantic query
  POST /rag/index       → Index raw text into ChromaDB

Usage (local dev):
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.config import get_settings
from backend.schemas import (
    AgentReportResponse,
    ErrorResponse,
    HealthResponse,
    MarketDataResponse,
    NewsResponse,
    PredictRequest,
    PredictionResponse,
    RagIndexRequest,
    RagQueryResponse,
    TrainRequest,
    TrainResponse,
    WeightsUsed,
)

# ---------------------------------------------------------------------------
# Lifespan — startup/shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize expensive resources on startup; clean up on shutdown."""
    logger.info("🚀 Hybrid AI Market Predictor — Backend starting up…")
    settings = get_settings()

    # Pre-warm the LangGraph (import & compile graph on startup)
    try:
        from backend.agents.graph import get_graph
        get_graph()
        logger.success("✅ LangGraph compiled and cached")
    except Exception as exc:
        logger.warning(f"⚠️  LangGraph pre-warm failed (will retry on first request): {exc}")

    yield  # Application runs here

    logger.info("🛑 Backend shutting down…")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

settings = get_settings()

app = FastAPI(
    title       = "Hybrid AI Market Predictor — Backend API",
    description = (
        "VN-Index forecasting system combining XGBoost time-series ML "
        "with LangGraph Multi-Agent analysis (Llama 3.3 via Groq)."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

# CORS — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins     = [o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health: model availability, ChromaDB status."""
    model_loaded = False
    chromadb_counts: dict[str, int] = {}

    try:
        from backend.ml_models.predictor import _find_latest_model
        model_loaded = _find_latest_model(Path(settings.model_path)) is not None
    except Exception:
        pass

    try:
        from backend.rag_engine.indexer import DocumentIndexer
        indexer = DocumentIndexer(chroma_path=Path(settings.chroma_db_path))
        chromadb_counts = indexer.get_collection_stats()
    except Exception:
        pass

    return HealthResponse(
        status          = "ok",
        model_loaded    = model_loaded,
        chromadb_counts = chromadb_counts,
    )


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Run the full Hybrid AI analysis pipeline:
    1. Fetch OHLCV + feature engineering
    2. XGBoost inference
    3. News scraping
    4. ChromaDB RAG retrieval
    5. LangGraph multi-agent orchestration (Quant + Sentiment + Economist + CIO)

    Returns the CIO's final investment decision with full agent reports.
    """
    start_time = time.monotonic()
    logger.info(f"POST /predict | ticker={request.ticker}")

    try:
        from backend.agents.graph import get_graph
        graph = get_graph()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LangGraph not available: {exc}")

    # Build initial state for the graph
    initial_state = {
        "ticker":        request.ticker,
        "analysis_date": request.analysis_date or str(date.today()),
        "error_messages": [],
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        logger.error(f"Graph execution failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Analysis pipeline failed: {exc}")

    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    # ── Build response ─────────────────────────────────────────────────
    fd   = final_state.get("final_decision",    {})
    ml   = final_state.get("ml_prediction",     {})
    news = final_state.get("news_articles",      [])
    ctx  = final_state.get("macro_context",      "")

    # Build agent report summaries
    agent_reports: list[AgentReportResponse] = []
    for agent_name, report_key, rec_key, conf_key in [
        ("Quant Agent",      "quant_report",     "recommendation",   "confidence"),
        ("Sentiment Agent",  "sentiment_report",  "overall_sentiment","confidence"),
        ("Economist Agent",  "economist_report",  "macro_sentiment",  "confidence"),
    ]:
        raw = final_state.get(report_key, {})
        agent_reports.append(AgentReportResponse(
            agent_name     = agent_name,
            recommendation = str(raw.get(rec_key, "N/A")),
            confidence     = int(raw.get(conf_key, 0)),
            summary        = str(raw.get("summary", "")),
            raw_report     = raw,
        ))

    weights_raw = fd.get("weights_used", {"quant": 0.4, "sentiment": 0.3, "macro": 0.3})

    return PredictionResponse(
        ticker           = request.ticker,
        prediction_date  = final_state.get("analysis_date", str(date.today())),
        direction        = fd.get("direction",        "ĐI NGANG"),
        confidence_score = float(fd.get("confidence_score", 0.5)),
        action           = fd.get("action",           "CHỜ ĐỢI"),
        ml_probability   = {"p_up": ml.get("p_up", 0.5), "p_down": ml.get("p_down", 0.5)},
        ml_direction     = ml.get("direction",        "ĐI NGANG"),
        ml_confidence    = ml.get("confidence",       "LOW"),
        weight_regime    = fd.get("weight_regime",    "BALANCED"),
        weights_used     = WeightsUsed(**weights_raw),
        weighted_score   = float(fd.get("weighted_score", 0.0)),
        consensus        = fd.get("consensus",        "MIXED"),
        key_signals      = fd.get("key_signals",      []),
        risk_factors     = fd.get("risk_factors",     []),
        cio_reasoning    = fd.get("reasoning",        ""),
        stop_loss_note   = fd.get("stop_loss_note",   ""),
        agent_reports    = agent_reports,
        articles_scraped = len(news),
        rag_context_chars= len(ctx),
        error_messages   = final_state.get("error_messages", []),
        processing_time_ms = elapsed_ms,
    )


# ---------------------------------------------------------------------------
# POST /train
# ---------------------------------------------------------------------------

@app.post("/train", response_model=TrainResponse, tags=["Model Management"])
async def train_model(request: TrainRequest):
    """
    Trigger a full XGBoost re-training run.
    Downloads fresh data, engineers features, trains, and saves the model.
    This endpoint may take 1-3 minutes to complete.
    """
    logger.info(f"POST /train | ticker={request.ticker}")
    try:
        from backend.ml_models.trainer import ModelTrainer
        trainer = ModelTrainer(
            ticker        = request.ticker,
            lookback_days = request.lookback_days,
            model_dir     = Path(settings.model_path),
        )
        result = trainer.train()

        # Reload predictor singleton to pick up new model
        from backend.ml_models.predictor import MarketPredictor
        MarketPredictor(model_dir=Path(settings.model_path)).reload_model()

        return TrainResponse(
            status     = "success",
            model_path = result["model_path"],
            accuracy   = result["accuracy"],
            auc_roc    = result["auc_roc"],
            trained_at = result["trained_at"],
            message    = f"Model trained with {result.get('accuracy', 0):.1%} accuracy",
        )
    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")


# ---------------------------------------------------------------------------
# GET /market-data
# ---------------------------------------------------------------------------

@app.get("/market-data", response_model=MarketDataResponse, tags=["Data"])
async def get_market_data(
    ticker: str = Query(default="^VNINDEX"),
    days:   int = Query(default=30, ge=5, le=504),
):
    """Return latest OHLCV data + technical indicators for charting."""
    try:
        from backend.scraper.market_data import MarketDataFetcher
        from backend.ml_models.feature_engineer import FeatureEngineer

        fetcher  = MarketDataFetcher(ticker=ticker, lookback_days=max(days, 60))
        result   = fetcher.fetch()
        df       = result.df

        fe       = FeatureEngineer()
        feat_df  = fe.build_features(df, add_target=False)
        latest   = feat_df.iloc[-1]

        indicators = {
            k: round(float(latest.get(k, 0) or 0), 4)
            for k in ["rsi_14", "macd", "macd_signal", "bb_width", "atr_14", "adx",
                       "stoch_k", "volume_ratio", "return_1d", "return_5d"]
        }

        ohlcv_tail = df.tail(days).reset_index().to_dict(orient="records")
        # Make dates JSON-serializable
        for row in ohlcv_tail:
            if hasattr(row.get("date", None), "isoformat"):
                row["date"] = row["date"].isoformat()

        summary = result.to_summary_dict()
        return MarketDataResponse(
            ticker        = ticker,
            date          = summary.get("date", ""),
            close         = summary.get("close", 0),
            open          = summary.get("open",  0),
            high          = summary.get("high",  0),
            low           = summary.get("low",   0),
            volume        = summary.get("volume", 0),
            return_1d     = summary.get("return_1d", 0),
            indicators    = indicators,
            ohlcv_history = ohlcv_tail,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# GET /news
# ---------------------------------------------------------------------------

@app.get("/news", response_model=NewsResponse, tags=["Data"])
async def get_news(limit: int = Query(default=20, ge=1, le=50)):
    """Crawl and return latest Vietnamese financial news."""
    from datetime import datetime
    try:
        from backend.scraper.news_scraper import NewsScraper
        scraper  = NewsScraper(max_articles_per_source=settings.max_articles_per_source)
        articles = scraper.scrape_all()
        return NewsResponse(
            articles   = articles[:limit],
            total      = len(articles),
            scraped_at = datetime.now().isoformat(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# GET /rag/query
# ---------------------------------------------------------------------------

@app.get("/rag/query", response_model=RagQueryResponse, tags=["RAG"])
async def rag_query(
    q:          str = Query(..., description="Search query"),
    collection: str = Query(default="macro_reports"),
    n_results:  int = Query(default=5, ge=1, le=20),
):
    """Perform a semantic search against a ChromaDB collection."""
    try:
        from backend.rag_engine.retriever import ContextRetriever
        retriever = ContextRetriever(chroma_path=Path(settings.chroma_db_path))
        results   = retriever.query_raw(query=q, collection=collection, n_results=n_results)
        return RagQueryResponse(
            query      = q,
            collection = collection,
            results    = results,
            total      = len(results),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /rag/index
# ---------------------------------------------------------------------------

@app.post("/rag/index", tags=["RAG"])
async def rag_index(request: RagIndexRequest):
    """Index raw text into a ChromaDB collection."""
    try:
        from backend.rag_engine.indexer import DocumentIndexer
        indexer = DocumentIndexer(chroma_path=Path(settings.chroma_db_path))
        n = indexer.index_text(
            text       = request.text,
            source     = request.source,
            collection = request.collection,
            doc_date   = request.doc_date or "",
        )
        return {"status": "success", "chunks_indexed": n, "collection": request.collection}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
