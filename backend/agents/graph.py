"""
backend/agents/graph.py
========================
LangGraph StateGraph — wires all agent nodes and the context loader
into a compiled graph with parallel fan-out execution.

Graph topology:
  START → load_context
            ├── quant_agent    ─┐
            ├── sentiment_agent ├── (parallel fan-out)
            └── economist_agent─┘
                   └──────────── cio_agent → END

The 'load_context' node is responsible for:
  1. Fetching OHLCV data and engineering features
  2. Running XGBoost inference
  3. Crawling news articles
  4. Querying ChromaDB for macro context
  All results are written to AgentState before the parallel agents run.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger

from backend.agents.state import AgentState
from backend.agents.quant_agent import quant_agent_node
from backend.agents.sentiment_agent import sentiment_agent_node
from backend.agents.economist_agent import economist_agent_node
from backend.agents.cio_agent import cio_agent_node
from backend.config import get_settings

# ---------------------------------------------------------------------------
# Load Context Node — populates all raw data before agents run
# ---------------------------------------------------------------------------

def load_context_node(state: AgentState) -> dict[str, Any]:
    """
    Pre-processing node: fetch market data, run ML inference, scrape news,
    and query ChromaDB. Writes results into AgentState for all downstream agents.
    """
    settings      = get_settings()
    ticker        = state.get("ticker", settings.vnindex_ticker)
    analysis_date = state.get("analysis_date", str(date.today()))
    errors: list[str] = []

    logger.info(f"[load_context] Fetching data for {ticker} | date={analysis_date}")

    # ── 1. Market Data + Feature Engineering ──────────────────────────
    ohlcv_summary: dict = {}
    technical_indicators: dict = {}
    ohlcv_df_json: str = "{}"
    ml_prediction: dict = {}

    try:
        from backend.scraper.market_data import MarketDataFetcher
        from backend.ml_models.feature_engineer import FeatureEngineer, FEATURE_COLUMNS

        fetcher = MarketDataFetcher(
            ticker        = ticker,
            lookback_days = settings.lookback_days,
        )
        result  = fetcher.fetch()
        df      = result.df

        ohlcv_summary = result.to_summary_dict()

        # Engineer features (add_target=False — inference mode)
        fe         = FeatureEngineer()
        feat_df    = fe.build_features(df, add_target=False)
        latest_row = feat_df.iloc[-1]

        # Build technical_indicators dict from latest feature row
        indicator_keys = [
            "rsi_14", "stoch_k", "stoch_d",
            "macd", "macd_signal", "macd_diff",
            "ema_9", "ema_21", "sma_50", "adx",
            "bb_upper", "bb_lower", "bb_width", "atr_14",
            "obv", "volume_ratio",
            "return_1d", "return_5d", "return_20d",
        ]
        technical_indicators = {
            k: round(float(latest_row.get(k, 0) or 0), 4)
            for k in indicator_keys
        }

        # Serialize OHLCV tail for downstream storage (last 10 rows)
        ohlcv_df_json = df.tail(10).reset_index().to_json(orient="records", date_format="iso")

        logger.success(f"[load_context] Market data ready | rows={result.num_rows}")
    except Exception as exc:
        logger.error(f"[load_context] Market data/feature engineering failed: {exc}")
        errors.append(f"Market data error: {exc}")

    # ── 2. XGBoost Inference ───────────────────────────────────────────
    try:
        from backend.ml_models.predictor import MarketPredictor
        model_dir  = Path(settings.model_path)
        predictor  = MarketPredictor(model_dir=model_dir)

        # Re-fetch full df (predictor does its own feature engineering internally)
        from backend.scraper.market_data import MarketDataFetcher
        fetcher2 = MarketDataFetcher(ticker=ticker, lookback_days=settings.lookback_days)
        result2  = fetcher2.fetch()
        ml_prediction = predictor.predict(result2.df)
        logger.success(f"[load_context] ML inference done | {ml_prediction.get('direction')} @ P_up={ml_prediction.get('p_up')}")
    except Exception as exc:
        logger.error(f"[load_context] XGBoost inference failed: {exc}")
        errors.append(f"ML inference error: {exc}")
        ml_prediction = {"p_up": 0.5, "p_down": 0.5, "direction": "ĐI NGANG", "confidence": "LOW"}

    # ── 3. News Scraping ───────────────────────────────────────────────
    news_articles: list[dict] = []
    try:
        from backend.scraper.news_scraper import NewsScraper
        scraper      = NewsScraper(max_articles_per_source=settings.max_articles_per_source)
        news_articles = scraper.scrape_all()
        logger.success(f"[load_context] News scraped | {len(news_articles)} articles")
    except Exception as exc:
        logger.error(f"[load_context] News scraping failed: {exc}")
        errors.append(f"News scraping error: {exc}")

    # ── 4. ChromaDB RAG Retrieval ──────────────────────────────────────
    macro_context: str = ""
    try:
        from backend.rag_engine.retriever import ContextRetriever
        chroma_path = Path(settings.chroma_db_path)
        retriever   = ContextRetriever(chroma_path=chroma_path)

        macro_context = retriever.multi_query({
            "macro_reports": f"Vietnam economic outlook monetary policy {date.today().year}",
            "macro_reports": "FED interest rate decision impact emerging markets Vietnam",
            "market_news":   f"VN-Index market sentiment {analysis_date}",
        }, n_results=5)

        # Also index today's news into ChromaDB for future queries
        if news_articles:
            from backend.rag_engine.indexer import DocumentIndexer
            indexer = DocumentIndexer(chroma_path=chroma_path)
            indexer.index_news_articles(news_articles)

        logger.success(f"[load_context] RAG context retrieved | {len(macro_context)} chars")
    except Exception as exc:
        logger.error(f"[load_context] RAG retrieval failed: {exc}")
        errors.append(f"RAG error: {exc}")
        macro_context = "Không có dữ liệu vĩ mô từ ChromaDB (RAG unavailable)."

    return {
        "ticker":               ticker,
        "analysis_date":        analysis_date,
        "ohlcv_summary":        ohlcv_summary,
        "technical_indicators": technical_indicators,
        "ohlcv_df_json":        ohlcv_df_json,
        "ml_prediction":        ml_prediction,
        "news_articles":        news_articles,
        "macro_context":        macro_context,
        "error_messages":       errors,
    }


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Graph topology:
      START → load_context → [quant | sentiment | economist] → cio → END

    The three specialist agents run in parallel (LangGraph handles this
    automatically when multiple edges leave the same node).

    Returns:
        Compiled LangGraph application (supports .invoke() and .stream()).
    """
    graph = StateGraph(AgentState)

    # ── Register Nodes ─────────────────────────────────────────────────
    graph.add_node("load_context",     load_context_node)
    graph.add_node("quant_agent",      quant_agent_node)
    graph.add_node("sentiment_agent",  sentiment_agent_node)
    graph.add_node("economist_agent",  economist_agent_node)
    graph.add_node("cio_agent",        cio_agent_node)

    # ── Edges — START → load_context ───────────────────────────────────
    graph.add_edge(START, "load_context")

    # ── Parallel Fan-out: load_context → 3 specialist agents ───────────
    graph.add_edge("load_context", "quant_agent")
    graph.add_edge("load_context", "sentiment_agent")
    graph.add_edge("load_context", "economist_agent")

    # ── Fan-in: all specialists → CIO ──────────────────────────────────
    # LangGraph waits for ALL incoming edges before executing cio_agent
    graph.add_edge("quant_agent",     "cio_agent")
    graph.add_edge("sentiment_agent", "cio_agent")
    graph.add_edge("economist_agent", "cio_agent")

    # ── CIO → END ──────────────────────────────────────────────────────
    graph.add_edge("cio_agent", END)

    compiled = graph.compile()
    logger.info("LangGraph StateGraph compiled successfully")
    return compiled


# ---------------------------------------------------------------------------
# Singleton — cached graph instance
# ---------------------------------------------------------------------------

_graph_instance = None

def get_graph() -> Any:
    """Return cached compiled graph (created once per process lifetime)."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance
