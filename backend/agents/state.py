"""
backend/agents/state.py
========================
Shared AgentState TypedDict — the single source of truth flowing
through the entire LangGraph StateGraph.

Every node reads from and writes to this structure.
LangGraph merges partial updates from parallel branches via its
built-in reducer (last-writer-wins per key).
"""

from __future__ import annotations

from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    """
    Shared state dictionary propagated across all LangGraph nodes.

    Keys are grouped by the phase that populates them:
      - INPUT CONTEXT      : set by the FastAPI caller before graph.invoke()
      - MARKET DATA        : set by load_context_node (scraper + feature eng.)
      - ML OUTPUT          : set by load_context_node (XGBoost predictor)
      - UNSTRUCTURED DATA  : set by load_context_node (news + RAG)
      - AGENT OUTPUTS      : set by respective agent nodes (parallel)
      - FINAL OUTPUT       : set by CIO agent node
      - META               : error tracking
    """

    # ── INPUT CONTEXT (caller sets these) ──────────────────────────────
    ticker:                str          # e.g. "^VNINDEX"
    analysis_date:         str          # e.g. "2026-05-07" (next trading day)

    # ── MARKET DATA (load_context_node) ────────────────────────────────
    ohlcv_summary:         dict         # Latest bar: close, volume, return_1d, …
    technical_indicators:  dict         # RSI, MACD, BB_width, ATR, ADX, …
    ohlcv_df_json:         str          # JSON-serialized OHLCV for downstream use

    # ── ML OUTPUT (load_context_node → XGBoost) ────────────────────────
    ml_prediction:         dict         # {p_up, p_down, direction, confidence, features_snapshot}

    # ── UNSTRUCTURED DATA (load_context_node) ──────────────────────────
    news_articles:         list         # list[dict] from NewsScraper
    macro_context:         str          # Formatted RAG context string from ChromaDB

    # ── AGENT OUTPUTS (set by parallel agent nodes) ────────────────────
    quant_report:          dict         # Quant Agent structured JSON output
    sentiment_report:      dict         # Sentiment Agent structured JSON output
    economist_report:      dict         # Economist Agent structured JSON output

    # ── FINAL OUTPUT (CIO Agent) ───────────────────────────────────────
    final_decision:        dict         # {direction, confidence_score, consensus,
                                        #  key_signals, risk_factors, reasoning, action,
                                        #  weights_used}

    # ── META ────────────────────────────────────────────────────────────
    error_messages:        list         # list[str] — non-fatal errors logged here
