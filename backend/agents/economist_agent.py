"""
backend/agents/economist_agent.py
====================================
Macro Economist Agent — LangGraph node that performs deep macroeconomic
analysis grounded in Managerial Economics principles and ChromaDB RAG context.

Role: Senior Macro Economist (Managerial Economics framework)
Goal: Analyze macro forces affecting corporate cost of capital, inflation,
      monetary policy cycles, and their transmission to VN-Index
Tools: Reads macro_context (ChromaDB RAG) + news_articles from AgentState
Output: economist_report dict written back to AgentState

Analytical Framework (Managerial Economics):
  1. Cost of Capital Channel: interest rate → WACC → equity valuation
  2. Aggregate Demand Channel: monetary policy → consumption/investment
  3. Exchange Rate Channel: VND/USD → export competitiveness + imported inflation
  4. Credit Channel: credit growth → corporate liquidity → capex cycle
  5. Expectations Channel: inflation expectations → real interest rate
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.agents.state import AgentState
from backend.config import get_settings

# ---------------------------------------------------------------------------
# System Prompt — Managerial Economics Foundation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Bạn là Nhà Kinh tế Vĩ mô Cấp cao (Senior Macro Economist) với chuyên môn sâu về Kinh tế Quản lý Doanh nghiệp (Managerial Economics) và thị trường tài chính Đông Nam Á, đặc biệt Việt Nam.

KHUNG PHÂN TÍCH — MANAGERIAL ECONOMICS:
Bạn phân tích tác động vĩ mô qua 5 kênh truyền dẫn:

1. KÊNH CHI PHÍ VỐN (Cost of Capital Channel):
   - Lãi suất cơ bản (FED, NHNN) → Chi phí nợ (Kd) → WACC → P/E định giá cổ phiếu
   - Khi lãi suất tăng: WACC tăng → NPV dự án giảm → Định giá cổ phiếu giảm

2. KÊNH CẦU TỔNG HỢP (Aggregate Demand Channel):
   - Chính sách tiền tệ → Tiêu dùng nội địa + Đầu tư doanh nghiệp → Doanh thu/lợi nhuận

3. KÊNH TỶ GIÁ (Exchange Rate Channel):
   - VND/USD biến động → Xuất khẩu (VN phụ thuộc nhiều) → Lạm phát nhập khẩu
   - Khối ngoại: tỷ giá ảnh hưởng đến dòng vốn ngoại vào/ra VN-Index

4. KÊNH TÍN DỤNG (Credit Channel):
   - Tăng trưởng tín dụng → Thanh khoản doanh nghiệp → Chu kỳ đầu tư vốn (Capex)
   - Room tín dụng ngân hàng ảnh hưởng trực tiếp đến cổ phiếu nhóm ngân hàng

5. KÊNH KỲ VỌNG (Expectations Channel):
   - Kỳ vọng lạm phát → Lãi suất thực → Quyết định đầu tư dài hạn
   - Forward guidance của FED và NHNN định hình tâm lý thị trường

QUY TẮC BẮT BUỘC:
- Phân tích ĐI SÂU vào cơ chế truyền dẫn, không chỉ đọc bề mặt tin tức
- Nêu rõ "Dữ liệu không đủ để kết luận về [X]" nếu thiếu thông tin
- Phân biệt rõ tác động NGẮN HẠN (1-5 ngày) và DÀI HẠN (1-3 tháng)
- Trả về JSON thuần túy, không markdown

OUTPUT FORMAT:
{
  "transmission_channels": {
    "cost_of_capital": "Phân tích kênh chi phí vốn...",
    "aggregate_demand": "...",
    "exchange_rate": "...",
    "credit_channel": "...",
    "expectations": "..."
  },
  "key_macro_factors": [
    {"factor": "FED giữ lãi suất 5.25%", "impact": "NEGATIVE", "mechanism": "WACC tăng → định giá giảm"},
    {"factor": "...", "impact": "POSITIVE|NEGATIVE|NEUTRAL", "mechanism": "..."}
  ],
  "macro_keywords_detected": ["tăng lãi suất", "bơm tiền"],
  "macro_regime": "TIGHTENING | EASING | NEUTRAL",
  "short_term_impact": "NEGATIVE | POSITIVE | NEUTRAL",
  "long_term_impact": "NEGATIVE | POSITIVE | NEUTRAL",
  "macro_sentiment": "BEARISH | BULLISH | CAUTIOUS | NEUTRAL",
  "confidence": 70,
  "data_gaps": ["Thiếu dữ liệu CPI tháng 5", "..."],
  "summary": "Tổng hợp phân tích vĩ mô sâu 3-4 câu theo khung Managerial Economics"
}"""

# ---------------------------------------------------------------------------
# Macro Keywords for CIO Agent weight signaling
# ---------------------------------------------------------------------------

HIGH_IMPACT_MACRO_KEYWORDS = [
    # Vietnamese
    "tăng lãi suất", "giảm lãi suất", "bơm tiền", "hút tiền", "khủng hoảng",
    "suy thoái", "lạm phát cao", "phá giá", "nới lỏng định lượng", "thắt chặt",
    "nâng lãi suất", "cắt giảm lãi suất", "tỷ giá biến động mạnh",
    "tăng trưởng âm", "khủng hoảng ngân hàng", "vỡ nợ", "bong bóng",
    # English (in Vietnamese financial news)
    "rate hike", "rate cut", "quantitative easing", "recession", "default",
    "banking crisis", "inflation surge", "currency crisis",
]


def _detect_macro_keywords(macro_context: str, news_articles: list[dict]) -> list[str]:
    """Scan context and news for high-impact macro keywords."""
    combined_text = macro_context.lower()
    for art in news_articles:
        combined_text += " " + art.get("title", "").lower()
        combined_text += " " + art.get("summary", "").lower()

    detected = [kw for kw in HIGH_IMPACT_MACRO_KEYWORDS if kw in combined_text]
    if detected:
        logger.info(f"[Economist Agent] High-impact macro keywords detected: {detected}")
    return detected


def _build_human_message(state: AgentState) -> str:
    macro_context = state.get("macro_context", "Không có dữ liệu vĩ mô từ ChromaDB.")
    news_articles = state.get("news_articles", [])
    ticker        = state.get("ticker", "^VNINDEX")
    analysis_date = state.get("analysis_date", "N/A")

    # Sample top 5 news titles for macro context enrichment
    news_sample = "\n".join(
        f"- {a.get('title', '')}" for a in news_articles[:5]
    ) or "Không có tin tức."

    return f"""PHÂN TÍCH VĨ MÔ VN-INDEX
Ticker: {ticker} | Ngày phân tích: {analysis_date}

=== NGỮ CẢNH VĨ MÔ (Từ ChromaDB RAG) ===
{macro_context}

=== TIN TỨC VĨ MÔ NỔI BẬT HÔM NAY ===
{news_sample}

Hãy phân tích theo khung Kinh tế Quản lý Doanh nghiệp (5 kênh truyền dẫn) và
trả về JSON theo đúng định dạng yêu cầu."""


# ---------------------------------------------------------------------------
# LLM Call with Retry
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_llm(llm: ChatGroq, messages: list) -> str:
    return llm.invoke(messages).content


# ---------------------------------------------------------------------------
# Node Function
# ---------------------------------------------------------------------------

def economist_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node for Macro Economist Agent.
    Reads: macro_context, news_articles, ticker, analysis_date
    Writes: economist_report (includes macro_keywords_detected for CIO weight logic)
    """
    logger.info("[Economist Agent] Starting macro analysis…")
    settings = get_settings()

    # Pre-detect keywords before LLM call (for CIO weight signaling)
    keywords = _detect_macro_keywords(
        macro_context  = state.get("macro_context", ""),
        news_articles  = state.get("news_articles", []),
    )

    llm = ChatGroq(
        model       = settings.groq_model,
        api_key     = settings.groq_api_key,
        temperature = 0.2,   # Slightly higher for nuanced economic reasoning
        max_tokens  = 2000,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_human_message(state)),
    ]

    try:
        raw = _call_llm(llm, messages)
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        report = json.loads(clean)

        # Ensure keyword list is populated (LLM may also detect some)
        existing_kw = report.get("macro_keywords_detected", [])
        report["macro_keywords_detected"] = list(set(existing_kw + keywords))

        logger.success(
            f"[Economist Agent] Done | regime={report.get('macro_regime')} "
            f"| sentiment={report.get('macro_sentiment')} "
            f"| keywords={len(report['macro_keywords_detected'])}"
        )
    except json.JSONDecodeError as exc:
        logger.error(f"[Economist Agent] JSON parse failed: {exc}")
        report = {
            "transmission_channels":  {},
            "key_macro_factors":      [],
            "macro_keywords_detected": keywords,
            "macro_regime":           "NEUTRAL",
            "short_term_impact":      "NEUTRAL",
            "long_term_impact":       "NEUTRAL",
            "macro_sentiment":        "NEUTRAL",
            "confidence":             0,
            "data_gaps":              ["Parse error"],
            "summary":                f"Economist Agent parse error: {exc}",
        }
    except Exception as exc:
        logger.error(f"[Economist Agent] LLM call failed: {exc}")
        report = {
            "transmission_channels":  {},
            "key_macro_factors":      [],
            "macro_keywords_detected": keywords,
            "macro_regime":           "NEUTRAL",
            "short_term_impact":      "NEUTRAL",
            "long_term_impact":       "NEUTRAL",
            "macro_sentiment":        "NEUTRAL",
            "confidence":             0,
            "data_gaps":              [f"LLM error: {exc}"],
            "summary":                f"Economist Agent unavailable: {exc}",
        }
        errors = list(state.get("error_messages", []))
        errors.append(f"Economist Agent error: {exc}")
        return {"economist_report": report, "error_messages": errors}

    return {"economist_report": report}
