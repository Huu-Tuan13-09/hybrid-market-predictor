"""
backend/agents/sentiment_agent.py
====================================
Sentiment Agent — LangGraph node that analyzes crawled news articles
and quantifies market sentiment (Positive / Negative / Neutral).

Role: Market Sentiment Analyst
Goal: Evaluate investor psychology and news flow direction
Tools: Reads news_articles from AgentState
Output: sentiment_report dict written back to AgentState
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.agents.state import AgentState
from backend.config import get_settings

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Bạn là Chuyên gia Phân tích Tâm lý Thị trường (Market Sentiment Analyst) chuyên về thị trường chứng khoán Việt Nam.

Phương pháp phân tích:
1. Đọc kỹ TIÊU ĐỀ và TÓM TẮT của từng bài báo
2. Phân loại từng bài: POSITIVE / NEGATIVE / NEUTRAL
3. Chú ý đặc biệt đến các từ khóa tác động mạnh:
   - TIÊU CỰC: "bán ròng", "bán tháo", "thua lỗ", "lo ngại", "áp lực", "giảm sâu", "khủng hoảng"
   - TÍCH CỰC: "mua ròng", "tăng vốn", "kỷ lục", "lạc quan", "phục hồi", "đột phá", "tăng trưởng"
4. Xác định 2-3 CHỦ ĐỀ CHÍNH nổi bật nhất trong ngày
5. Tính sentiment_score từ -1.0 (rất tiêu cực) đến +1.0 (rất tích cực)

Quy tắc:
- Chỉ kết luận từ nội dung tin tức được cung cấp
- Không suy diễn hay bổ sung thông tin ngoài
"""


# ---------------------------------------------------------------------------
# Pydantic Schemas for Structured Output
# ---------------------------------------------------------------------------

class ArticleSentiment(BaseModel):
    title: str = Field(description="Tiêu đề bài báo")
    sentiment: str = Field(description="Phân loại tâm lý: POSITIVE, NEGATIVE, hoặc NEUTRAL")

class SentimentReport(BaseModel):
    article_sentiments: list[ArticleSentiment] = Field(description="Danh sách đánh giá tâm lý cho từng bài báo")
    positive_count: int = Field(description="Số lượng tin tức POSITIVE")
    negative_count: int = Field(description="Số lượng tin tức NEGATIVE")
    neutral_count: int = Field(description="Số lượng tin tức NEUTRAL")
    overall_sentiment: str = Field(description="Tâm lý tổng thể: POSITIVE, NEGATIVE, NEUTRAL, hoặc MIXED")
    dominant_themes: list[str] = Field(description="2-3 chủ đề chính nổi bật nhất trong ngày")
    sentiment_score: float = Field(description="Điểm tâm lý từ -1.0 (rất tiêu cực) đến +1.0 (rất tích cực)")
    market_fear_greed: str = Field(description="Trạng thái thị trường: FEAR, GREED, hoặc NEUTRAL")
    confidence: int = Field(description="Độ tin cậy của đánh giá từ 0 đến 100")
    summary: str = Field(description="Tóm tắt tổng thể tâm lý thị trường hôm nay trong 2-3 câu")


# ---------------------------------------------------------------------------
# Helper — format articles for prompt
# ---------------------------------------------------------------------------

def _format_articles(articles: list[dict], max_articles: int = 15) -> str:
    """Format news articles list into prompt-friendly text."""
    if not articles:
        return "Không có tin tức nào được cung cấp."

    lines = [f"DANH SÁCH TIN TỨC TÀI CHÍNH ({len(articles[:max_articles])} bài):\n"]
    for i, art in enumerate(articles[:max_articles], start=1):
        title   = art.get("title", "No title")
        summary = art.get("summary", "")
        source  = art.get("source", "unknown")
        date    = art.get("published_at", "")
        lines.append(
            f"{i}. [{source} | {date}]\n"
            f"   Tiêu đề: {title}\n"
            f"   Tóm tắt: {summary}\n"
        )
    return "\n".join(lines)


def _build_human_message(state: AgentState) -> str:
    articles = state.get("news_articles", [])
    articles_text = _format_articles(articles)
    return f"""{articles_text}

Hãy phân tích toàn bộ tin tức trên và trích xuất dữ liệu theo đúng cấu trúc được yêu cầu."""


# ---------------------------------------------------------------------------
# LLM Call with Retry
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_llm_structured(llm: ChatGroq, messages: list) -> dict:
    structured_llm = llm.with_structured_output(SentimentReport)
    result = structured_llm.invoke(messages)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Node Function
# ---------------------------------------------------------------------------

def sentiment_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node for Sentiment Agent.
    Reads: news_articles
    Writes: sentiment_report
    """
    logger.info("[Sentiment Agent] Starting analysis…")
    settings = get_settings()

    llm = ChatGroq(
        model       = settings.groq_model,
        api_key     = settings.groq_api_key,
        temperature = 0.1,
        max_tokens  = 1500,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_human_message(state)),
    ]

    try:
        report = _call_llm_structured(llm, messages)
        logger.success(
            f"[Sentiment Agent] Done | overall={report.get('overall_sentiment')} "
            f"| score={report.get('sentiment_score')}"
        )
    except Exception as exc:
        logger.error(f"[Sentiment Agent] LLM call failed: {exc}")
        report = {
            "article_sentiments": [],
            "positive_count":     0,
            "negative_count":     0,
            "neutral_count":      0,
            "overall_sentiment":  "NEUTRAL",
            "dominant_themes":    [f"Error: {exc}"],
            "sentiment_score":    0.0,
            "market_fear_greed":  "NEUTRAL",
            "confidence":         0,
            "summary":            f"Sentiment Agent unavailable: {exc}",
        }
        errors = list(state.get("error_messages", []))
        errors.append(f"Sentiment Agent error: {exc}")
        return {"sentiment_report": report, "error_messages": errors}

    return {"sentiment_report": report}
