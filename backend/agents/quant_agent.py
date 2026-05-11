"""
backend/agents/quant_agent.py
================================
Quant Agent — LangGraph node that interprets XGBoost predictions
and technical indicators into a structured quantitative analysis report.

Role: Senior Quantitative Analyst (15 years experience)
Goal: Translate ML probability + TA signals into actionable BUY/SELL/NEUTRAL recommendation
Tools: Reads ml_prediction + technical_indicators from AgentState
Output: quant_report dict written back to AgentState
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
# System Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Bạn là một Chuyên gia Phân tích Định lượng (Quantitative Analyst) cấp cao với 15 năm kinh nghiệm giao dịch tại thị trường chứng khoán Việt Nam và quốc tế.

Phong cách phân tích:
- Luôn tham chiếu con số cụ thể (RSI=X, MACD=Y, P_up=Z%)
- Xác định 2-3 tín hiệu kỹ thuật QUAN TRỌNG NHẤT
- Đánh giá sự ĐỒNG THUẬN hoặc MÂU THUẪN giữa mô hình ML và chỉ báo kỹ thuật
- Kết luận rõ ràng, không mơ hồ

Quy tắc bắt buộc:
- Chỉ phân tích từ dữ liệu được cung cấp — KHÔNG tự ý bịa số liệu
- Nếu tín hiệu mâu thuẫn nhau, nêu rõ điều đó và đánh trọng số cho tín hiệu mạnh hơn
- Trả lời ĐÚNG định dạng JSON được yêu cầu, không thêm text ngoài JSON

OUTPUT FORMAT (JSON thuần túy, không markdown):
{
  "signals": ["RSI=72 → Overbought", "MACD histogram dương và tăng → Bullish momentum"],
  "ml_alignment": "XGBoost P(Tăng)=72% ĐỒNG THUẬN với tín hiệu RSI và MACD",
  "trend_assessment": "BULLISH | BEARISH | NEUTRAL",
  "key_levels": {"support": "...", "resistance": "..."},
  "recommendation": "BUY | SELL | NEUTRAL",
  "confidence": 75,
  "summary": "Tóm tắt phân tích ngắn gọn 2-3 câu"
}"""

# ---------------------------------------------------------------------------
# Helper — build human message
# ---------------------------------------------------------------------------

def _build_human_message(state: AgentState) -> str:
    ml  = state.get("ml_prediction", {})
    ind = state.get("technical_indicators", {})
    ohv = state.get("ohlcv_summary", {})

    return f"""PHÂN TÍCH DỮ LIỆU KỸ THUẬT VN-INDEX

=== KẾT QUẢ MÔ HÌNH XGBoost ===
- Xác suất Tăng (P_up):   {ml.get('p_up', 'N/A')}
- Xác suất Giảm (P_down): {ml.get('p_down', 'N/A')}
- Dự báo:                 {ml.get('direction', 'N/A')}
- Độ tin cậy ML:          {ml.get('confidence', 'N/A')}

=== CHỈ BÁO KỸ THUẬT ===
Momentum:
  RSI(14)    = {ind.get('rsi_14', 'N/A')}
  Stoch K    = {ind.get('stoch_k', 'N/A')}
  Stoch D    = {ind.get('stoch_d', 'N/A')}

Trend:
  MACD       = {ind.get('macd', 'N/A')}
  MACD Signal= {ind.get('macd_signal', 'N/A')}
  MACD Hist  = {ind.get('macd_diff', 'N/A')}
  EMA(9)     = {ind.get('ema_9', 'N/A')}
  EMA(21)    = {ind.get('ema_21', 'N/A')}
  ADX(14)    = {ind.get('adx', 'N/A')}

Volatility:
  BB Width   = {ind.get('bb_width', 'N/A')}
  ATR(14)    = {ind.get('atr_14', 'N/A')}

Volume:
  Vol Ratio  = {ind.get('volume_ratio', 'N/A')}
  OBV        = {ind.get('obv', 'N/A')}

=== GIÁ THỊ TRƯỜNG ===
  Đóng cửa   = {ohv.get('close', 'N/A')}
  Return 1D  = {ohv.get('return_1d', 'N/A')}%
  Volume     = {ohv.get('volume', 'N/A')}

Hãy phân tích và trả về JSON theo đúng định dạng yêu cầu."""


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
    response = llm.invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# Node Function (LangGraph entry point)
# ---------------------------------------------------------------------------

def quant_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node for Quant Agent.
    Reads: ml_prediction, technical_indicators, ohlcv_summary
    Writes: quant_report
    """
    logger.info("[Quant Agent] Starting analysis…")
    settings = get_settings()

    llm = ChatGroq(
        model       = settings.groq_model,
        api_key     = settings.groq_api_key,
        temperature = 0.1,   # Low temp for analytical consistency
        max_tokens  = 1024,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_human_message(state)),
    ]

    try:
        raw_response = _call_llm(llm, messages)
        # Strip any accidental markdown code fences
        clean = raw_response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        report = json.loads(clean)
        logger.success(
            f"[Quant Agent] Done | recommendation={report.get('recommendation')} "
            f"| confidence={report.get('confidence')}%"
        )
    except json.JSONDecodeError as exc:
        logger.error(f"[Quant Agent] JSON parse failed: {exc} | raw={raw_response[:200]}")
        report = {
            "signals":          ["Parse error — raw LLM response"],
            "ml_alignment":     "N/A",
            "trend_assessment": "NEUTRAL",
            "key_levels":       {},
            "recommendation":   "NEUTRAL",
            "confidence":       0,
            "summary":          f"Quant Agent parse error: {exc}",
        }
    except Exception as exc:
        logger.error(f"[Quant Agent] LLM call failed: {exc}")
        report = {
            "signals":          [f"LLM error: {exc}"],
            "ml_alignment":     "N/A",
            "trend_assessment": "NEUTRAL",
            "key_levels":       {},
            "recommendation":   "NEUTRAL",
            "confidence":       0,
            "summary":          f"Quant Agent unavailable: {exc}",
        }
        errors = list(state.get("error_messages", []))
        errors.append(f"Quant Agent error: {exc}")
        return {"quant_report": report, "error_messages": errors}

    return {"quant_report": report}
