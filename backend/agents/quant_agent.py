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
# Quant Veto — Falling Knife / Circuit Breaker
# ---------------------------------------------------------------------------

_VETO_MIN_SIGNALS = 2  # Số tín hiệu tối thiểu để kích hoạt Veto (trên 3)


def _check_falling_knife_veto(
    indicators: dict, ohlcv: dict
) -> tuple[bool, str]:
    """
    Phát hiện kịch bản 'Bắt dao rơi' (Catching a Falling Knife).

    Kiểm tra 3 tín hiệu độc lập — kích hoạt Veto nếu >= 2 tín hiệu cùng bật:
      1. Volume Breakdown  : volume_ratio < 0.55  (thanh khoản cạn kiệt)
      2. Downtrend Accel   : ADX > 28 VÀ return_1d < -1.5%
      3. Volatility Spike  : ATR/Close > 2.5%     (biến động bùng nổ)

    Khi kích hoạt → ghi đè toàn bộ dự báo XGBoost → STRONG SELL vô điều kiện.

    Returns:
        (veto_triggered: bool, reason: str)
    """
    signals_fired: list[str] = []

    # Tín hiệu 1 — Volume Breakdown (sốc thanh khoản)
    try:
        vol_ratio = float(indicators.get("volume_ratio") or 0)
        if 0 < vol_ratio < 0.55:
            signals_fired.append(
                f"Volume Breakdown (ratio={vol_ratio:.2f}<0.55 — thanh khoản cạn kiệt)"
            )
    except (TypeError, ValueError):
        pass

    # Tín hiệu 2 — ADX Downtrend Acceleration (gia tốc xu hướng giảm)
    try:
        adx       = float(indicators.get("adx") or 0)
        return_1d = float(ohlcv.get("return_1d") or 0)
        if adx > 28 and return_1d < -1.5:
            signals_fired.append(
                f"Downtrend Accel (ADX={adx:.1f}>28, return_1d={return_1d:.2f}%<-1.5%)"
            )
    except (TypeError, ValueError):
        pass

    # Tín hiệu 3 — ATR Volatility Spike (độ giãn nở biến động)
    try:
        atr   = float(indicators.get("atr_14") or 0)
        close = float(ohlcv.get("close") or 1)
        atr_pct = atr / close if close > 0 else 0
        if atr_pct > 0.025:
            signals_fired.append(
                f"Volatility Spike (ATR/Close={atr_pct:.3%}>2.5%)"
            )
    except (TypeError, ValueError):
        pass

    veto   = len(signals_fired) >= _VETO_MIN_SIGNALS
    reason = " | ".join(signals_fired) if signals_fired else "No veto conditions met"
    return veto, reason


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
    Writes: quant_report  (bao gồm veto_signal: bool)
    """
    logger.info("[Quant Agent] Starting analysis…")
    settings = get_settings()

    indicators = state.get("technical_indicators", {})
    ohlcv      = state.get("ohlcv_summary", {})

    # ── STEP 0: Falling Knife Veto — Circuit Breaker ────────────────────────
    veto_triggered, veto_reason = _check_falling_knife_veto(indicators, ohlcv)
    if veto_triggered:
        logger.warning(
            f"[Quant Agent] 🚨 FALLING KNIFE VETO TRIGGERED — "
            f"XGBoost prediction OVERRIDDEN | reason: {veto_reason}"
        )
        report = {
            "signals":          [f"🚨 CIRCUIT BREAKER: {veto_reason}"],
            "ml_alignment":     "XGBoost bị GHI ĐÈ bởi Veto Circuit Breaker — không có giá trị tham khảo",
            "trend_assessment": "BEARISH",
            "key_levels":       {},
            "recommendation":   "STRONG SELL",
            "confidence":       95,
            "summary": (
                f"Phát hiện nguy cơ 'Bắt dao rơi'. "
                f"Circuit Breaker kích hoạt: {veto_reason}. "
                f"Toàn bộ tín hiệu XGBoost và TA bị vô hiệu hóa. KHÔNG MUA."
            ),
            "veto_signal":      True,
            "veto_reason":      veto_reason,
        }
        return {"quant_report": report}

    # ── STEP 1: Normal LLM Analysis Path ────────────────────────────────────
    llm = ChatGroq(
        model       = settings.groq_model,
        api_key     = settings.groq_api_key,
        temperature = 0.1,
        max_tokens  = 1024,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_human_message(state)),
    ]

    try:
        raw_response = _call_llm(llm, messages)
        clean = raw_response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        report = json.loads(clean)
        report["veto_signal"] = False
        report["veto_reason"] = None
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
            "veto_signal":      False,
            "veto_reason":      None,
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
            "veto_signal":      False,
            "veto_reason":      None,
        }
        errors = list(state.get("error_messages", []))
        errors.append(f"Quant Agent error: {exc}")
        return {"quant_report": report, "error_messages": errors}

    return {"quant_report": report}
