"""
backend/agents/cio_agent.py
==============================
CIO Agent — Chief Investment Officer synthesizer node.

Role: Chief Investment Officer (CIO) of a major Vietnamese investment fund
Goal: Synthesize reports from 3 specialist agents into ONE final investment decision
      using DYNAMIC weight adjustment based on market regime and macro signal strength.

DYNAMIC WEIGHT LOGIC:
  Default:
    Quant (XGBoost + TA): 40%
    Sentiment (News):     30%
    Macro (Economist):    30%

  If HIGH-IMPACT MACRO KEYWORDS detected (tăng lãi suất, khủng hoảng, v.v.):
    → Macro + Sentiment weight increases to 60% combined (macro event dominates)
    Quant:     25%
    Sentiment: 35%
    Macro:     40%

  If market is RANGING (low volatility, no big news):
    → Quant weight increases (ML signal most reliable in low-noise environment)
    Quant:     55%
    Sentiment: 25%
    Macro:     20%

This implements the user's requirement:
  "Nếu Economist Agent phát hiện từ khóa vĩ mô mạnh → trọng số Vĩ mô/Tin tức lấn át kỹ thuật.
   Nếu thị trường đi ngang không có tin tức lớn → trọng số Quant (XGBoost) chiếm ưu thế."
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.agents.state import AgentState
from backend.config import get_settings

# ---------------------------------------------------------------------------
# Weight Regime Definitions
# ---------------------------------------------------------------------------

WeightRegime = Literal["MACRO_DOMINANT", "QUANT_DOMINANT", "BALANCED"]

_WEIGHTS: dict[WeightRegime, dict[str, float]] = {
    "BALANCED": {
        "quant":     0.40,
        "sentiment": 0.30,
        "macro":     0.30,
    },
    "MACRO_DOMINANT": {
        "quant":     0.25,
        "sentiment": 0.35,
        "macro":     0.40,
    },
    "QUANT_DOMINANT": {
        "quant":     0.55,
        "sentiment": 0.25,
        "macro":     0.20,
    },
}

# ---------------------------------------------------------------------------
# Temporal Discounting — γ Factor
# ---------------------------------------------------------------------------

# Hệ số chiết khấu theo chân trời thời gian (Spatio-Temporal Discounting)
# Tin dài hạn KHÔNG được phép ảnh hưởng mạnh đến lệnh giao dịch hôm nay
_TEMPORAL_DISCOUNT: dict[str, float] = {
    "SHORT":  1.0,   # Tác động ngay lập tức — giữ nguyên 100%
    "MEDIUM": 0.5,   # Tác động trung hạn — chiết khấu 50%
    "LONG":   0.1,   # Tầm nhìn dài hạn — chiết khấu 90%
}


def _apply_temporal_discount(state: AgentState) -> tuple[float, float, str]:
    """
    Tính điểm Sentiment và Macro sau khi áp dụng hệ số chiết khấu γ.

    Logic Sentiment:
      - Ưu tiên dùng `short_term_score` (LLM tự tính cho tin SHORT+MEDIUM)
      - Fallback: ước tính γ từ tỷ lệ bài LONG trong article_sentiments

    Logic Macro:
      - Phân tách short_term_impact (γ=1.0) và long_term_impact (γ=0.1)
      - Tỷ lệ: 70% ngắn hạn + 30% dài hạn × γ_long

    Returns:
        (discounted_sentiment: float, discounted_macro: float, log_msg: str)
    """
    sentiment_report = state.get("sentiment_report", {})
    economist_report = state.get("economist_report", {})
    long_count = 0

    # ── Sentiment Discounting ─────────────────────────────────────────────
    raw_s_score = float(sentiment_report.get("sentiment_score", 0) or 0)

    if "short_term_score" in sentiment_report:
        # LLM đã tính sẵn điểm ngắn hạn — dùng trực tiếp
        disc_s_score = float(sentiment_report["short_term_score"] or 0)
        long_count   = int(sentiment_report.get("long_term_article_count", 0))
        s_method     = "short_term_score từ LLM"
    else:
        # Fallback: ước tính γ từ tỷ lệ bài LONG
        articles   = sentiment_report.get("article_sentiments", [])
        total      = len(articles)
        long_count = sum(
            1 for a in articles
            if (a.get("time_horizon") or "").upper() == "LONG"
        )
        if total > 0:
            long_ratio   = long_count / total
            gamma_s      = 1.0 - long_ratio * (1.0 - _TEMPORAL_DISCOUNT["LONG"])
            disc_s_score = raw_s_score * gamma_s
        else:
            disc_s_score = raw_s_score
        s_method = f"γ ước tính (long={long_count}/{total})"

    # ── Macro Discounting ─────────────────────────────────────────────────
    _macro_map = {
        "bullish": 1.0,  "positive": 1.0,
        "bearish": -1.0, "negative": -1.0,
        "cautious": -0.3, "neutral": 0.0,
    }

    def _to_m(val: str) -> float:
        return _macro_map.get((val or "").lower().strip(), 0.0)

    short_impact = economist_report.get("short_term_impact", "NEUTRAL")
    long_impact  = economist_report.get("long_term_impact",  "NEUTRAL")
    short_m      = _to_m(short_impact)
    long_m       = _to_m(long_impact)

    # 70% ngắn hạn (γ=1.0) + 30% dài hạn (γ=0.1)
    disc_m_score = short_m * 0.70 + long_m * 0.30 * _TEMPORAL_DISCOUNT["LONG"]

    log_msg = (
        f"[Temporal Discount] "
        f"S: raw={raw_s_score:+.3f} → disc={disc_s_score:+.3f} ({s_method}) | "
        f"M: ST={short_impact}({short_m:+.2f}) LT={long_impact}({long_m:+.2f}) "
        f"→ disc_macro={disc_m_score:+.3f} | long_articles={long_count}"
    )
    return round(disc_s_score, 4), round(disc_m_score, 4), log_msg


# ---------------------------------------------------------------------------
# CIO Veto — Systemic Failure Override
# ---------------------------------------------------------------------------

_SYSTEMIC_FAILURE_KEYWORDS: list[str] = [
    # Tiếng Việt — Rủi ro hệ thống
    "vỡ nợ", "chiến tranh", "đại dịch", "khủng hoảng hệ thống",
    "sụp đổ ngân hàng", "bank run", "bắt bớt lãnh đạo", "phá sản hàng loạt",
    "thảm họa tài chính", "vỡ bóng bóng", "tháo chạy vốn",
    # English — Systemic failure signals
    "default", "war outbreak", "pandemic", "systemic crisis",
    "bank collapse", "mass bankruptcy", "financial meltdown",
]


def _check_cio_veto(state: AgentState) -> tuple[bool, str]:
    """
    Quét toàn bộ báo cáo Agent tìm kiếm tín hiệu thảm họa hệ thống.

    Ưu tiên:
      1. Quant Veto đã kích hoạt (cascade)
      2. Từ khóa rủi ro hệ thống trong Sentiment / Economist

    Khi kích hoạt → ngắt mạch (Short-circuit) toàn bộ weighted scoring
                   → phát STRONG SELL vô điều kiện.

    Returns:
        (veto_triggered: bool, veto_reason: str)
    """
    # Ưu tiên 1: Cascade từ Quant Veto
    quant_report = state.get("quant_report", {})
    if quant_report.get("veto_signal"):
        reason = quant_report.get("veto_reason", "Quant Veto đã kích hoạt")
        logger.warning(f"[CIO Agent] Cascading from Quant Veto: {reason}")
        return True, f"[CASCADE từ Quant Veto] {reason}"

    # Ưu tiên 2: Quét từ khóa rủi ro hệ thống
    sentiment_report = state.get("sentiment_report",  {})
    economist_report = state.get("economist_report", {})

    scan_text = " ".join([
        str(sentiment_report.get("summary",                   "")),
        str(sentiment_report.get("dominant_themes",           "")),
        str(economist_report.get("summary",                   "")),
        str(economist_report.get("macro_keywords_detected",   "")),
        str(economist_report.get("macro_regime",              "")),
    ]).lower()

    triggered = [kw for kw in _SYSTEMIC_FAILURE_KEYWORDS if kw.lower() in scan_text]
    if triggered:
        reason = f"Systemic keywords: {triggered[:3]}"
        logger.warning(f"[CIO Agent] Systemic failure keywords detected: {triggered}")
        return True, reason

    return False, ""


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Bạn là Giám đốc Đầu tư (CIO) của một quỹ đầu tư chứng khoán hàng đầu Việt Nam với AUM $500 triệu.

Bạn vừa nhận được báo cáo từ 3 chuyên gia cùng với TRỌNG SỐ được tính toán động:
  1. Quant Analyst (Phân tích kỹ thuật + XGBoost)
  2. Sentiment Analyst (Tâm lý thị trường)
  3. Macro Economist (Kinh tế vĩ mô)

NHIỆM VỤ:
1. Đánh giá mức độ ĐỒNG THUẬN giữa 3 báo cáo (STRONG / MIXED / CONFLICTED)
2. Áp dụng trọng số được cung cấp để tính điểm tổng hợp
3. Nếu có MÂU THUẪN giữa các agent: ưu tiên agent có confidence cao hơn và trọng số lớn hơn
4. Ra MỘT QUYẾT ĐẦU TƯ DUY NHẤT rõ ràng

PHONG CÁCH: Súc tích, chuyên nghiệp, quyết đoán.
Một nhà đầu tư cần đọc xong trong 30 giây và hiểu ngay cần làm gì.
Không mơ hồ. Không "có thể là". Luôn có một kết luận rõ ràng.

OUTPUT FORMAT (JSON thuần túy):
{
  "direction": "TĂNG | GIẢM | ĐI NGANG",
  "confidence_score": 0.72,
  "consensus": "STRONG | MIXED | CONFLICTED",
  "weight_regime": "BALANCED | MACRO_DOMINANT | QUANT_DOMINANT",
  "weights_used": {"quant": 0.40, "sentiment": 0.30, "macro": 0.30},
  "weighted_score": 0.25,
  "key_signals": [
    "XGBoost P(Tăng)=72% + RSI chưa overbought → Bullish setup",
    "Sentiment tiêu cực nhẹ (score=-0.2) → Rủi ro ngắn hạn",
    "FED giữ lãi suất → Áp lực WACC ổn định"
  ],
  "risk_factors": ["Volume thấp → Tín hiệu yếu", "Macro không chắc chắn"],
  "reasoning": "3-5 câu tổng hợp lý luận CIO",
  "action": "MUA | BÁN | CHỜ ĐỢI",
  "stop_loss_note": "Ngắn gọn về mức rủi ro cần chú ý"
}"""

# ---------------------------------------------------------------------------
# Dynamic Weight Calculator
# ---------------------------------------------------------------------------

def _determine_weight_regime(state: AgentState) -> tuple[WeightRegime, dict[str, float]]:
    """
    Determine weight regime based on:
      1. Macro keyword count from Economist Agent
      2. Market volatility proxy (BB width / ATR)
      3. Sentiment score
    """
    economist_report = state.get("economist_report", {})
    indicators       = state.get("technical_indicators", {})
    sentiment_report = state.get("sentiment_report", {})

    macro_keywords  = economist_report.get("macro_keywords_detected", [])
    bb_width        = float(indicators.get("bb_width", 0) or 0)
    sentiment_score = float(sentiment_report.get("sentiment_score", 0) or 0)

    # Rule 1: High-impact macro keywords → macro dominates
    if len(macro_keywords) >= 2:
        logger.info(
            f"[CIO] Macro keywords detected ({len(macro_keywords)}): "
            f"{macro_keywords[:3]} → MACRO_DOMINANT weights"
        )
        return "MACRO_DOMINANT", _WEIGHTS["MACRO_DOMINANT"]

    # Rule 2: Low volatility + neutral sentiment → quant signal most reliable
    # BB width < 0.02 typically signals a tight-range market
    low_volatility  = 0 < bb_width < 0.02
    neutral_news    = abs(sentiment_score) < 0.15
    if low_volatility and neutral_news:
        logger.info(
            f"[CIO] Low volatility (BB_width={bb_width:.4f}) + neutral news "
            f"(score={sentiment_score}) → QUANT_DOMINANT weights"
        )
        return "QUANT_DOMINANT", _WEIGHTS["QUANT_DOMINANT"]

    # Default: balanced
    logger.info("[CIO] Using BALANCED weights (default regime)")
    return "BALANCED", _WEIGHTS["BALANCED"]


def _compute_weighted_score(state: AgentState, weights: dict[str, float]) -> float:
    """
    Compute a scalar score in [-1, +1] by mapping each agent's recommendation
    to a numeric value and applying weights.

    Mapping:
      BUY/POSITIVE/BULLISH    →  +1.0
      SELL/NEGATIVE/BEARISH   →  -1.0
      NEUTRAL/MIXED/CAUTIOUS  →   0.0
    """
    _score_map = {
        "buy": 1.0, "mua": 1.0, "bullish": 1.0, "positive": 1.0, "tăng": 1.0,
        "sell": -1.0, "bán": -1.0, "bearish": -1.0, "negative": -1.0, "giảm": -1.0,
        "neutral": 0.0, "mixed": 0.0, "cautious": 0.0, "đi ngang": 0.0, "chờ": 0.0,
    }

    def _to_score(value: str) -> float:
        return _score_map.get(value.lower().strip(), 0.0)

    q_rec  = state.get("quant_report",     {}).get("recommendation",    "NEUTRAL")
    s_sent = state.get("sentiment_report", {}).get("overall_sentiment",  "NEUTRAL")
    m_sent = state.get("economist_report", {}).get("macro_sentiment",    "NEUTRAL")

    score = (
        _to_score(q_rec)  * weights["quant"]     +
        _to_score(s_sent) * weights["sentiment"]  +
        _to_score(m_sent) * weights["macro"]
    )
    return round(score, 4)


def _score_to_direction(score: float) -> tuple[str, float]:
    """Convert weighted score to (direction, confidence_score)."""
    abs_score = abs(score)
    if score > 0.15:
        return "TĂNG", min(0.5 + abs_score * 0.5, 0.95)
    elif score < -0.15:
        return "GIẢM", min(0.5 + abs_score * 0.5, 0.95)
    else:
        return "ĐI NGANG", max(0.5 - abs_score, 0.35)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_human_message(
    state: AgentState, regime: WeightRegime, weights: dict, score: float,
    disc_sentiment: float = 0.0, disc_macro: float = 0.0,
) -> str:
    q = state.get("quant_report",     {})
    s = state.get("sentiment_report", {})
    m = state.get("economist_report", {})

    direction_hint, conf_hint = _score_to_direction(score)
    long_count = s.get("long_term_article_count", "N/A")

    return f"""BÁO CÁO TỪ 3 CHUÊN GIA — VN-INDEX FORECAST

=== CẤU HÌNH TRỌNG SỐ ===
Chế độ: {regime}
Quant Agent:     {weights['quant']*100:.0f}%
Sentiment Agent: {weights['sentiment']*100:.0f}%
Macro Economist: {weights['macro']*100:.0f}%
Điểm tổng hợp có trọng số: {score:+.4f} (thang -1 đến +1)
Gợi ý sơ bộ từ scoring: {direction_hint} (confidence sơ bộ: {conf_hint:.0%})

=== BÁO CÁO QUANT ANALYST (weight={weights['quant']*100:.0f}%) ===
Khuyến nghị:     {q.get('recommendation', 'N/A')}
Confidence ML:   {q.get('confidence', 0)}%
Xu hướng KT:    {q.get('trend_assessment', 'N/A')}
Tín hiệu chính: {q.get('signals', [])}
Veto Signal:     {'🚨 CIRCUIT BREAKER KÍCH HOẠT' if q.get('veto_signal') else '✅ Bình thường'}
Tóm tắt:        {q.get('summary', 'N/A')}

=== BÁO CÁO SENTIMENT ANALYST (weight={weights['sentiment']*100:.0f}%) ===
Tâm lý tổng thể:    {s.get('overall_sentiment', 'N/A')}
Sentiment score:   {s.get('sentiment_score', 0)} (raw)
Disc. score (γ):   {disc_sentiment:+.4f} ← Đã chiết khấu tin dài hạn
Bài báo dài hạn:  {long_count} bài bị loại bỏ khỏi scoring
Fear/Greed:        {s.get('market_fear_greed', 'N/A')}
Chủ đề chính:    {s.get('dominant_themes', [])}
Tóm tắt:         {s.get('summary', 'N/A')}

=== BÁO CÁO MACRO ECONOMIST (weight={weights['macro']*100:.0f}%) ===
Macro regime:    {m.get('macro_regime', 'N/A')}
Tác động NH:    {m.get('short_term_impact', 'N/A')} (trực tiếp, γ=1.0)
Tác động DH:    {m.get('long_term_impact', 'N/A')} (chiết khấu, γ=0.1)
Disc. macro (γ):  {disc_macro:+.4f} ← 70% ngắn hạn + 30%×0.1 dài hạn
Macro sentiment: {m.get('macro_sentiment', 'N/A')}
Keywords:        {m.get('macro_keywords_detected', [])}
Tóm tắt:        {m.get('summary', 'N/A')}

Với tư cách CIO, hãy tổng hợp và đưa ra quyết định đầu tư cuối cùng theo JSON format yêu cầu."""


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

def cio_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node for CIO Agent.
    Reads: quant_report, sentiment_report, economist_report, technical_indicators
    Writes: final_decision
    """
    logger.info("[CIO Agent] Starting synthesis…")
    settings = get_settings()

    # ── STEP 0: CIO Veto — Systemic Failure Override ──────────────────────
    veto_triggered, veto_reason = _check_cio_veto(state)
    if veto_triggered:
        logger.warning(f"[CIO Agent] 🚨 CIO VETO TRIGGERED — Short-circuiting all scoring: {veto_reason}")
        decision = {
            "direction":        "GIẢM",
            "confidence_score": 0.97,
            "consensus":        "STRONG",
            "weight_regime":    "VETO_OVERRIDE",
            "weights_used":     {"quant": 0.0, "sentiment": 0.0, "macro": 1.0},
            "weighted_score":   -1.0,
            "key_signals":      [f"🚨 CIO VETO KÍCH HOẠT: {veto_reason}"],
            "risk_factors": [
                "Phát hiện sự kiện rủi ro hệ thống cực độ",
                "Toàn bộ phương trình scoring bị ngắt mạch (Short-circuited)",
            ],
            "reasoning": (
                f"CIO Veto kích hoạt do: {veto_reason}. "
                f"Theo nguyên tắc sinh tồn, toàn bộ điểm số XGBoost, "
                f"Sentiment và Macro bị ghi đè về mức âm tuyệt đối. "
                f"Hệ thống phát lệnh STRONG SELL vô điều kiện."
            ),
            "action":           "BÁN",
            "stop_loss_note":   "🚨 KHẨN CẤP: Thoát toàn bộ vị thế ngay lập tức. Rủi ro hệ thống cực cao.",
            "veto_triggered":   True,
            "veto_reason":      veto_reason,
        }
        return {"final_decision": decision}

    # ── STEP 1: Determine dynamic weights ────────────────────────────────
    regime, weights = _determine_weight_regime(state)

    # ── STEP 1.5: Temporal Discounting — γ Factor ─────────────────────────
    disc_sentiment, disc_macro, td_log = _apply_temporal_discount(state)
    logger.info(td_log)

    # ── STEP 2: Compute pre-LLM weighted score (with discounted inputs) ───
    score = _compute_weighted_score(state, weights)
    direction_hint, conf_hint = _score_to_direction(score)
    logger.info(f"[CIO Agent] Pre-LLM score={score:+.4f} → hint={direction_hint} ({conf_hint:.0%})")

    llm = ChatGroq(
        model       = settings.groq_model,
        api_key     = settings.groq_api_key,
        temperature = 0.15,
        max_tokens  = 1500,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_human_message(
            state, regime, weights, score,
            disc_sentiment=disc_sentiment,
            disc_macro=disc_macro,
        )),
    ]

    try:
        raw   = _call_llm(llm, messages)
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        decision = json.loads(clean)

        # Ensure computed values are embedded (LLM may override, but these are ground truth)
        decision["weight_regime"]   = regime
        decision["weights_used"]    = weights
        decision["weighted_score"]  = score

        logger.success(
            f"[CIO Agent] FINAL DECISION → {decision.get('direction')} "
            f"| confidence={decision.get('confidence_score')} "
            f"| action={decision.get('action')} "
            f"| regime={regime}"
        )
    except json.JSONDecodeError as exc:
        logger.error(f"[CIO Agent] JSON parse failed: {exc}")
        decision = {
            "direction":        direction_hint,
            "confidence_score": conf_hint,
            "consensus":        "MIXED",
            "weight_regime":    regime,
            "weights_used":     weights,
            "weighted_score":   score,
            "key_signals":      ["Parse error — using mechanical score"],
            "risk_factors":     ["CIO LLM parse failed"],
            "reasoning":        f"Fallback to mechanical scoring: {score:+.4f}",
            "action":           "CHỜ ĐỢI",
            "stop_loss_note":   "N/A",
        }
    except Exception as exc:
        logger.error(f"[CIO Agent] LLM call failed: {exc}")
        decision = {
            "direction":        direction_hint,
            "confidence_score": conf_hint,
            "consensus":        "MIXED",
            "weight_regime":    regime,
            "weights_used":     weights,
            "weighted_score":   score,
            "key_signals":      [f"LLM unavailable: {exc}"],
            "risk_factors":     ["CIO Agent error — mechanical fallback used"],
            "reasoning":        f"Mechanical score fallback: weighted_score={score:+.4f}",
            "action":           "CHỜ ĐỢI",
            "stop_loss_note":   "N/A",
        }
        errors = list(state.get("error_messages", []))
        errors.append(f"CIO Agent error: {exc}")
        return {"final_decision": decision, "error_messages": errors}

    return {"final_decision": decision}
