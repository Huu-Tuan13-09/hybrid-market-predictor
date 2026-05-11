# 📐 Kiến trúc Hệ thống Hybrid AI — VN-Index Predictor

> **Phiên bản:** v1.0 | **Ngày:** 2026-05-06

---

## 1. Tổng quan Kiến trúc

Hệ thống theo mô hình **Microservices** gồm hai service độc lập giao tiếp qua REST API, chia sẻ lớp dữ liệu bền vững (ChromaDB + model files).

```
╔══════════════════════════════════════════════════════════╗
║          HYBRID AI MARKET PREDICTOR - ARCHITECTURE       ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  [TIER 1] PRESENTATION — Streamlit (Port 8501)           ║
║  Dashboard | Prediction Panel | Agent Trace | Settings   ║
║                        │ HTTP REST                       ║
║                        ▼                                 ║
║  [TIER 2] APPLICATION — FastAPI (Port 8000)              ║
║  /predict  /train  /market-data  /news  /health          ║
║     │                                                    ║
║  ┌──┴──────────────────────────────────────┐             ║
║  │         ORCHESTRATION LAYER             │             ║
║  │  ML Pipeline (XGBoost)                  │             ║
║  │  LangGraph Multi-Agent Orchestrator     │             ║
║  └──────────────────────────────────────────┘            ║
║                        │                                 ║
║  [TIER 3] DATA LAYER                                     ║
║  ChromaDB (Vectors) | Model Store | Groq / yfinance APIs ║
╚══════════════════════════════════════════════════════════╝
```

---

## 2. API Contract — Frontend → Backend

| Method | Endpoint | Mô tả | Response Key Fields |
|--------|----------|--------|---------------------|
| `GET` | `/health` | Health check | `status` |
| `POST` | `/predict` | Full analysis pipeline | `direction`, `confidence_score`, `agent_reports`, `cio_reasoning` |
| `POST` | `/train` | Re-train XGBoost | `accuracy`, `model_path` |
| `GET` | `/market-data` | OHLCV gần nhất | `ohlcv[]`, `indicators{}` |
| `GET` | `/news` | Tin tức đã crawl | `articles[]` |
| `GET` | `/rag/query` | Query ChromaDB | `results[]` |

### Pydantic Schema Chính

```
PredictRequest:
  ticker: str = "^VNINDEX"
  lookback_days: int = 504

PredictionResponse:
  ticker: str
  prediction_date: str
  direction: "TĂNG" | "GIẢM" | "ĐI NGANG"
  confidence_score: float          # 0.0 - 1.0
  ml_probability: {up: float, down: float}
  agent_reports: AgentReport[]     # Báo cáo từng Agent
  cio_reasoning: str               # Lý luận CIO tổng hợp
  processing_time_ms: int
```

---

## 3. Request Flow — POST /predict

```
Streamlit ──POST /predict──► FastAPI
                              │
                              ├─ yfinance → OHLCV DataFrame
                              ├─ Feature Engineering (ta lib)
                              ├─ XGBoost.predict_proba() → P(up), P(down)
                              ├─ News Scraper (BS4) → articles[]
                              ├─ ChromaDB.query() → macro_context[]
                              │
                              └─ LangGraph.invoke()
                                  ├─ [PARALLEL]
                                  │   ├── Quant Agent
                                  │   ├── Sentiment Agent
                                  │   └── Economist Agent
                                  ├─ [AGGREGATE] Merge reports
                                  └─ CIO Agent → final_decision{}

FastAPI ──PredictionResponse──► Streamlit
```

---

## 4. LangGraph StateGraph

```
START(load_context)
       │
  ┌────┴──────────────┐
  │   PARALLEL FORK   │
  ├── Quant Agent     │
  ├── Sentiment Agent │
  └── Economist Agent─┘
         │
  Aggregation Node (merge_reports)
         │
    CIO Agent (synthesize → final_decision)
         │
        END
```

### AgentState (TypedDict — Shared Memory)

```
AgentState:
  ticker:               str
  ohlcv_data:           dict
  technical_indicators: dict      # RSI, MACD, BB, ATR...
  ml_prediction:        dict      # XGBoost output
  news_articles:        list[dict]
  macro_context:        list[str] # RAG chunks
  quant_report:         str
  sentiment_report:     str
  economist_report:     str
  final_decision:       dict
```

---

## 5. ChromaDB Collections

| Collection | Nội dung | Embedding |
|---|---|---|
| `macro_reports` | Báo cáo GDP, lạm phát, FED | `all-MiniLM-L6-v2` |
| `market_news` | Tin tức tài chính | `all-MiniLM-L6-v2` |
| `company_filings` | Báo cáo tài chính niêm yết | `all-MiniLM-L6-v2` |

---

## 6. Docker Architecture

```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes: ["./data:/app/data"]   # ChromaDB + models persist

  frontend:
    build: ./frontend
    ports: ["8501:8501"]
    environment:
      BACKEND_URL: http://backend:8000
    depends_on: [backend]
```

> **Lưu ý:** ChromaDB chạy **embedded** trong FastAPI process — không cần container riêng, đơn giản hóa deployment mà vẫn đảm bảo persistence qua volume.

---

## 7. Bảo mật

- Secrets quản lý qua `.env`, không commit lên VCS
- `pydantic-settings` để load env vars type-safe
- CORS chỉ cho phép origin từ frontend service
- Rate limiting trên `/predict` để kiểm soát chi phí Groq API
