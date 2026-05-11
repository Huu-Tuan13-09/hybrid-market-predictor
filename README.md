# 🤖 Hybrid AI Market Predictor — VN-Index Forecasting System

> **Hệ thống AI Phân tích Vĩ mô và Dự báo Xu hướng Thị trường**  
> Kết hợp Machine Learning (XGBoost Time-series) + LLM Multi-Agent (LangGraph + Llama 3.3 via Groq)

---

## 🎯 Mục tiêu Dự án

Dự báo xu hướng VN-Index (Tăng / Giảm / Đi ngang) của **ngày giao dịch tiếp theo** bằng cách tích hợp hai nguồn trí tuệ:

1. **Quant ML Engine** — XGBoost học từ dữ liệu OHLCV lịch sử + Technical Indicators (RSI, MACD, BB, v.v.)
2. **Multi-Agent AI System** — Ba chuyên gia AI (Quant, Sentiment, Economist) hội tụ báo cáo cho CIO Agent ra quyết định cuối.

---

## 🏗️ Kiến trúc Tổng quan (Microservices)

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER BROWSER / CLIENT                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │  HTTP (port 8501)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              FRONTEND SERVICE — Streamlit (port 8501)           │
│  • Dashboard VN-Index     • Prediction Panel                    │
│  • Agent Reasoning View   • Historical Chart                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │  REST API calls (port 8000)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               BACKEND SERVICE — FastAPI (port 8000)             │
│                                                                 │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │  ML Engine   │  │  Multi-Agent    │  │   RAG Engine     │   │
│  │  (XGBoost)   │  │  (LangGraph)    │  │  (ChromaDB)      │   │
│  └──────────────┘  └─────────────────┘  └──────────────────┘   │
│  ┌──────────────┐  ┌─────────────────┐                          │
│  │   Scraper    │  │  Data Pipeline  │                          │
│  │ (yfinance +  │  │  (ta library)   │                          │
│  │  BS4)        │  │                 │                          │
│  └──────────────┘  └─────────────────┘                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌─────────────────────┐             ┌───────────────────────┐
│  ChromaDB Volume    │             │  Groq Cloud API       │
│  (Vector Store)     │             │  (Llama 3.3-70b)      │
└─────────────────────┘             └───────────────────────┘
```

---

## 📁 Cấu trúc Thư mục

```
hybrid_market_predictor/
│
├── README.md                       # File này
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Orchestration toàn hệ thống
├── .env.example                    # Template biến môi trường
│
├── docs/                           # Tài liệu thiết kế kỹ thuật
│   ├── architecture.md             # Sơ đồ kiến trúc chi tiết
│   ├── module_1_quant_ml.md        # Thiết kế ML Pipeline
│   └── module_2_multi_agent.md     # Thiết kế Multi-Agent System
│
├── data/                           # Persistent data volumes
│   ├── chromadb/                   # Vector database storage
│   └── models/                     # Trained ML models (.pkl, .json)
│
├── backend/                        # FastAPI Backend Service
│   ├── Dockerfile
│   ├── main.py                     # FastAPI entry point & router registry
│   ├── config.py                   # Settings, env vars (Pydantic BaseSettings)
│   ├── schemas.py                  # Pydantic request/response models
│   │
│   ├── agents/                     # LangGraph Multi-Agent System
│   │   ├── __init__.py
│   │   ├── graph.py                # LangGraph StateGraph definition
│   │   ├── state.py                # Shared AgentState TypedDict
│   │   ├── quant_agent.py          # Quant Agent node
│   │   ├── sentiment_agent.py      # Sentiment Agent node
│   │   ├── economist_agent.py      # Macro Economist Agent node
│   │   └── cio_agent.py            # CIO Synthesizer Agent node
│   │
│   ├── ml_models/                  # XGBoost ML Pipeline
│   │   ├── __init__.py
│   │   ├── feature_engineer.py     # Feature creation (ta library)
│   │   ├── trainer.py              # Model training & serialization
│   │   └── predictor.py            # Inference & probability output
│   │
│   ├── rag_engine/                 # RAG with ChromaDB
│   │   ├── __init__.py
│   │   ├── embedder.py             # Text embedding (HuggingFace)
│   │   ├── indexer.py              # Document ingestion & indexing
│   │   └── retriever.py            # Semantic search & context builder
│   │
│   └── scraper/                    # Data Ingestion Layer
│       ├── __init__.py
│       ├── market_data.py          # yfinance OHLCV fetcher
│       └── news_scraper.py         # BeautifulSoup4 news crawler
│
└── frontend/                       # Streamlit Frontend Service
    ├── Dockerfile
    ├── app.py                      # Main Streamlit application
    └── components/                 # UI components
        ├── charts.py               # Plotly charts
        ├── prediction_panel.py     # Prediction display
        └── agent_view.py           # Agent reasoning display
```

---

## 🛠️ Tech Stack

| Layer | Technology | Phiên bản |
|---|---|---|
| **Language** | Python | 3.10+ |
| **Backend API** | FastAPI + Uvicorn | 0.115+ |
| **Frontend UI** | Streamlit | 1.35+ |
| **ML Forecasting** | XGBoost + scikit-learn | 2.x / 1.5+ |
| **Technical Analysis** | ta (Technical Analysis) | 0.11+ |
| **Market Data** | yfinance | 0.2+ |
| **News Scraping** | BeautifulSoup4 + httpx | 4.12+ |
| **LLM Provider** | Groq API (Llama 3.3-70b) | Latest |
| **Agent Framework** | LangGraph | 0.2+ |
| **Vector Database** | ChromaDB | 0.5+ |
| **Embeddings** | sentence-transformers | 3.x |
| **Document Parsing** | unstructured | 0.15+ |
| **Containerization** | Docker + Docker Compose | 24+ |

---

## 🚀 Quick Start (Docker)

```bash
# 1. Clone và cấu hình môi trường
git clone <repo-url> && cd hybrid_market_predictor
cp .env.example .env
# Điền GROQ_API_KEY vào .env

# 2. Khởi tạo Dữ liệu và Huấn luyện Mô hình lần đầu
#    (Bước này chỉ cần chạy một lần duy nhất khi setup)
docker-compose run --rm backend python backend/rag_engine/indexer.py
docker-compose run --rm backend python backend/ml_models/trainer.py

# 3. Khởi động toàn bộ hệ thống
docker-compose up --build

# 4. Truy cập
# Frontend: http://localhost:8501
# Backend API Docs: http://localhost:8000/docs
```

---

## 🔑 Biến Môi trường

| Biến | Mô tả | Bắt buộc |
|---|---|---|
| `GROQ_API_KEY` | API key từ console.groq.com | ✅ |
| `CHROMA_DB_PATH` | Đường dẫn ChromaDB volume | ✅ |
| `MODEL_PATH` | Đường dẫn lưu XGBoost model | ✅ |
| `NEWS_SOURCES` | Danh sách URL nguồn tin tức | ⚙️ |
| `VNINDEX_TICKER` | Symbol VN-Index (mặc định: `^VNINDEX`) | ⚙️ |
| `LOOKBACK_DAYS` | Số ngày lịch sử dùng cho ML | ⚙️ |

---

## 🔄 Luồng Dự báo Tổng thể

```
[Trigger: User click "Run Analysis"]
        │
        ├─→ [Scraper] Fetch OHLCV 2 năm từ yfinance
        │
        ├─→ [Feature Engineer] Tính RSI, MACD, BB, ATR, v.v.
        │
        ├─→ [XGBoost Predictor] → P(Tăng), P(Giảm) + Confidence
        │
        ├─→ [News Scraper] Crawl tin tức tài chính hôm nay
        │
        ├─→ [RAG Engine] Query ChromaDB → Context vĩ mô
        │
        └─→ [LangGraph Orchestrator]
                ├── Quant Agent     → Phân tích kỹ thuật
                ├── Sentiment Agent → Phân tích cảm xúc tin tức
                ├── Economist Agent → Phân tích vĩ mô
                └── CIO Agent       → Tổng hợp → Final Decision
```

---

## 📄 Tài liệu Kỹ thuật

- 📐 [Kiến trúc Hệ thống](docs/architecture.md)
- 📊 [Module 1: Quant ML Pipeline](docs/module_1_quant_ml.md)
- 🤖 [Module 2: Multi-Agent System](docs/module_2_multi_agent.md)

---

## ⚠️ Disclaimer

> Đây là dự án nghiên cứu & học thuật. Các dự báo được tạo ra **không phải là lời khuyên đầu tư tài chính**. Thị trường chứng khoán chứa đựng rủi ro cao và không thể dự báo hoàn toàn chính xác bằng bất kỳ mô hình AI nào.
