# 🤖 Module 2: Multi-Agent System — LangGraph Orchestration

> **Module:** LangGraph Multi-Agent (CIO, Quant, Sentiment, Economist)  
> **Phiên bản:** v1.0 | **Ngày:** 2026-05-06

---

## 1. Tổng quan Hệ thống Đa tác nhân

Hệ thống Multi-Agent được điều phối bởi **LangGraph StateGraph**, bao gồm 4 Agent chuyên biệt hoạt động theo cơ chế **parallel fan-out → aggregate → synthesize**. Mọi Agent sử dụng **Llama 3.3-70b-versatile** qua Groq API.

```
┌──────────────────────────────────────────────────────────────┐
│                   LANGGRAPH STATEGRAPH                       │
│                                                              │
│  [START] → load_context_node                                 │
│                    │                                         │
│         ┌──────────┼──────────┐                             │
│         ▼          ▼          ▼                              │
│   [Quant]    [Sentiment]  [Economist]   ← PARALLEL           │
│      Agent      Agent        Agent                           │
│         │          │          │                              │
│         └──────────┼──────────┘                             │
│                    ▼                                         │
│           [aggregate_reports_node]                           │
│                    │                                         │
│               [CIO Agent]          ← SYNTHESIZER             │
│                    │                                         │
│                  [END]                                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. AgentState — Shared Memory Schema

`AgentState` là `TypedDict` được truyền xuyên suốt toàn bộ graph. Mỗi node đọc và ghi vào state này.

```python
# backend/agents/state.py (định nghĩa cấu trúc)

AgentState:
  # === INPUT CONTEXT ===
  ticker:               str          # "^VNINDEX"
  analysis_date:        str          # "2026-05-07"

  # === MARKET DATA (từ Scraper + Feature Engineer) ===
  ohlcv_summary:        dict         # {close, volume, return_1d, ...}
  technical_indicators: dict         # {RSI_14, MACD, BB_width, ATR_14, ...}

  # === ML OUTPUT (từ XGBoost Predictor) ===
  ml_prediction:        dict         # {p_up, p_down, direction, confidence}

  # === UNSTRUCTURED DATA ===
  news_articles:        list[dict]   # [{title, summary, source, date}, ...]
  macro_context:        list[str]    # Chunks từ ChromaDB RAG

  # === AGENT OUTPUTS (ghi bởi từng agent) ===
  quant_report:         str          # Báo cáo từ Quant Agent
  sentiment_report:     str          # Báo cáo từ Sentiment Agent
  economist_report:     str          # Báo cáo từ Economist Agent

  # === FINAL OUTPUT (ghi bởi CIO Agent) ===
  final_decision:       dict         # {direction, confidence, reasoning, signals}
  error_messages:       list[str]    # Error log nếu có
```

---

## 3. Agent 1: Quant Agent

### 3.1 Định nghĩa

| Thuộc tính | Nội dung |
|-----------|---------|
| **Role** | Chuyên gia Phân tích Kỹ thuật (Quantitative Analyst) |
| **Goal** | Diễn giải tín hiệu kỹ thuật từ XGBoost + Technical Indicators thành ngôn ngữ phân tích có cấu trúc |
| **Input từ State** | `ml_prediction`, `technical_indicators`, `ohlcv_summary` |
| **Output vào State** | `quant_report` (string JSON có cấu trúc) |

### 3.2 Tools

| Tool | Mô tả | Cách dùng |
|------|--------|-----------|
| `read_ml_prediction` | Đọc output XGBoost từ State | Lấy `p_up`, `p_down`, `confidence` |
| `read_technical_indicators` | Đọc dict indicators từ State | Lấy RSI, MACD, BB, ADX, ATR |
| `interpret_signals` | Logic rule-based (Python function) | Phân loại RSI oversold/overbought, MACD cross |

### 3.3 System Prompt (Template)

```
Bạn là một Chuyên gia Phân tích Kỹ thuật với 15 năm kinh nghiệm giao dịch.

Nhiệm vụ: Phân tích dữ liệu kỹ thuật VN-Index và đưa ra báo cáo ngắn gọn.

Quy tắc:
- Luôn tham chiếu con số cụ thể (RSI=X, MACD=Y)
- Chỉ ra 2-3 tín hiệu quan trọng nhất
- Kết luận bằng: BUY / SELL / NEUTRAL + độ tin cậy (%)
- Không tự ý tạo dữ liệu nếu thiếu thông tin

Định dạng output (JSON):
{
  "signals": ["RSI=72 → Overbought", "MACD cross bearish"],
  "trend": "BEARISH",
  "ml_alignment": "XGBoost P(Tăng)=0.38 → Consistent với tín hiệu kỹ thuật",
  "recommendation": "SELL",
  "confidence": 72,
  "summary": "..."
}
```

---

## 4. Agent 2: Sentiment Agent

### 4.1 Định nghĩa

| Thuộc tính | Nội dung |
|-----------|---------|
| **Role** | Chuyên gia Phân tích Cảm xúc Thị trường (Market Sentiment Analyst) |
| **Goal** | Đánh giá tâm lý nhà đầu tư và xu hướng tin tức tài chính, phân loại Positive/Negative/Neutral |
| **Input từ State** | `news_articles` (list các bài báo đã crawl) |
| **Output vào State** | `sentiment_report` |

### 4.2 Tools

| Tool | Mô tả | Cách dùng |
|------|--------|-----------|
| `read_news_articles` | Đọc `news_articles` từ State | Lấy title + summary |
| `score_sentiment` | Zero-shot classification via LLM | Phân loại từng bài báo |
| `aggregate_sentiment` | Python function | Tính overall sentiment score |

### 4.3 News Scraper Target Sources

```
Nguồn tin cào (BeautifulSoup4 + httpx):
  - cafef.vn/chung-khoan (Cafef Chứng khoán)
  - tinnhanhchungkhoan.vn
  - vietstock.vn/tin-tuc
  - vneconomy.vn/chung-khoan

Mỗi bài báo lưu:
  {
    "title": "...",
    "summary": "...",   # 2-3 câu đầu
    "source": "cafef",
    "url": "...",
    "published_at": "2026-05-06"
  }
```

### 4.4 System Prompt (Template)

```
Bạn là chuyên gia phân tích tâm lý thị trường chứng khoán Việt Nam.

Nhiệm vụ: Đọc các tiêu đề & tóm tắt tin tức sau và đánh giá tổng thể sentiment thị trường.

Quy tắc:
- Đếm số lượng tin tích cực / tiêu cực / trung lập
- Chú ý từ khóa quan trọng: bán ròng, mua ròng, tăng vốn, thua lỗ, kỷ lục, bán tháo
- Xác định chủ đề nổi bật nhất trong ngày
- Không suy diễn vượt quá nội dung tin tức

Output (JSON):
{
  "positive_count": 5,
  "negative_count": 8,
  "neutral_count": 3,
  "overall_sentiment": "NEGATIVE",
  "dominant_themes": ["Bán ròng khối ngoại", "Lo ngại lãi suất FED"],
  "sentiment_score": -0.35,   # -1.0 đến +1.0
  "summary": "..."
}
```

---

## 5. Agent 3: Macro Economist Agent

### 5.1 Định nghĩa

| Thuộc tính | Nội dung |
|-----------|---------|
| **Role** | Nhà kinh tế Vĩ mô (Macro Economist) |
| **Goal** | Phân tích bối cảnh kinh tế vĩ mô ảnh hưởng đến thị trường VN-Index dựa trên kiến thức được truy xuất từ ChromaDB |
| **Input từ State** | `macro_context` (chunks từ RAG), `analysis_date` |
| **Output vào State** | `economist_report` |

### 5.2 Tools

| Tool | Mô tả | Cách dùng |
|------|--------|-----------|
| `query_chromadb` | Semantic search trong ChromaDB | Query: "VN-Index macro factors 2026", "FED rate decision", "Vietnam GDP growth" |
| `read_macro_context` | Đọc `macro_context` từ State | Lấy pre-retrieved chunks |
| `cross_reference_indicators` | Kết hợp nhiều chỉ số vĩ mô | VND/USD, CPI, tăng trưởng tín dụng |

### 5.3 RAG Query Strategy

```
ChromaDB Collections được truy vấn:
  1. macro_reports    → Query: "Vietnam economic outlook 2026 Q2"
  2. macro_reports    → Query: "FED interest rate decision impact emerging markets"
  3. market_news      → Query: "VN-Index foreign investor sentiment May 2026"

Top-K: 5 chunks per query
Similarity threshold: 0.75
```

### 5.4 System Prompt (Template)

```
Bạn là Nhà kinh tế Vĩ mô chuyên về thị trường Đông Nam Á, đặc biệt Việt Nam.

Nhiệm vụ: Dựa vào ngữ cảnh vĩ mô được cung cấp, đánh giá ảnh hưởng lên VN-Index.

Phân tích các yếu tố:
- Chính sách tiền tệ (FED, NHNN Việt Nam)
- Tỷ giá VND/USD, dòng vốn ngoại
- Tăng trưởng GDP, CPI, tăng trưởng tín dụng
- Địa chính trị (thương mại quốc tế, xuất khẩu)

Quy tắc: Chỉ kết luận từ thông tin được cung cấp. Nêu rõ "Không đủ dữ liệu" nếu thiếu.

Output (JSON):
{
  "key_macro_factors": ["FED giữ lãi suất cao → áp lực VND", "..."],
  "macro_sentiment": "CAUTIOUS",
  "impact_on_market": "NEGATIVE",
  "confidence": 65,
  "summary": "..."
}
```

---

## 6. Agent 4: CIO Agent (Chief Investment Officer)

### 6.1 Định nghĩa

| Thuộc tính | Nội dung |
|-----------|---------|
| **Role** | Giám đốc Đầu tư (Chief Investment Officer - CIO) |
| **Goal** | Tổng hợp báo cáo từ 3 Agent, cân nhắc trọng số, ra quyết định đầu tư cuối cùng với lý luận rõ ràng |
| **Input từ State** | `quant_report`, `sentiment_report`, `economist_report`, `ml_prediction` |
| **Output vào State** | `final_decision` |

### 6.2 Tools

| Tool | Mô tả | Cách dùng |
|------|--------|-----------|
| `read_all_reports` | Đọc 3 báo cáo từ State | Tổng hợp signals |
| `weigh_signals` | Tính weighted average | Quant:40% + Sentiment:30% + Macro:30% |
| `format_final_report` | Cấu trúc output chuẩn | Pydantic validation |

### 6.3 Trọng số Tổng hợp

```
Final Score = (Quant Score × 0.40) + (Sentiment Score × 0.30) + (Macro Score × 0.30)

Mapping Score → Direction:
  Score > +0.20  → TĂNG (HIGH)
  Score > +0.05  → TĂNG (MEDIUM)
  -0.05 to +0.05 → ĐI NGANG
  Score < -0.05  → GIẢM (MEDIUM)
  Score < -0.20  → GIẢM (HIGH)
```

### 6.4 System Prompt (Template)

```
Bạn là Giám đốc Đầu tư (CIO) của một quỹ đầu tư lớn tại Việt Nam.

Bạn nhận được báo cáo từ 3 chuyên gia:
1. Quant Analyst (trọng số 40%): phân tích kỹ thuật + ML
2. Sentiment Analyst (trọng số 30%): tâm lý thị trường
3. Macro Economist (trọng số 30%): vĩ mô

Nhiệm vụ:
- Đánh giá sự đồng thuận / mâu thuẫn giữa 3 báo cáo
- Nếu có mâu thuẫn, ưu tiên báo cáo có confidence cao hơn
- Đưa ra 1 quyết định duy nhất với lý luận rõ ràng

Phong cách: Súc tích, chuyên nghiệp, không mơ hồ. Một nhà đầu tư cần đọc trong 30 giây.

Output (JSON):
{
  "direction": "TĂNG | GIẢM | ĐI NGANG",
  "confidence_score": 0.0 - 1.0,
  "consensus": "STRONG | MIXED | CONFLICTED",
  "key_signals": ["...", "...", "..."],
  "risk_factors": ["...", "..."],
  "reasoning": "Tổng hợp 3-5 câu súc tích",
  "action": "MUA | BÁN | CHỜ ĐỢI"
}
```

---

## 7. LangGraph Implementation — `agents/graph.py`

### 7.1 Graph Construction

```python
# Cấu trúc graph (minh họa logic, không phải code chính thức)

graph = StateGraph(AgentState)

# Nodes
graph.add_node("load_context", load_context_node)
graph.add_node("quant_agent", quant_agent_node)
graph.add_node("sentiment_agent", sentiment_agent_node)
graph.add_node("economist_agent", economist_agent_node)
graph.add_node("aggregate", aggregate_reports_node)
graph.add_node("cio_agent", cio_agent_node)

# Edges — Parallel execution
graph.add_edge(START, "load_context")
graph.add_edge("load_context", "quant_agent")
graph.add_edge("load_context", "sentiment_agent")
graph.add_edge("load_context", "economist_agent")
graph.add_edge("quant_agent", "aggregate")
graph.add_edge("sentiment_agent", "aggregate")
graph.add_edge("economist_agent", "aggregate")
graph.add_edge("aggregate", "cio_agent")
graph.add_edge("cio_agent", END)

app_graph = graph.compile()
```

### 7.2 Error Handling Strategy

```
Nếu một Agent thất bại (LLM timeout, API error):
  → Ghi lỗi vào state.error_messages
  → Tiếp tục với báo cáo placeholder: "Agent unavailable"
  → CIO Agent vẫn tổng hợp từ các Agent còn lại
  → confidence_score giảm tương ứng với số Agent thất bại
```

---

## 8. RAG Engine — `rag_engine/`

### 8.1 Document Ingestion Pipeline

```
PDF/TXT/DOCX (báo cáo vĩ mô, phân tích)
        │
        ▼
unstructured.partition() → elements[]
        │
        ▼
Chunking: RecursiveCharacterTextSplitter
  chunk_size=512, chunk_overlap=64
        │
        ▼
sentence-transformers/all-MiniLM-L6-v2
  → embeddings (384-dim)
        │
        ▼
ChromaDB.add_documents()
  collection: "macro_reports"
```

### 8.2 Retrieval Strategy

```
Query → embed query → ChromaDB.query(n_results=5)
  → Retrieved chunks[]
  → Format thành context string
  → Ghi vào AgentState.macro_context
```

---

## 9. Dependency Map — Module 2

```
backend/
  agents/
    state.py           → TypedDict definition
    graph.py           → LangGraph StateGraph
    quant_agent.py     → langchain_groq.ChatGroq + state read
    sentiment_agent.py → langchain_groq.ChatGroq + news parsing
    economist_agent.py → langchain_groq.ChatGroq + ChromaDB query
    cio_agent.py       → langchain_groq.ChatGroq + weighted synthesis

  rag_engine/
    embedder.py   → sentence_transformers.SentenceTransformer
    indexer.py    → unstructured + chromadb
    retriever.py  → chromadb.Collection.query()

  scraper/
    news_scraper.py → httpx + bs4.BeautifulSoup
```

---

## 10. Luồng Tích hợp với Module 1

```
Module 1 (XGBoost) Output:
  ml_prediction = {p_up: 0.72, p_down: 0.28, direction: "TĂNG", confidence: "HIGH"}
        │
        ▼ Ghi vào AgentState.ml_prediction
        │
Module 2 (LangGraph):
  Quant Agent đọc ml_prediction + technical_indicators
  → Phân tích alignment: "XGBoost đồng thuận với tín hiệu RSI"
  → Báo cáo: "STRONG BUY signal từ cả ML và TA"
```

> Hai module **không phụ thuộc trực tiếp vào nhau về code** — giao tiếp hoàn toàn qua `AgentState` dict, đảm bảo loose coupling và testability độc lập.
