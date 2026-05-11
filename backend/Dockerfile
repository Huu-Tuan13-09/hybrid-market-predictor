# ============================================================
# Backend Dockerfile
# Base: python:3.10-slim for minimal image size
# ============================================================
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by unstructured + lxml + httpx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libssl-dev \
    poppler-utils \
    tesseract-ocr \
    tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --retries 5 --timeout 120 --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install in groups to handle network instability (BrokenPipe workaround)
# Group 1: Web framework (small, fast)
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
    fastapi==0.115.5 \
    uvicorn[standard]==0.32.1 \
    pydantic==2.9.2 \
    pydantic-settings==2.6.1 \
    python-multipart==0.0.12 \
    httpx==0.27.2 \
    python-dotenv==1.0.1 \
    loguru==0.7.2 \
    tenacity==9.0.0

# Group 2: Data & ML (large packages — xgboost, scikit-learn)
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    yfinance \
    curl_cffi \
    vnstock3 \
    vnstock \
    ta==0.11.0 \
    scikit-learn==1.5.2 \
    xgboost==2.1.3 \
    joblib==1.4.2

# Group 3: Scraping & Frontend
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
    beautifulsoup4==4.12.3 \
    lxml==5.3.0 \
    streamlit==1.40.2 \
    plotly==5.24.1

# Group 4: LangChain / LangGraph / Groq
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
    langchain==0.3.9 \
    langchain-groq==0.2.2 \
    langgraph==0.2.56 \
    langchain-community==0.3.9

# Group 5: RAG / Vector DB (large: sentence-transformers, chromadb, unstructured)
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
    chromadb==0.5.23 \
    sentence-transformers==3.3.1

RUN pip install --no-cache-dir --retries 5 --timeout 180 \
    "unstructured[pdf]==0.15.13"

# Copy application source code
COPY backend/ ./backend/

# Create data directories at build time (will be overridden by volume mounts at runtime)
RUN mkdir -p data/chromadb data/models data/seed_documents

# Set Python path so `from backend.xxx import` works from /app
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Entry point
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
