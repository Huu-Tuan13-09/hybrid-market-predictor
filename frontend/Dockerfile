# ============================================================
# Frontend Dockerfile — Streamlit
# ============================================================
FROM python:3.10-slim

WORKDIR /app

# Install tzdata for timezone support
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Only the packages Streamlit + charting need (lightweight image)
RUN pip install --no-cache-dir \
    streamlit==1.40.2 \
    plotly==5.24.1 \
    pandas==2.2.3 \
    httpx==0.27.2 \
    python-dotenv==1.0.1

# Copy frontend source
COPY frontend/ ./frontend/

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

# Streamlit config: disable browser auto-open, set port, disable telemetry
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
