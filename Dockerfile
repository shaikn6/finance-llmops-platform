FROM python:3.11-slim

WORKDIR /app

# Install system deps for sentence-transformers / faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Pre-build FAISS index at container build time (optional — will rebuild on first run if missing)
# RUN MOCK_MODE=true python -c "from pipeline.ingestion import build_faiss_index; build_faiss_index(force_rebuild=True)"

# Default: run Streamlit dashboard
ENV MOCK_MODE=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
