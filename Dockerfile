# ============================================================
# UFM Admission Bot — Multi-stage Dockerfile
# Stage 1: Builder (cài dependencies)
# Stage 2: Runtime (image nhẹ, chỉ copy artifacts)
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Cài system dependencies cần thiết cho numpy, psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước (tận dụng Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================
# STAGE 2: Runtime — Image nhẹ (~150MB thay vì ~800MB)
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="UFM Tech Team <tech@ufm.edu.vn>"
LABEL description="UFM Admission RAG Chatbot — Production Image"

WORKDIR /app

# Chỉ cài runtime lib (không có gcc, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages từ builder stage
COPY --from=builder /install /usr/local

# Copy source code
COPY . .

# Biến môi trường mặc định
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Port mặc định cho FastAPI
EXPOSE 8000

# Entrypoint: chạy uvicorn
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
