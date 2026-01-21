FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# System deps
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
        ffmpeg \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies
COPY pyproject.toml uv.lock ./
ENV UV_LINK_MODE=copy
RUN uv sync --locked --no-install-project

# Project code
COPY src src/
COPY configs configs/
COPY models models/
COPY yolov8n.pt yolov8n.pt
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --locked

RUN mkdir -p logs
ENV LOG_LEVEL=INFO

# Cloud Run port
ENV PORT=8080
EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uv run uvicorn ml_ops.api:app --host 0.0.0.0 --port ${PORT} --workers 2"]
