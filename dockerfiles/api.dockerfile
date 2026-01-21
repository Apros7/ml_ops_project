FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

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

COPY pyproject.toml uv.lock ./
ENV UV_LINK_MODE=copy
RUN uv sync --locked --no-install-project

COPY src src/
COPY configs configs/
# COPY models models/
# COPY yolov8n.pt yolov8n.pt
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --locked
RUN mkdir -p logs
ENV LOG_LEVEL=INFO

# CMD ["uv", "run", "uvicorn", "ml_ops.api:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["sh", "-c", "uv run uvicorn ml_ops.api:app --host 0.0.0.0 --port $PORT"]

