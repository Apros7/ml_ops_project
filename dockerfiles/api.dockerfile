FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# System deps – same as train, since YOLO/OpenCV also run here
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
        ffmpeg \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency metadata first
COPY pyproject.toml uv.lock ./

ENV UV_LINK_MODE=copy
RUN uv sync --locked --no-install-project


# Copy project code + configs
COPY src src/
COPY configs configs/
COPY README.md README.md
COPY LICENSE LICENSE

# Install local package as well
RUN uv sync --locked

# API listens on 8080 (Cloud Run default)
EXPOSE 8080

# Start FastAPI app via uvicorn
# Assumes you have `app = FastAPI(...)` in src/ml_ops/api.py
ENTRYPOINT ["/bin/sh", "-c", "uv run uvicorn ml_ops.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
