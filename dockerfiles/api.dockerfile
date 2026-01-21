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
        curl \
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

# Copy model weights into the image (for offline inference)
COPY models models/

# Create logs directory
RUN mkdir -p logs

# Set log level
ENV LOG_LEVEL=INFO

# API listens on 8000 inside the container
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start FastAPI app via uvicorn with 2 workers
CMD ["uv", "run", "uvicorn", "ml_ops.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
