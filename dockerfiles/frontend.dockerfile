FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# System deps for Streamlit and image processing
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

# Copy sample images for frontend
COPY assets assets/

# Install local package as well
RUN uv sync --locked

# Streamlit listens on 8501 inside the container
EXPOSE 8501

# Set backend URL (will be overridden by Cloud Run env var)
ENV BACKEND_URL=http://localhost:8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start Streamlit app
CMD ["uv", "run", "streamlit", "run", "src/ml_ops/frontend.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
