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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Work directory
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install Python dependencies from uv.lock
ENV UV_LINK_MODE=copy
RUN uv sync --locked --no-install-project

# Copy source code
COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE
COPY configs configs/

# Final dependency sync including local package
RUN uv sync --locked

# Copy model weights into the image (optional; useful for warm-start / offline runs)
COPY models models/

# Default entrypoint = training script
ENTRYPOINT ["uv", "run", "-m", "ml_ops.train"]
