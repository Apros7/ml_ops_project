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

WORKDIR /app

COPY pyproject.toml uv.lock ./
ENV UV_LINK_MODE=copy
RUN uv sync --locked --no-install-project

COPY src src/
COPY configs configs/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --locked

ENV PORT=8501
EXPOSE ${PORT}

ENV BACKEND_URL=http://localhost:8080

CMD ["sh", "-c", "uv run streamlit run src/ml_ops/frontend.py --server.port ${PORT} --server.address 0.0.0.0"]
