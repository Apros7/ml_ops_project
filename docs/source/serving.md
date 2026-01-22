## Serving

### Backend API (FastAPI)

Run locally:

```bash
uv run invoke api
```

Key endpoints:

- `GET /health`: health + model load status
- `POST /detect`: bounding boxes only
- `POST /recognize`: bounding boxes + plate text
- `GET /metrics`: Prometheus-style metrics
- `GET /docs`: Swagger UI

#### Model weights

By default the backend will:

- Prefer `models/yolo_best.pt` if present, otherwise fall back to `yolov8n.pt`
- Prefer fine-tuned EasyOCR weights at `models/ocr_best.pth` if present

The Docker image in `dockerfiles/api.dockerfile` copies `models/` into the container, so deploying with the tracked
artifacts makes the API self-contained.

### Frontend (Streamlit)

Run locally:

```bash
uv run streamlit run src/ml_ops/frontend.py --server.port 8501
```

To point the UI at a remote backend:

```bash
export BACKEND_URL=https://<cloud-run-service-url>
uv run streamlit run src/ml_ops/frontend.py --server.port 8501
```

### Docker

Build images locally:

```bash
uv run invoke docker-build
```

Run the API container (example):

```bash
docker run --rm -p 8080:8080 api:latest
```
