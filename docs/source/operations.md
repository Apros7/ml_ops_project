## Operations

### CI/CD

Workflows live in `.github/workflows/` and cover:

- **Tests**: `pytest` + coverage (multi-OS matrix)
- **Linting/formatting**: `ruff`
- **Docs**: MkDocs build + GitHub Pages deploy
- **Data/model triggers**: workflows that run when DVC pointers or model artifacts change

### Cloud training (Vertex AI)

The repo includes a Cloud Build config that submits a Vertex AI Custom Job using the `train` container image:

- `configs/vertex_train.yaml` builds the job spec (injects secrets)
- `configs/vertex_config.yaml` defines the Vertex worker pool and runs:
  - `uv run dvc pull`
  - `uv run -m ml_ops.train train-both …`

Trigger from your machine (requires `gcloud` auth):

```bash
uv run invoke vertex-job
```

### Cloud deployment (Cloud Run)

For the backend API the `invoke` tasks wrap:

- Build + tag + push the image to Artifact Registry
- Deploy to Cloud Run

```bash
uv run invoke api-release
```

### Monitoring

The FastAPI service exposes Prometheus-style metrics at `GET /metrics`:

- HTTP request counters and latency histograms
- Active requests gauge
- Basic system metrics (CPU, memory, disk)

### Profiling

Profiling can be enabled via `ENABLE_PROFILING=true` and relevant Hydra settings.

```bash
export ENABLE_PROFILING=true
ENABLE_PROFILING=true uv run python -m ml_ops.train train-ocr data/ccpd_tiny
```
