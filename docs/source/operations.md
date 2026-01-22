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

### Training container image

The training image is defined in `dockerfiles/train.dockerfile`. It uses `uv` + `uv.lock` for reproducible installs and
sets the container entrypoint to the Typer CLI:

- **Entrypoint**: `uv run -m ml_ops.train`

The image includes:

- `configs/` (Hydra configs)
- `.dvc/` + `data.dvc` (so you can run `dvc pull` inside the container when credentials are available)
- `data/ccpd_tiny/` (small dataset for smoke tests)

Build locally:

```bash
uv run invoke docker-build
```

Run training locally (persist outputs by mounting `runs/` and `models/`):

```bash
docker run --rm \
  -v "$PWD/runs:/app/runs" \
  -v "$PWD/models:/app/models" \
  train:latest train-both data/ccpd_tiny
```

If you need to run `dvc pull` inside the container, override the entrypoint:

```bash
docker run --rm --entrypoint sh train:latest -c "uv run dvc pull && uv run -m ml_ops.train train-both data/ccpd_tiny"
```

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
