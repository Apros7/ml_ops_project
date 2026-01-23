## Quickstart

### Install dependencies

```bash
uv sync --locked --dev
```

### Data (no DVC required for a demo)

This repo includes a small CCPD subset at `data/ccpd_tiny/`, so you can run training/evaluation and the API locally
without pulling from DVC.

### Optional: Pull the full dataset (DVC)

The dataset is tracked with DVC, but the remote is private. To use `dvc pull`, you need a working `.dvc/config` /
service account setup. Please contact the project authors for access.

Once you have credentials, you can pull the exact dataset version with:

```bash
uv run dvc pull
```

If your DVC remote requires a service account key, set it locally (this writes to `.dvc/config.local`, not git):

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
uv run dvc remote modify --local storage credentialpath "$GOOGLE_APPLICATION_CREDENTIALS"
uv run dvc pull
```

### Run the backend (FastAPI)

```bash
uv run invoke api
```

Open:

- **Swagger UI**: `http://localhost:8000/docs`
- **Health**: `http://localhost:8000/health`
- **Metrics**: `http://localhost:8000/metrics`

### Try it (local inference)

Pick any image from the bundled tiny dataset and send it to the API:

```bash
IMG=$(ls data/ccpd_tiny/val/*.jpg | head -n 1)
curl -X POST "http://localhost:8000/recognize" -F "file=@${IMG}" -F "conf_threshold=0.25"
```

### Run the frontend (Streamlit)

In a second terminal:

```bash
uv run invoke frontend --port 8501
```

Open `http://localhost:8501`.

### Run tests and linting

```bash
uv run invoke test
uv run invoke lint
uv run invoke format
```
