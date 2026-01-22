## Quickstart

### Install dependencies

```bash
uv sync --locked --dev
```

### Pull data (DVC)

The dataset is tracked with DVC. If you have access to the remote, pull the exact dataset version with:

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

### Run the frontend (Streamlit)

In a second terminal:

```bash
uv run streamlit run src/ml_ops/frontend.py --server.port 8501
```

Open `http://localhost:8501`.

### Run tests and linting

```bash
uv run invoke test
uv run invoke lint
uv run invoke format
```
