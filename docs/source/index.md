## License plate recognition (ml_ops)

Two-stage license plate recognition:

- **Detector**: YOLOv8 (Ultralytics)
- **OCR**: fine-tuned EasyOCR (recommended) or a CRNN baseline (for comparison)

### Quickstart

```bash
uv sync --locked --dev
uv run dvc pull

uv run invoke api
uv run streamlit run src/ml_ops/frontend.py --server.port 8501
```

### Guides

- [Architecture](architecture.md)
- [Data](data.md)
- [Training](training.md)
- [Evaluation](evaluation.md)
- [Serving](serving.md)
- [Operations](operations.md)

### Developer tooling

```bash
uv tool install pre-commit
pre-commit install
uv run invoke --list
```
