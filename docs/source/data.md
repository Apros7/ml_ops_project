## Data

### Datasets

This project uses two public datasets:

- [CCPD2019](https://www.kaggle.com/datasets/binh234/ccpd2019) (primary)
- [ALPR License Plates](https://www.kaggle.com/datasets/raj4126/alpr-license-plates) (additional)

For training and evaluation we rely on CCPD-style filenames (the bounding box + plate text is encoded in the filename).

### Layout

The code supports two common layouts:

- **Pre-split folders**: `data_dir/train/`, `data_dir/val/` (and optionally `data_dir/test/`)
- **Flat folder**: `data_dir/*.jpg` with explicit split files

Split files live in `data/splits/` (e.g. `train.txt`, `val.txt`, `test.txt`).

### Data versioning (DVC)

The full `data/` directory is tracked by DVC (`data.dvc`). Typical workflows:

```bash
uv run dvc status
uv run dvc pull
uv run dvc push
```

If your remote needs credentials, configure them locally (this is not committed):

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
uv run dvc remote modify --local storage credentialpath "$GOOGLE_APPLICATION_CREDENTIALS"
```

### Export to YOLO format (optional)

The detector training can export CCPD images into YOLO-format labels (Ultralytics-compatible).

```bash
uv run python -m ml_ops.data data/ccpd_tiny data/processed/yolo
```
