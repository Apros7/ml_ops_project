# ml_ops

DTU ML Ops course project: license plate recognition.

## Project description

This repository trains and serves a two-stage pipeline:

- License plate detection (YOLOv8)
- License plate text recognition (CRNN + EasyOCR baselines)

## Data

We use two public datasets:

- https://www.kaggle.com/datasets/raj4126/alpr-license-plates
- https://www.kaggle.com/datasets/binh234/ccpd2019

## Project structure

The project is organized as follows:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Hydra configuration files
├── data/                     # Local datasets and processed data
├── dockerfiles/              # Dockerfiles for train/api/eval
├── docs/                     # MkDocs documentation
├── models/                   # Trained model artifacts
├── notebooks/                # Notebooks
├── reports/                  # Reports and figures
├── runs/                     # Training and inference runs
├── src/                      # Source code
│   └── ml_ops/
│       ├── api.py            # FastAPI service
│       ├── data.py           # Data utilities
│       ├── eval_easyocr.py   # EasyOCR evaluation
│       ├── evaluate.py       # Evaluation utilities
│       ├── make_small_dataset.py
│       ├── model.py          # Model components
│       ├── profile.py        # Profiling
│       ├── train.py          # Training CLI (Typer)
│       └── visualize.py      # Visualization helpers
├── tests/                    # Pytest suite
├── data.dvc                  # DVC data pipeline entry
├── pyproject.toml            # Python project metadata
├── README.md                 # Project README
└── tasks.py                  # Invoke task definitions
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Setup

This project uses `uv` for environment and dependency management.

```bash
uv venv
uv pip install -e ".[dev]"
```

## Common commands

List tasks:

```bash
uv run invoke --list
```

Train detector or OCR:

```bash
uv run invoke train-detector -- data/ccpd_tiny
uv run invoke train-ocr -- data/ccpd_tiny
```

Run the API:

```bash
uv run invoke api
```

Run tests and linting:

```bash
uv run invoke test
uv run invoke lint
uv run invoke format
```

Build or serve documentation:

```bash
uv run invoke build-docs
uv run invoke serve-docs
```

## Configuration and Hydra usage

Training scripts use Hydra configs stored in `configs/`:

```
configs/
├── config.yaml
├── data/                 # dataset presets (base, augmented, ...)
├── model/
│   ├── detector/         # YOLO variants (yolov8n, yolov8s, ...)
│   └── ocr/              # CRNN variants (crnn_default, crnn_full, ...)
├── training/
│   ├── detector/         # detector schedules (default, fast, ...)
│   └── ocr/              # OCR schedules (default, quick, ...)
└── wandb_configs/          # logging presets (default, disabled)
```

### Direct CLI overrides

Every Typer command in `src/ml_ops/train.py` accepts Hydra overrides via `-o/--override`:

```
uv run python -m ml_ops.train train-detector data/ccpd_tiny \
	-o model/detector=yolov8s -o training/detector=fast -o _wandb_configs_=disabled

uv run python -m ml_ops.train train-ocr data/ccpd_tiny \
	-o model/ocr=crnn_full -o training/ocr=quick
```

### Using Invoke tasks with overrides

Invoke tasks wrap the same commands. Use `--` to pass overrides through the task interface:

```
uv run invoke train-detector -- data/ccpd_tiny --override model/detector=yolov8s
uv run invoke train-ocr -- data/ccpd_tiny --override _wandb_configs_=disabled
uv run invoke train-both -- data/ccpd_tiny --override training/ocr=quick
uv run invoke train-detector -- data/ccpd_tiny --override model/detector=yolov8s --override training/detector=fast
```

You can chain multiple `--override key=value` pairs to mix dataset, model, training, and logging presets without editing config files. To add new presets, simply add new config files in the appropriate subdirectories under `configs/`.
