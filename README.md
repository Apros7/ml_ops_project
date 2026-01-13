# ml_ops

DTU ML Ops course project

## Project Description

Overall goal of the project:

Read license plate from image

What data are you going to run on (initially, may change):

The data comes from 2 datasets, one with license plates from this source:
https://www.kaggle.com/datasets/raj4126/alpr-license-plates

And ine of chinese license plates from this source:
https://www.kaggle.com/datasets/binh234/ccpd2019

What models do you expect to use:

We are going to use object detection models such as YOLO and recogntion models based on CNNs.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Configuration & Hydra usage

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
└── wandb_configs/                # logging presets (default, disabled)
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
