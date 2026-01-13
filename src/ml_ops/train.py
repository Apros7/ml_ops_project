"""Training scripts for license plate detection and OCR."""

from pathlib import Path
from typing import Any

import typer
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import wandb
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from ml_ops.model import PlateDetector, PlateOCR
from ml_ops.data import CCPDDataModule, export_yolo_format, ENGLISH_NUM_CLASSES, NUM_CLASSES

app = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Weights & Biases configuration
WANDB_ENTITY = "ml_ops_number_plates"
WANDB_PROJECT = "license-plate-ocr"


def _normalize_path(path_value: Path | str | None) -> Path | None:
    """Convert strings to absolute paths relative to the project root."""

    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_hydra_config(
    config_path: Path = CONFIGS_DIR,
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load a Hydra configuration, clearing older state if needed."""

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_path)):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg


def _start_wandb_run(wandb_cfg: DictConfig, name: str, metadata: dict[str, Any]) -> bool:
    """Initialize WandB if enabled in the config."""

    enabled = bool(wandb_cfg.get("enabled", False))
    if not enabled:
        return False
    entity = wandb_cfg.get("entity", WANDB_ENTITY)
    project = wandb_cfg.get("project", WANDB_PROJECT)
    wandb.init(entity=entity, project=project, name=name, config=metadata)
    print(f"📊 Logging to Weights & Biases: {entity}/{project}")
    return True


def _finish_wandb_run(active: bool) -> None:
    """Finish WandB run if one was started."""

    if active:
        wandb.finish()


def get_accelerator(force_cpu: bool = False) -> tuple[str, str]:
    """Get the appropriate accelerator and device configuration.

    Args:
        force_cpu: If True, always use CPU (needed for ops not supported on MPS like CTC loss).

    Returns:
        Tuple of (accelerator, devices) for Lightning Trainer.
    """
    if force_cpu:
        return "cpu", "auto"
    if torch.cuda.is_available():
        return "gpu", "auto"
    elif torch.backends.mps.is_available():
        return "mps", "1"
    else:
        return "cpu", "auto"


def _train_detector_with_cfg(
    cfg: DictConfig,
    *,
    data_dir: Path | None = None,
    split_dir: Path | None = None,
    output_dir: Path | None = None,
    model_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    img_size: int | None = None,
    max_train_images: int | None = None,
    max_val_images: int | None = None,
    project: Path | None = None,
    name: str | None = None,
) -> tuple[Path, str]:
    """Train the detector using Hydra-provided defaults plus CLI overrides."""

    data_cfg = cfg.data
    training_cfg = cfg.training.detector
    model_cfg = cfg.model.detector
    wandb_cfg = cfg.wandb

    resolved_data_dir = _normalize_path(data_dir) or _normalize_path(data_cfg.get("data_dir")) or PROJECT_ROOT / "data"
    resolved_split_dir = _normalize_path(split_dir) or _normalize_path(data_cfg.get("split_dir"))
    resolved_output_dir = (
        _normalize_path(output_dir)
        or _normalize_path(data_cfg.get("processed_yolo_dir"))
        or PROJECT_ROOT / "data" / "processed" / "yolo"
    )
    project_path = _normalize_path(project) or _normalize_path(training_cfg.get("project_dir")) or RUNS_DIR / "detect"
    experiment_name = name or training_cfg.get("experiment_name", "plate_detection")

    model_name_value = model_name or model_cfg.get("model_name", "yolov8n.pt")
    epochs_value = epochs or training_cfg.get("epochs", 10)
    batch_size_value = batch_size or training_cfg.get("batch_size", 32)
    img_size_value = img_size or training_cfg.get("img_size", 320)

    train_split = float(data_cfg.get("train_split", 0.8))
    max_total = data_cfg.get("max_total_images")

    max_train_value = max_train_images or data_cfg.get("max_train_images")
    max_val_value = max_val_images or data_cfg.get("max_val_images")

    if max_train_value is None and max_total is not None:
        max_train_value = int(max_total * train_split)
    if max_val_value is None and max_total is not None and max_train_value is not None:
        max_val_value = max_total - max_train_value

    if max_train_value is None or max_val_value is None:
        raise ValueError("Specify max_train_images and max_val_images via config or CLI overrides.")

    project_path.mkdir(parents=True, exist_ok=True)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {project_path / experiment_name}")
    print("Step 1: Converting CCPD to YOLO format...")

    train_output = resolved_output_dir / "train"
    val_output = resolved_output_dir / "val"

    train_data_dir = resolved_data_dir / "train" if (resolved_data_dir / "train").exists() else resolved_data_dir
    val_data_dir = resolved_data_dir / "val" if (resolved_data_dir / "val").exists() else resolved_data_dir

    if train_data_dir != resolved_data_dir:
        print(f"Detected pre-split dataset: {resolved_data_dir}")
        print(f"  Train folder: {train_data_dir}")
        print(f"  Val folder: {val_data_dir}")

    train_split_file = resolved_split_dir / "train.txt" if resolved_split_dir else None
    val_split_file = resolved_split_dir / "val.txt" if resolved_split_dir else None

    export_yolo_format(train_data_dir, train_output, train_split_file, max_images=max_train_value)
    export_yolo_format(val_data_dir, val_output, val_split_file, max_images=max_val_value)

    print("Step 2: Creating data.yaml config...")
    data_yaml_path = resolved_output_dir / "data.yaml"
    data_yaml_content = f"""
path: {resolved_output_dir.absolute()}
train: train/images
val: val/images

names:
  0: license_plate
"""
    with open(data_yaml_path, "w") as f:
        f.write(data_yaml_content.strip())

    print("Step 3: Training YOLOv8...")

    wandb_active = _start_wandb_run(
        wandb_cfg,
        name=f"detector-{experiment_name}",
        metadata={
            "architecture": "YOLOv8",
            "task": "detection",
            "model_name": model_name_value,
            "epochs": epochs_value,
            "img_size": img_size_value,
            "batch_size": batch_size_value,
            "max_train_images": max_train_value,
            "max_val_images": max_val_value,
            "dataset": data_cfg.get("name", "CCPD"),
        },
    )

    detector = PlateDetector(
        model_name=model_name_value,
        num_classes=model_cfg.get("num_classes", 1),
        pretrained=bool(model_cfg.get("pretrained", True)),
    )

    accelerator, _ = get_accelerator()
    device = 0 if accelerator == "gpu" else "cpu"

    detector.train_yolo(
        data_yaml=str(data_yaml_path),
        epochs=epochs_value,
        imgsz=img_size_value,
        batch=batch_size_value,
        device=device,
        project=str(project_path),
        name=experiment_name,
    )

    _finish_wandb_run(wandb_active)

    print(f"\nTraining complete! Results saved to {project_path / experiment_name}")
    print(f"  - Training curves: {project_path / experiment_name / 'results.png'}")
    print(f"  - Metrics CSV: {project_path / experiment_name / 'results.csv'}")
    print(f"  - Best weights: {project_path / experiment_name / 'weights' / 'best.pt'}")
    if wandb_active:
        entity = wandb_cfg.get("entity", WANDB_ENTITY)
        project = wandb_cfg.get("project", WANDB_PROJECT)
        print(f"  - W&B Dashboard: https://wandb.ai/{entity}/{project}")

    return project_path, experiment_name


def _train_ocr_with_cfg(
    cfg: DictConfig,
    *,
    data_dir: Path | None = None,
    split_dir: Path | None = None,
    batch_size: int | None = None,
    max_epochs: int | None = None,
    learning_rate: float | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    img_height: int | None = None,
    img_width: int | None = None,
    max_images: int | None = None,
    num_workers: int | None = None,
    project: Path | None = None,
    name: str | None = None,
    english_only: bool | None = None,
) -> tuple[Path, str]:
    """Train the OCR model using Hydra configs plus CLI overrides."""

    data_cfg = cfg.data
    training_cfg = cfg.training.ocr
    model_cfg = cfg.model.ocr
    wandb_cfg = cfg.wandb

    resolved_data_dir = _normalize_path(data_dir) or _normalize_path(data_cfg.get("data_dir")) or PROJECT_ROOT / "data"
    resolved_split_dir = _normalize_path(split_dir) or _normalize_path(data_cfg.get("split_dir"))
    project_path = _normalize_path(project) or _normalize_path(training_cfg.get("project_dir")) or RUNS_DIR / "ocr"
    experiment_name = name or training_cfg.get("experiment_name", "plate_ocr")

    batch_size_value = batch_size or training_cfg.get("batch_size", 64)
    max_epochs_value = max_epochs or training_cfg.get("max_epochs", 15)
    learning_rate_value = learning_rate or training_cfg.get("learning_rate", 1e-3)
    num_workers_value = num_workers or data_cfg.get("num_workers", 4)

    english_only_value = english_only if english_only is not None else model_cfg.get("english_only", True)
    img_height_value = img_height or model_cfg.get("img_height", 32)
    img_width_value = img_width or model_cfg.get("img_width", 200)
    hidden_size_value = hidden_size or model_cfg.get("hidden_size", 256)
    num_layers_value = num_layers or model_cfg.get("num_layers", 2)
    dropout_value = model_cfg.get("dropout", 0.3)

    max_images_value = max_images or data_cfg.get("max_total_images")
    if max_images_value is None:
        raise ValueError("Specify max_images via config or CLI overrides.")

    train_split = float(data_cfg.get("train_split", 0.8))
    max_train_images = int(max_images_value * train_split)
    max_val_images = max_images_value - max_train_images

    print(f"Max images: {max_images_value} -> {max_train_images} train, {max_val_images} val")
    print(f"OCR Mode: {'English-only (6 chars)' if english_only_value else 'Full (7 chars with Chinese)'}")

    project_path.mkdir(parents=True, exist_ok=True)
    output_dir = project_path / experiment_name / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {project_path / experiment_name}")
    print(f"Prediction visualizations will be saved to: {output_dir}")
    print("Setting up data module...")

    data_module = CCPDDataModule(
        data_dir=resolved_data_dir,
        split_dir=resolved_split_dir,
        task="ocr",
        batch_size=batch_size_value,
        num_workers=num_workers_value,
        img_height=img_height_value,
        img_width=img_width_value,
        max_train_images=max_train_images,
        max_val_images=max_val_images,
        english_only=english_only_value,
    )

    print("Creating CRNN model...")
    model = PlateOCR(
        img_height=img_height_value,
        img_width=img_width_value,
        hidden_size=hidden_size_value,
        num_layers=num_layers_value,
        dropout=dropout_value,
        learning_rate=learning_rate_value,
        max_epochs=max_epochs_value,
        output_dir=str(output_dir),
        english_only=english_only_value,
    )

    checkpoint_dir = project_path / experiment_name / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="ocr-{epoch:02d}-{val_accuracy:.4f}",
            monitor="val_accuracy",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=training_cfg.get("patience", 5),
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    csv_logger = CSVLogger(save_dir=str(project_path), name=experiment_name)

    loggers: list[Any] = [csv_logger]
    wandb_logger = None
    if wandb_cfg.get("enabled", False):
        wandb_logger = WandbLogger(
            entity=wandb_cfg.get("entity", WANDB_ENTITY),
            project=wandb_cfg.get("project", WANDB_PROJECT),
            name=f"ocr-{experiment_name}",
            save_dir=str(project_path),
            config={
                "architecture": "CRNN",
                "task": "OCR",
                "english_only": english_only_value,
                "num_classes": ENGLISH_NUM_CLASSES if english_only_value else NUM_CLASSES,
                "img_height": img_height_value,
                "img_width": img_width_value,
                "hidden_size": hidden_size_value,
                "num_layers": num_layers_value,
                "learning_rate": learning_rate_value,
                "batch_size": batch_size_value,
                "max_epochs": max_epochs_value,
                "max_train_images": max_train_images,
                "max_val_images": max_val_images,
                "dataset": data_cfg.get("name", "CCPD"),
            },
        )
        loggers.append(wandb_logger)
        print(
            f"📊 Logging to Weights & Biases: {wandb_cfg.get('entity', WANDB_ENTITY)}/"
            f"{wandb_cfg.get('project', WANDB_PROJECT)}"
        )

    force_cpu = torch.backends.mps.is_available()
    accelerator, devices = get_accelerator(force_cpu=force_cpu)

    if force_cpu:
        print("Note: Using CPU for OCR training (CTC loss not supported on MPS)")

    precision_cfg = training_cfg.get("precision", "auto")
    precision_value = "16-mixed" if precision_cfg == "auto" and accelerator == "gpu" else precision_cfg
    if precision_value == "auto":
        precision_value = 32

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs_value,
        callbacks=callbacks,
        logger=loggers,
        precision=precision_value,
        gradient_clip_val=training_cfg.get("gradient_clip_val", 5.0),
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)

    if wandb_logger is not None:
        wandb.finish()

    print(f"\nTraining complete! Results saved to {project_path / experiment_name}")
    print(f"  - Metrics CSV: {project_path / experiment_name}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    if wandb_logger is not None:
        entity = wandb_cfg.get("entity", WANDB_ENTITY)
        project = wandb_cfg.get("project", WANDB_PROJECT)
        print(f"  - W&B Dashboard: https://wandb.ai/{entity}/{project}")

    return project_path, experiment_name


@app.command()
def train_detector(
    data_dir: Path | None = typer.Argument(None, help="Path to CCPD dataset directory"),
    output_dir: Path | None = typer.Option(None, help="Output directory for YOLO format data"),
    split_dir: Path | None = typer.Option(None, help="Directory containing train/val/test split files"),
    model_name: str | None = typer.Option(None, help="YOLOv8 model variant"),
    epochs: int | None = typer.Option(None, help="Number of training epochs"),
    batch_size: int | None = typer.Option(None, help="Batch size"),
    img_size: int | None = typer.Option(None, help="Input image size (smaller = faster)"),
    max_train_images: int | None = typer.Option(None, help="Maximum training images"),
    max_val_images: int | None = typer.Option(None, help="Maximum validation images"),
    project: Path | None = typer.Option(None, help="Project directory for saving results"),
    name: str | None = typer.Option(None, help="Experiment name"),
    config_name: str = typer.Option("config", "--config-name", help="Hydra config name to load."),
    config_path: Path = typer.Option(CONFIGS_DIR, "--config-path", help="Directory that stores Hydra configs."),
    overrides: list[str] | None = typer.Option(
        None,
        "--override",
        "-o",
        help="Hydra override strings (key=value). Repeat the flag for multiple overrides.",
    ),
) -> None:
    """Train the license plate detector using YOLOv8 and Hydra configs."""

    cfg = load_hydra_config(config_path=config_path, config_name=config_name, overrides=overrides)
    _train_detector_with_cfg(
        cfg,
        data_dir=data_dir,
        split_dir=split_dir,
        output_dir=output_dir,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        max_train_images=max_train_images,
        max_val_images=max_val_images,
        project=project,
        name=name,
    )


@app.command()
def train_ocr(
    data_dir: Path | None = typer.Argument(None, help="Path to CCPD dataset directory"),
    split_dir: Path | None = typer.Option(None, help="Directory containing train/val/test split files"),
    batch_size: int | None = typer.Option(None, help="Batch size"),
    max_epochs: int | None = typer.Option(None, help="Maximum number of epochs"),
    learning_rate: float | None = typer.Option(None, help="Learning rate"),
    hidden_size: int | None = typer.Option(None, help="LSTM hidden size"),
    num_layers: int | None = typer.Option(None, help="Number of LSTM layers"),
    img_height: int | None = typer.Option(None, help="Input image height (32 required for model architecture)"),
    img_width: int | None = typer.Option(None, help="Input image width (larger = better quality, more compute)"),
    max_images: int | None = typer.Option(None, help="Maximum total images (train/val split controlled via config)"),
    num_workers: int | None = typer.Option(None, help="Number of dataloader workers"),
    project: Path | None = typer.Option(None, help="Project directory for saving results"),
    name: str | None = typer.Option(None, help="Experiment name"),
    english_only: bool | None = typer.Option(None, help="Only predict English chars (no Chinese province)"),
    config_name: str = typer.Option("config", "--config-name", help="Hydra config name to load."),
    config_path: Path = typer.Option(CONFIGS_DIR, "--config-path", help="Directory that stores Hydra configs."),
    overrides: list[str] | None = typer.Option(
        None,
        "--override",
        "-o",
        help="Hydra override strings (key=value). Repeat the flag for multiple overrides.",
    ),
) -> None:
    """Train the license plate OCR model using PyTorch Lightning and Hydra configs."""

    cfg = load_hydra_config(config_path=config_path, config_name=config_name, overrides=overrides)
    _train_ocr_with_cfg(
        cfg,
        data_dir=data_dir,
        split_dir=split_dir,
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        num_layers=num_layers,
        img_height=img_height,
        img_width=img_width,
        max_images=max_images,
        num_workers=num_workers,
        project=project,
        name=name,
        english_only=english_only,
    )


@app.command()
def train_both(
    data_dir: Path | None = typer.Argument(None, help="Path to CCPD dataset directory"),
    split_dir: Path | None = typer.Option(None, help="Directory containing train/val/test split files"),
    output_dir: Path | None = typer.Option(None, help="Output directory for YOLO format data"),
    detector_epochs: int | None = typer.Option(None, help="Detector training epochs"),
    detector_batch_size: int | None = typer.Option(None, help="Detector batch size"),
    ocr_epochs: int | None = typer.Option(None, help="OCR training epochs"),
    ocr_batch_size: int | None = typer.Option(None, help="OCR training batch size"),
    max_images: int | None = typer.Option(None, help="Maximum total images for OCR"),
    english_only: bool | None = typer.Option(None, help="Only predict English chars (no Chinese province)"),
    config_name: str = typer.Option("config", "--config-name", help="Hydra config name to load."),
    config_path: Path = typer.Option(CONFIGS_DIR, "--config-path", help="Directory that stores Hydra configs."),
    overrides: list[str] | None = typer.Option(
        None,
        "--override",
        "-o",
        help="Hydra override strings (key=value). Repeat the flag for multiple overrides.",
    ),
) -> None:
    """Train both detector and OCR models sequentially (~30 min total).

    Example:
        uv run python -m ml_ops.train train-both data/ccpd_small --max-images 5000
    """
    cfg = load_hydra_config(config_path=config_path, config_name=config_name, overrides=overrides)

    print(f"Results will be saved to: {RUNS_DIR}")
    print("=" * 50)
    print("PHASE 1: Training License Plate Detector")
    print("=" * 50)

    detector_project, detector_name = _train_detector_with_cfg(
        cfg,
        data_dir=data_dir,
        split_dir=split_dir,
        output_dir=output_dir,
        epochs=detector_epochs,
        batch_size=detector_batch_size,
        project=None,
        name=None,
    )

    print("\n" + "=" * 50)
    print("PHASE 2: Training License Plate OCR")
    print("=" * 50)

    ocr_project, ocr_name = _train_ocr_with_cfg(
        cfg,
        data_dir=data_dir,
        split_dir=split_dir,
        batch_size=ocr_batch_size,
        max_epochs=ocr_epochs,
        max_images=max_images,
        english_only=english_only,
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"\nAll results saved to: {RUNS_DIR}")
    print(f"  - Detector results: {detector_project / detector_name}")
    print(f"  - OCR results: {ocr_project / ocr_name}")


if __name__ == "__main__":
    app()
