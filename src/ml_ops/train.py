"""Training scripts for license plate detection and OCR."""

from pathlib import Path

import typer
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from ml_ops.model import PlateDetector, PlateOCR
from ml_ops.data import CCPDDataModule, export_yolo_format

app = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"


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


@app.command()
def train_detector(
    data_dir: Path = typer.Argument(..., help="Path to CCPD dataset directory"),
    output_dir: Path = typer.Option(Path("data/processed/yolo"), help="Output directory for YOLO format data"),
    split_dir: Path = typer.Option(None, help="Directory containing train/val/test split files"),
    model_name: str = typer.Option("yolov8n.pt", help="YOLOv8 model variant"),
    epochs: int = typer.Option(10, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    img_size: int = typer.Option(320, help="Input image size (smaller = faster)"),
    max_train_images: int = typer.Option(5000, help="Maximum training images"),
    max_val_images: int = typer.Option(1000, help="Maximum validation images"),
    project: str = typer.Option(None, help="Project directory for saving results"),
    name: str = typer.Option("plate_detection", help="Experiment name"),
) -> None:
    """Train the license plate detector using YOLOv8.

    Optimized defaults for ~15 min training:
    - 5000 train images, 1000 val images
    - 10 epochs
    - 320px image size
    - batch size 32

    Supports two data layouts:
    1. Flat: all images in data_dir (will sample max_train_images + max_val_images)
    2. Split: data_dir/train/ and data_dir/val/ subfolders (uses as-is)
    """
    project_path = Path(project) if project else RUNS_DIR / "detect"
    project_path.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {project_path / name}")
    print("Step 1: Converting CCPD to YOLO format...")

    train_output = output_dir / "train"
    val_output = output_dir / "val"

    train_data_dir = data_dir / "train" if (data_dir / "train").exists() else data_dir
    val_data_dir = data_dir / "val" if (data_dir / "val").exists() else data_dir

    if train_data_dir != data_dir:
        print(f"Detected pre-split dataset: {data_dir}")
        print(f"  Train folder: {train_data_dir}")
        print(f"  Val folder: {val_data_dir}")

    train_split = split_dir / "train.txt" if split_dir else None
    val_split = split_dir / "val.txt" if split_dir else None

    export_yolo_format(train_data_dir, train_output, train_split, max_images=max_train_images)
    export_yolo_format(val_data_dir, val_output, val_split, max_images=max_val_images)

    print("Step 2: Creating data.yaml config...")
    data_yaml_path = output_dir / "data.yaml"
    data_yaml_content = f"""
path: {output_dir.absolute()}
train: train/images
val: val/images

names:
  0: license_plate
"""
    with open(data_yaml_path, "w") as f:
        f.write(data_yaml_content.strip())

    print("Step 3: Training YOLOv8...")
    detector = PlateDetector(model_name=model_name)

    accelerator, _ = get_accelerator()
    device = 0 if accelerator == "gpu" else "cpu"

    detector.train_yolo(
        data_yaml=str(data_yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=str(project_path),
        name=name,
    )

    print(f"\nTraining complete! Results saved to {project_path / name}")
    print(f"  - Training curves: {project_path / name / 'results.png'}")
    print(f"  - Metrics CSV: {project_path / name / 'results.csv'}")
    print(f"  - Best weights: {project_path / name / 'weights' / 'best.pt'}")


@app.command()
def train_ocr(
    data_dir: Path = typer.Argument(..., help="Path to CCPD dataset directory"),
    split_dir: Path = typer.Option(None, help="Directory containing train/val/test split files"),
    batch_size: int = typer.Option(128, help="Batch size"),
    max_epochs: int = typer.Option(15, help="Maximum number of epochs"),
    learning_rate: float = typer.Option(3e-3, help="Learning rate"),
    hidden_size: int = typer.Option(128, help="LSTM hidden size"),
    num_layers: int = typer.Option(1, help="Number of LSTM layers"),
    img_height: int = typer.Option(32, help="Input image height"),
    img_width: int = typer.Option(100, help="Input image width"),
    max_train_images: int = typer.Option(5000, help="Maximum training images"),
    max_val_images: int = typer.Option(1000, help="Maximum validation images"),
    num_workers: int = typer.Option(4, help="Number of dataloader workers"),
    project: str = typer.Option(None, help="Project directory for saving results"),
    name: str = typer.Option("plate_ocr", help="Experiment name"),
) -> None:
    """Train the license plate OCR model using PyTorch Lightning.

    Optimized defaults for ~15 min training:
    - 5000 train images, 1000 val images
    - 15 epochs
    - Smaller model (hidden_size=128, 1 LSTM layer)
    - Larger batch size (128)
    """
    project_path = Path(project) if project else RUNS_DIR / "ocr"
    project_path.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {project_path / name}")
    print("Setting up data module...")
    data_module = CCPDDataModule(
        data_dir=data_dir,
        split_dir=split_dir,
        task="ocr",
        batch_size=batch_size,
        num_workers=num_workers,
        img_height=img_height,
        img_width=img_width,
        max_train_images=max_train_images,
        max_val_images=max_val_images,
    )

    print("Creating model...")
    model = PlateOCR(
        img_height=img_height,
        img_width=img_width,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
    )

    checkpoint_dir = project_path / name / "checkpoints"
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
            patience=5,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    tb_logger = TensorBoardLogger(
        save_dir=str(project_path),
        name=name,
    )
    csv_logger = CSVLogger(
        save_dir=str(project_path),
        name=name,
    )

    # CTC loss is not supported on MPS (Apple Silicon), so force CPU on Mac
    force_cpu = torch.backends.mps.is_available()
    accelerator, devices = get_accelerator(force_cpu=force_cpu)

    if force_cpu:
        print("Note: Using CPU for OCR training (CTC loss not supported on MPS)")
    print(f"Training on {accelerator}...")
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=[tb_logger, csv_logger],
        precision="16-mixed" if accelerator == "gpu" else 32,
        gradient_clip_val=5.0,
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)

    print(f"\nTraining complete! Results saved to {project_path / name}")
    print(f"  - TensorBoard logs: {project_path / name}")
    print(f"  - Metrics CSV: {project_path / name / 'metrics.csv'}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print("\nTo view training curves, run:")
    print(f"  tensorboard --logdir {project_path / name}")


@app.command()
def train_both(
    data_dir: Path = typer.Argument(..., help="Path to CCPD dataset directory"),
    split_dir: Path = typer.Option(None, help="Directory containing train/val/test split files"),
    output_dir: Path = typer.Option(Path("data/processed/yolo"), help="Output directory for YOLO format data"),
    detector_epochs: int = typer.Option(10, help="Detector training epochs"),
    ocr_epochs: int = typer.Option(15, help="OCR training epochs"),
    batch_size: int = typer.Option(32, help="Batch size for detector"),
    max_images: int = typer.Option(5000, help="Maximum total images (80%% train, 20%% val)"),
) -> None:
    """Train both detector and OCR models sequentially (~30 min total).

    Example:
        uv run python -m ml_ops.train train-both data/ccpd_small --max-images 5000
    """
    max_train_images = int(max_images * 0.8)
    max_val_images = max_images - max_train_images

    print(f"Max images: {max_images} -> {max_train_images} train, {max_val_images} val")
    print(f"Results will be saved to: {RUNS_DIR}")
    print("=" * 50)
    print("PHASE 1: Training License Plate Detector")
    print("=" * 50)

    train_detector(
        data_dir=data_dir,
        output_dir=output_dir,
        split_dir=split_dir,
        model_name="yolov8n.pt",
        epochs=detector_epochs,
        batch_size=batch_size,
        img_size=320,
        max_train_images=max_train_images,
        max_val_images=max_val_images,
        project=str(RUNS_DIR / "detect"),
        name="plate_detection",
    )

    print("\n" + "=" * 50)
    print("PHASE 2: Training License Plate OCR")
    print("=" * 50)

    train_ocr(
        data_dir=data_dir,
        split_dir=split_dir,
        batch_size=batch_size * 4,
        max_epochs=ocr_epochs,
        learning_rate=3e-3,
        hidden_size=128,
        num_layers=1,
        img_height=32,
        img_width=100,
        max_train_images=max_train_images,
        max_val_images=max_val_images,
        num_workers=4,
        project=str(RUNS_DIR / "ocr"),
        name="plate_ocr",
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"\nAll results saved to: {RUNS_DIR}")
    print(f"  - Detector results: {RUNS_DIR / 'detect' / 'plate_detection'}")
    print(f"  - OCR results: {RUNS_DIR / 'ocr' / 'plate_ocr'}")
    print("\nTo view OCR training curves:")
    print(f"  tensorboard --logdir {RUNS_DIR / 'ocr'}")


if __name__ == "__main__":
    app()
