"""Training scripts for license plate detection and OCR."""

from pathlib import Path

import typer
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from ml_ops.model import PlateDetector, PlateOCR
from ml_ops.data import CCPDDataModule, export_yolo_format

app = typer.Typer()


def get_accelerator() -> tuple[str, str]:
    """Get the appropriate accelerator and device configuration.

    Returns:
        Tuple of (accelerator, devices) for Lightning Trainer.
    """
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
    epochs: int = typer.Option(100, help="Number of training epochs"),
    batch_size: int = typer.Option(16, help="Batch size"),
    img_size: int = typer.Option(640, help="Input image size"),
    project: str = typer.Option("runs/detect", help="Project directory for saving results"),
    name: str = typer.Option("plate_detection", help="Experiment name"),
) -> None:
    """Train the license plate detector using YOLOv8.

    This function:
    1. Converts CCPD data to YOLO format
    2. Creates a data.yaml config file
    3. Trains YOLOv8 using Ultralytics native training
    """
    print("Step 1: Converting CCPD to YOLO format...")

    train_output = output_dir / "train"
    val_output = output_dir / "val"

    train_split = split_dir / "train.txt" if split_dir else None
    val_split = split_dir / "val.txt" if split_dir else None

    export_yolo_format(data_dir, train_output, train_split)
    export_yolo_format(data_dir, val_output, val_split)

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
        project=project,
        name=name,
    )

    print(f"Training complete! Results saved to {project}/{name}")


@app.command()
def train_ocr(
    data_dir: Path = typer.Argument(..., help="Path to CCPD dataset directory"),
    split_dir: Path = typer.Option(None, help="Directory containing train/val/test split files"),
    batch_size: int = typer.Option(64, help="Batch size"),
    max_epochs: int = typer.Option(100, help="Maximum number of epochs"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate"),
    hidden_size: int = typer.Option(256, help="LSTM hidden size"),
    num_layers: int = typer.Option(2, help="Number of LSTM layers"),
    img_height: int = typer.Option(48, help="Input image height"),
    img_width: int = typer.Option(168, help="Input image width"),
    num_workers: int = typer.Option(4, help="Number of dataloader workers"),
    project: str = typer.Option("runs/ocr", help="Project directory for saving results"),
    name: str = typer.Option("plate_ocr", help="Experiment name"),
) -> None:
    """Train the license plate OCR model using PyTorch Lightning.

    This trains a CRNN model with CTC loss for character recognition.
    """
    print("Setting up data module...")
    data_module = CCPDDataModule(
        data_dir=data_dir,
        split_dir=split_dir,
        task="ocr",
        batch_size=batch_size,
        num_workers=num_workers,
        img_height=img_height,
        img_width=img_width,
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

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{project}/{name}/checkpoints",
            filename="ocr-{epoch:02d}-{val_accuracy:.4f}",
            monitor="val_accuracy",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = TensorBoardLogger(
        save_dir=project,
        name=name,
    )

    accelerator, devices = get_accelerator()

    print(f"Training on {accelerator}...")
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        precision="16-mixed" if accelerator == "gpu" else 32,
        gradient_clip_val=5.0,
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)

    print(f"Training complete! Results saved to {project}/{name}")


@app.command()
def train_both(
    data_dir: Path = typer.Argument(..., help="Path to CCPD dataset directory"),
    split_dir: Path = typer.Option(None, help="Directory containing train/val/test split files"),
    output_dir: Path = typer.Option(Path("data/processed/yolo"), help="Output directory for YOLO format data"),
    detector_epochs: int = typer.Option(100, help="Detector training epochs"),
    ocr_epochs: int = typer.Option(100, help="OCR training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    project: str = typer.Option("runs", help="Project directory"),
) -> None:
    """Train both detector and OCR models sequentially."""
    print("=" * 50)
    print("PHASE 1: Training License Plate Detector")
    print("=" * 50)

    train_detector(
        data_dir=data_dir,
        output_dir=output_dir,
        split_dir=split_dir,
        epochs=detector_epochs,
        batch_size=batch_size,
        project=f"{project}/detect",
        name="plate_detection",
    )

    print("\n" + "=" * 50)
    print("PHASE 2: Training License Plate OCR")
    print("=" * 50)

    train_ocr(
        data_dir=data_dir,
        split_dir=split_dir,
        max_epochs=ocr_epochs,
        batch_size=batch_size * 2,
        project=f"{project}/ocr",
        name="plate_ocr",
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)


if __name__ == "__main__":
    app()
