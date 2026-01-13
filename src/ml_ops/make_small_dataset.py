"""Create a small subset of the CCPD dataset for faster training."""

import random
import shutil
from pathlib import Path
from loguru import logger

import typer
from tqdm import tqdm


def make_small_dataset(
    input_dir: Path = typer.Argument(..., help="Path to original CCPD dataset"),
    output_dir: Path = typer.Option(Path("data/ccpd_small"), help="Output directory for small dataset"),
    train_size: int = typer.Option(4000, help="Number of training images"),
    val_size: int = typer.Option(1000, help="Number of validation images"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Create a small subset of CCPD dataset with train/val split.

    Args:
        input_dir: Path to original CCPD images.
        output_dir: Output directory for the small dataset.
        train_size: Number of images for training.
        val_size: Number of images for validation.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning for images in {input_dir}...")
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(input_dir.rglob(ext))

    logger.info(f"Found {len(image_paths)} images")

    total_needed = train_size + val_size
    if len(image_paths) < total_needed:
        logger.warning(f"Only {len(image_paths)} images available, need {total_needed}")
        ratio = train_size / total_needed
        train_size = int(len(image_paths) * ratio)
        val_size = len(image_paths) - train_size

    random.shuffle(image_paths)
    selected = image_paths[: train_size + val_size]

    train_images = selected[:train_size]
    val_images = selected[train_size:]

    logger.info(f"\nCopying {len(train_images)} training images...")
    for img_path in tqdm(train_images, desc="Train", unit="img"):
        shutil.copy(img_path, train_dir / img_path.name)

    logger.info(f"\nCopying {len(val_images)} validation images...")
    for img_path in tqdm(val_images, desc="Val", unit="img"):
        shutil.copy(img_path, val_dir / img_path.name)

    logger.info(f"\n{'='*50}")
    logger.info("Small dataset created!")
    logger.info(f"  Train: {len(train_images)} images -> {train_dir}")
    logger.info(f"  Val:   {len(val_images)} images -> {val_dir}")
    logger.info(f"  Total: {len(train_images) + len(val_images)} images")
    logger.info(f"{'='*50}")
    logger.info("\nTo train with this dataset:")
    logger.info(f"  uv run python -m ml_ops.train train-detector {output_dir}")
    logger.info(f"  uv run python -m ml_ops.train train-ocr {output_dir}")


if __name__ == "__main__":
    typer.run(make_small_dataset)
