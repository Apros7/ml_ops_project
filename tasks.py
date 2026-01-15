"""
Invoke tasks for the ML Ops License Plate Recognition project.

Usage:
    uv run invoke --list              # List all available tasks
    uv run invoke <task-name>         # Run a specific task
    uv run invoke <task> --help       # Get help for a specific task

Available tasks:
    Training:
        uv run invoke train-detector          # Train YOLO license plate detector
        uv run invoke train-ocr               # Train CRNN OCR model (English-only)
        uv run invoke train-both              # Train both detector and OCR

    Data:
        uv run invoke make-small-dataset      # Create small dataset for testing

    Evaluation:
        uv run invoke eval-easyocr            # Evaluate EasyOCR on dataset

    Development:
        uv run invoke test                    # Run tests with coverage
        uv run invoke lint                    # Run linter (ruff)
        uv run invoke format                  # Format code (ruff)

    Documentation:
        uv run invoke build-docs              # Build documentation
        uv run invoke serve-docs              # Serve documentation locally

    Docker:
        uv run invoke docker-build            # Build docker images
"""

import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "ml_ops"
PYTHON_VERSION = "3.12"

# Default data directory
DEFAULT_DATA_DIR = "data/base"
# DEFAULT_DATA_DIR = "data/ccpd_tiny"



# ============================================================================
# Training tasks
# ============================================================================


@task
def train_detector(
    ctx: Context,
    data_dir: str = DEFAULT_DATA_DIR,
    max_train: int = 5000,
    max_val: int = 1000,
    epochs: int = 10,
    batch_size: int = 32,
) -> None:
    """Train the YOLO license plate detector.

    Args:
        data_dir: Path to CCPD dataset directory.
        max_train: Maximum training images.
        max_val: Maximum validation images.
        epochs: Number of training epochs.
        batch_size: Batch size.
    """
    cmd = (
        f"uv run python -m {PROJECT_NAME}.train train-detector {data_dir} "
        f"--max-train-images {max_train} --max-val-images {max_val} "
        f"--epochs {epochs} --batch-size {batch_size}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def train_ocr(
    ctx: Context,
    data_dir: str = DEFAULT_DATA_DIR,
    max_images: int = 5000,
    epochs: int = 15,
    batch_size: int = 64,
    english_only: bool = True,
) -> None:
    """Train the CRNN OCR model.

    Args:
        data_dir: Path to CCPD dataset directory.
        max_images: Maximum total images (80% train, 20% val).
        epochs: Maximum number of epochs.
        batch_size: Batch size.
        english_only: Only predict English characters (recommended).
    """
    english_flag = "--english-only" if english_only else "--no-english-only"
    cmd = (
        f"uv run python -m {PROJECT_NAME}.train train-ocr {data_dir} "
        f"--max-images {max_images} --max-epochs {epochs} "
        f"--batch-size {batch_size} {english_flag}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def train_both(
    ctx: Context,
    data_dir: str = DEFAULT_DATA_DIR,
    max_images: int = 5000,
    detector_epochs: int = 10,
    ocr_epochs: int = 15,
) -> None:
    """Train both detector and OCR models (~30 min total).

    Args:
        data_dir: Path to CCPD dataset directory.
        max_images: Maximum total images.
        detector_epochs: Detector training epochs.
        ocr_epochs: OCR training epochs.
    """
    cmd = (
        f"uv run python -m {PROJECT_NAME}.train train-both {data_dir} "
        f"--max-images {max_images} --detector-epochs {detector_epochs} "
        f"--ocr-epochs {ocr_epochs}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


# ============================================================================
# Data tasks
# ============================================================================


@task
def make_small_dataset(
    ctx: Context,
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = "data/ccpd_small",
    train_size: int = 4000,
    val_size: int = 1000,
) -> None:
    """Create a small subset of the CCPD dataset for testing.

    Args:
        data_dir: Path to original CCPD dataset.
        output_dir: Output directory for small dataset.
        train_size: Number of training images.
        val_size: Number of validation images.
    """
    cmd = (
        f"uv run python -m {PROJECT_NAME}.make_small_dataset {data_dir} "
        f"--output-dir {output_dir} --train-size {train_size} --val-size {val_size}"
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


# ============================================================================
# Evaluation tasks
# ============================================================================


@task
def eval_easyocr(
    ctx: Context,
    data_dir: str = DEFAULT_DATA_DIR,
    max_images: int = 100,
) -> None:
    """Evaluate EasyOCR on CCPD images for comparison.

    Args:
        data_dir: Path to CCPD dataset directory.
        max_images: Maximum images to evaluate.
    """
    cmd = f"uv run python -m {PROJECT_NAME}.eval_easyocr evaluate {data_dir} --max-images {max_images}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


# ============================================================================
# Development tasks
# ============================================================================


@task
def test(ctx: Context) -> None:
    """Run tests with coverage."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def lint(ctx: Context, fix: bool = True) -> None:
    """Run ruff linter.

    Args:
        fix: Auto-fix issues where possible.
    """
    fix_flag = "--fix" if fix else ""
    ctx.run(f"uv run ruff check . {fix_flag}", echo=True, pty=not WINDOWS)


@task
def format(ctx: Context) -> None:
    """Format code with ruff."""
    ctx.run("uv run ruff format .", echo=True, pty=not WINDOWS)


@task
def lint_and_format(ctx: Context) -> None:
    """Run both linter and formatter."""
    lint(ctx)
    format(ctx)


# ============================================================================
# Docker tasks
# ============================================================================


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images.

    Args:
        progress: Docker build progress output type.
    """
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t eval:latest . -f dockerfiles/eval.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


# ============================================================================
# Documentation tasks
# ============================================================================


@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation locally."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
