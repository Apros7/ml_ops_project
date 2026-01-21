"""
Invoke tasks for the ML Ops License Plate Recognition project.

Usage:
    uv run invoke --list              # List all available tasks
    uv run invoke <task-name>         # Run a specific task
    uv run invoke <task> --help       # Get help for a specific task

Available tasks:
    Training:
        uv run invoke train-detector          # Train YOLO license plate detector
        uv run invoke train-ocr               # Fine-tune EasyOCR recognizer (recommended)
        uv run invoke train-crnn              # Train in-house CRNN OCR model (for comparison)
        uv run invoke train-both              # Train both detector and OCR

    Data:
        uv run invoke make-small-dataset      # Create small dataset for testing

    Evaluation:
        uv run invoke eval-easyocr            # Evaluate EasyOCR on dataset

    Development:
        uv run invoke test                    # Run tests with coverage
        uv run invoke lint                    # Run linter (ruff)
        uv run invoke format                  # Format code (ruff)
        uv run invoke pre-commit              # Run pre-commit hooks on all files

    Documentation:
        uv run invoke build-docs              # Build documentation
        uv run invoke serve-docs              # Serve documentation locally

    Docker:
        uv run invoke docker-build            # Build docker images
        uv run invoke docker-push             # Push docker images to registry

    Cloud:
        uv run invoke cloud-build             # Trigger a Cloud Build
        uv run invoke api-build                # Build API docker image
        uv run invoke api-tag                  # Tag API image for registry
        uv run invoke api-push                 # Push API image to registry
        uv run invoke api-deploy               # Deploy API image to Cloud Run
        uv run invoke api-release              # Build, tag, push, deploy API image
        uv run invoke train-deploy             # Deploy train image to Cloud Run
        uv run invoke train-release            # Build, tag, push, deploy train image

    Vertex:
        uv run invoke vertex-job              # Submit a Vertex AI custom training job. ie train in the cloud

    API:
        uv run invoke api                     # Run the FastAPI service
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
def api(ctx: Context, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI service for license plate recognition."""

    cmd = f"uv run uvicorn ml_ops.api:app --host {host} --port {port}"
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
    """Fine-tune the EasyOCR recognizer model.

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
def train_crnn(
    ctx: Context,
    data_dir: str = DEFAULT_DATA_DIR,
    max_images: int = 5000,
    epochs: int = 15,
    batch_size: int = 64,
    english_only: bool = True,
) -> None:
    """Train the in-house CRNN OCR model (Lightning).

    Args:
        data_dir: Path to CCPD dataset directory.
        max_images: Maximum total images (80% train, 20% val).
        epochs: Maximum number of epochs.
        batch_size: Batch size.
        english_only: Only predict English characters (recommended).
    """
    english_flag = "--english-only" if english_only else "--no-english-only"
    cmd = (
        f"uv run python -m {PROJECT_NAME}.train train-crnn {data_dir} "
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


@task(name="pre-commit")
def pre_commit_task(ctx: Context) -> None:
    """Run pre-commit hooks on all files."""

    ctx.run("uv run pre-commit run --all-files", echo=True, pty=not WINDOWS)


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
        f"docker buildx build --platform linux/amd64 -t train:latest . -f dockerfiles/train.dockerfile --progress={progress} --load",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker buildx build --platform linux/amd64 -t api:latest . -f dockerfiles/api.dockerfile --progress={progress} --load",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker buildx build --platform linux/amd64 -t eval:latest . -f dockerfiles/eval.dockerfile --progress={progress} --load",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_push(ctx: Context) -> None:
    """Push docker images to registry."""
    ctx.run(
        "docker tag train:latest europe-west1-docker.pkg.dev/mlops-license-plate-484109/license-plate-repo/train:latest",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        "docker push europe-west1-docker.pkg.dev/mlops-license-plate-484109/license-plate-repo/train:latest",
        echo=True,
        pty=not WINDOWS,
    )


@task
def api_build(ctx: Context) -> None:
    """Build the API docker image for Cloud Run (linux/amd64)."""
    ctx.run(
        "docker buildx build --platform linux/amd64 -t api:latest -f dockerfiles/api.dockerfile . --load",
        echo=True,
        pty=not WINDOWS,
    )


@task
def api_tag(
    ctx: Context,
    project: str = "mlops-license-plate-484109",
    repo: str = "license-plate-repo",
    tag: str = "latest",
    region: str = "europe-west1",
) -> None:
    """Tag the API image for Artifact Registry.

    Args:
        project: GCP project ID.
        repo: Artifact Registry repository name.
        tag: Docker image tag.
        region: Artifact Registry region.
    """
    image = f"{region}-docker.pkg.dev/{project}/{repo}/api:{tag}"
    ctx.run(f"docker tag api:latest {image}", echo=True, pty=not WINDOWS)


@task
def api_push(
    ctx: Context,
    project: str = "mlops-license-plate-484109",
    repo: str = "license-plate-repo",
    tag: str = "latest",
    region: str = "europe-west1",
) -> None:
    """Push the API image to Artifact Registry.

    Args:
        project: GCP project ID.
        repo: Artifact Registry repository name.
        tag: Docker image tag.
        region: Artifact Registry region.
    """
    image = f"{region}-docker.pkg.dev/{project}/{repo}/api:{tag}"
    ctx.run(f"docker push {image}", echo=True, pty=not WINDOWS)


@task
def api_deploy(
    ctx: Context,
    project: str = "mlops-license-plate-484109",
    repo: str = "license-plate-repo",
    tag: str = "latest",
    region: str = "europe-west1",
    service: str = "api",
) -> None:
    """Deploy the API image to Cloud Run.

    Args:
        project: GCP project ID.
        repo: Artifact Registry repository name.
        tag: Docker image tag.
        region: Cloud Run region.
        service: Cloud Run service name.
    """
    image = f"{region}-docker.pkg.dev/{project}/{repo}/api:{tag}"
    cmd = f"gcloud run deploy {service} " f"--region={region} " f"--image={image} " "--allow-unauthenticated"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def api_release(
    ctx: Context,
    project: str = "mlops-license-plate-484109",
    repo: str = "license-plate-repo",
    tag: str = "latest",
    region: str = "europe-west1",
    service: str = "api",
) -> None:
    """Build, tag, push, and deploy the API image.

    Args:
        project: GCP project ID.
        repo: Artifact Registry repository name.
        tag: Docker image tag.
        region: Cloud Run/Artifact Registry region.
        service: Cloud Run service name.
    """
    api_build(ctx)
    api_tag(ctx, project=project, repo=repo, tag=tag, region=region)
    api_push(ctx, project=project, repo=repo, tag=tag, region=region)
    api_deploy(ctx, project=project, repo=repo, tag=tag, region=region, service=service)


@task
def train_deploy(
    ctx: Context,
    project: str = "mlops-license-plate-484109",
    repo: str = "license-plate-repo",
    tag: str = "latest",
    region: str = "europe-west1",
    service: str = "train",
) -> None:
    """Deploy the training image to Cloud Run.

    Args:
        project: GCP project ID.
        repo: Artifact Registry repository name.
        tag: Docker image tag.
        region: Cloud Run region.
        service: Cloud Run service name.
    """
    image = f"{region}-docker.pkg.dev/{project}/{repo}/train:{tag}"
    cmd = f"gcloud run deploy {service} " f"--region={region} " f"--image={image} " "--allow-unauthenticated"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def train_release(
    ctx: Context,
    project: str = "mlops-license-plate-484109",
    repo: str = "license-plate-repo",
    tag: str = "latest",
    region: str = "europe-west1",
    service: str = "train",
) -> None:
    """Build, tag, push, and deploy the training image.

    Args:
        project: GCP project ID.
        repo: Artifact Registry repository name.
        tag: Docker image tag.
        region: Cloud Run/Artifact Registry region.
        service: Cloud Run service name.
    """
    ctx.run(
        "docker buildx build --platform linux/amd64 -t train:latest -f dockerfiles/train.dockerfile . --load",
        echo=True,
        pty=not WINDOWS,
    )
    image = f"{region}-docker.pkg.dev/{project}/{repo}/train:{tag}"
    ctx.run(f"docker tag train:latest {image}", echo=True, pty=not WINDOWS)
    ctx.run(f"docker push {image}", echo=True, pty=not WINDOWS)
    cmd = f"gcloud run deploy {service} " f"--region={region} " f"--image={image} " "--allow-unauthenticated"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def vertex_job(
    ctx: Context,
    config_path: str = "configs/vertex_train.yaml",
) -> None:
    """Submit a Vertex AI training job.

    Args:
        config_path: Path to the Vertex job spec YAML.
    """
    cmd = f"gcloud builds submit . --config={config_path}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def cloud_build(ctx: Context, config_path: str = "cloudbuild.yaml") -> None:
    """Trigger a Cloud Build using a config file.

    Args:
        config_path: Path to the Cloud Build config YAML.
    """
    ctx.run(f"gcloud builds submit . --config={config_path}", echo=True, pty=not WINDOWS)


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


# create cloud run:

# gcloud run jobs create train-job2 --region=europe-west1 --image=europe-west1-docker.pkg.dev/mlops-license-plate-484109/license-plate-repo/train:latest --command=uv --args=run,-m,ml_ops.train,train-both,data/ccpd_small,--max-images,50
# gcloud run jobs execute train-job2 --region=europe-west1
