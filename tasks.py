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
        uv run invoke train-deploy             # Create Cloud Run Job for training
        uv run invoke train-release            # Build, tag, push, update & execute training job

    Vertex:
        uv run invoke vertex-job              # Submit a Vertex AI custom training job. ie train in the cloud

    API:
        uv run invoke api                     # Run the FastAPI service
"""

import os
import shlex
from pathlib import Path

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "ml_ops"
PYTHON_VERSION = "3.12"

# Default data directory
# NOTE: `invoke train-*` tasks pass this path as the first CLI argument, overriding Hydra defaults.
# Keep it pointed at a real dataset directory in the repo.
DEFAULT_DATA_DIR = "data/ccpd_tiny"
PROJECT_ROOT = Path(__file__).resolve().parent


def _format_overrides(overrides: tuple[str, ...] | None) -> str:
    """Format Hydra overrides for CLI forwarding."""
    if not overrides:
        return ""
    return " ".join(f"--override {shlex.quote(override)}" for override in overrides)


def _absolute_config_path(config_path: str) -> str:
    """Hydra requires an absolute `--config-path`; resolve relative paths from the project root."""
    candidate = Path(config_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


# ============================================================================
# Training tasks
# ============================================================================


@task(iterable=["override"])
def train_detector(
    ctx: Context,
    data_dir: str = DEFAULT_DATA_DIR,
    max_train: int | None = None,
    max_val: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    img_size: int | None = None,
    model_name: str | None = None,
    output_dir: str | None = None,
    split_dir: str | None = None,
    project: str | None = None,
    name: str | None = None,
    config_name: str | None = None,
    config_path: str | None = None,
    override: tuple[str, ...] | None = None,
) -> None:
    """Train the YOLO license plate detector.

    Args:
        data_dir: Path to CCPD dataset directory.
        max_train: Maximum training images.
        max_val: Maximum validation images.
        epochs: Number of training epochs.
        batch_size: Batch size.
        img_size: Input image size.
        model_name: YOLOv8 model variant (e.g. yolov8n.pt).
        output_dir: Output directory for YOLO-format data.
        split_dir: Directory containing train/val split files.
        project: Output directory for Ultralytics training runs.
        name: Experiment name.
        config_name: Hydra config name (defaults to configs/config.yaml).
        config_path: Hydra config directory.
        override: Hydra override strings (repeatable).
    """
    # Preserve the historical defaults when running without Hydra overrides.
    # If overrides are provided, prefer config-driven values and only pass explicit CLI flags if set.
    use_config_values = bool(override)
    if not use_config_values:
        max_train = 5000 if max_train is None else max_train
        max_val = 1000 if max_val is None else max_val
        epochs = 10 if epochs is None else epochs
        batch_size = 32 if batch_size is None else batch_size

    cmd_parts = [
        "uv",
        "run",
        "python",
        "-m",
        f"{PROJECT_NAME}.train",
        "train-detector",
        shlex.quote(data_dir),
    ]
    if config_name is not None:
        cmd_parts.extend(["--config-name", shlex.quote(config_name)])
    if config_path is not None:
        cmd_parts.extend(["--config-path", shlex.quote(_absolute_config_path(config_path))])
    if output_dir is not None:
        cmd_parts.extend(["--output-dir", shlex.quote(output_dir)])
    if split_dir is not None:
        cmd_parts.extend(["--split-dir", shlex.quote(split_dir)])
    if model_name is not None:
        cmd_parts.extend(["--model-name", shlex.quote(model_name)])
    if epochs is not None:
        cmd_parts.extend(["--epochs", str(epochs)])
    if batch_size is not None:
        cmd_parts.extend(["--batch-size", str(batch_size)])
    if img_size is not None:
        cmd_parts.extend(["--img-size", str(img_size)])
    if max_train is not None:
        cmd_parts.extend(["--max-train-images", str(max_train)])
    if max_val is not None:
        cmd_parts.extend(["--max-val-images", str(max_val)])
    if project is not None:
        cmd_parts.extend(["--project", shlex.quote(project)])
    if name is not None:
        cmd_parts.extend(["--name", shlex.quote(name)])

    overrides_str = _format_overrides(override)
    cmd = " ".join(cmd_parts) + (f" {overrides_str}" if overrides_str else "")
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def api(ctx: Context, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI service for license plate recognition."""

    cmd = f"uv run uvicorn ml_ops.api:app --host {host} --port {port}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task(iterable=["override"])
def train_ocr(
    ctx: Context,
    data_dir: str = DEFAULT_DATA_DIR,
    max_images: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    english_only: bool | None = None,
    split_dir: str | None = None,
    project: str | None = None,
    name: str | None = None,
    config_name: str | None = None,
    config_path: str | None = None,
    override: tuple[str, ...] | None = None,
) -> None:
    """Fine-tune the EasyOCR recognizer model.

    Args:
        data_dir: Path to CCPD dataset directory.
        max_images: Maximum total images (80% train, 20% val).
        epochs: Maximum number of epochs.
        batch_size: Batch size.
        english_only: Only predict English characters (recommended).
        split_dir: Directory containing train/val split files.
        project: Output directory for logs/checkpoints.
        name: Experiment name.
        config_name: Hydra config name (defaults to configs/config.yaml).
        config_path: Hydra config directory.
        override: Hydra override strings (repeatable).
    """
    use_config_values = bool(override)
    if not use_config_values:
        max_images = 5000 if max_images is None else max_images
        epochs = 15 if epochs is None else epochs
        batch_size = 64 if batch_size is None else batch_size
        english_only = True if english_only is None else english_only

    cmd_parts = [
        "uv",
        "run",
        "python",
        "-m",
        f"{PROJECT_NAME}.train",
        "train-ocr",
        shlex.quote(data_dir),
    ]
    if config_name is not None:
        cmd_parts.extend(["--config-name", shlex.quote(config_name)])
    if config_path is not None:
        cmd_parts.extend(["--config-path", shlex.quote(_absolute_config_path(config_path))])
    if split_dir is not None:
        cmd_parts.extend(["--split-dir", shlex.quote(split_dir)])
    if max_images is not None:
        cmd_parts.extend(["--max-images", str(max_images)])
    if epochs is not None:
        cmd_parts.extend(["--max-epochs", str(epochs)])
    if batch_size is not None:
        cmd_parts.extend(["--batch-size", str(batch_size)])
    if english_only is True:
        cmd_parts.append("--english-only")
    elif english_only is False:
        cmd_parts.append("--no-english-only")
    if project is not None:
        cmd_parts.extend(["--project", shlex.quote(project)])
    if name is not None:
        cmd_parts.extend(["--name", shlex.quote(name)])

    overrides_str = _format_overrides(override)
    cmd = " ".join(cmd_parts) + (f" {overrides_str}" if overrides_str else "")
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task(iterable=["override"])
def train_crnn(
    ctx: Context,
    data_dir: str = DEFAULT_DATA_DIR,
    max_images: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    english_only: bool | None = None,
    split_dir: str | None = None,
    project: str | None = None,
    name: str | None = None,
    config_name: str | None = None,
    config_path: str | None = None,
    override: tuple[str, ...] | None = None,
) -> None:
    """Train the in-house CRNN OCR model (Lightning).

    Args:
        data_dir: Path to CCPD dataset directory.
        max_images: Maximum total images (80% train, 20% val).
        epochs: Maximum number of epochs.
        batch_size: Batch size.
        english_only: Only predict English characters (recommended).
        split_dir: Directory containing train/val split files.
        project: Output directory for logs/checkpoints.
        name: Experiment name.
        config_name: Hydra config name (defaults to configs/config.yaml).
        config_path: Hydra config directory.
        override: Hydra override strings (repeatable).
    """
    use_config_values = bool(override)
    if not use_config_values:
        max_images = 5000 if max_images is None else max_images
        epochs = 15 if epochs is None else epochs
        batch_size = 64 if batch_size is None else batch_size
        english_only = True if english_only is None else english_only

    cmd_parts = [
        "uv",
        "run",
        "python",
        "-m",
        f"{PROJECT_NAME}.train",
        "train-crnn",
        shlex.quote(data_dir),
    ]
    if config_name is not None:
        cmd_parts.extend(["--config-name", shlex.quote(config_name)])
    if config_path is not None:
        cmd_parts.extend(["--config-path", shlex.quote(_absolute_config_path(config_path))])
    if split_dir is not None:
        cmd_parts.extend(["--split-dir", shlex.quote(split_dir)])
    if max_images is not None:
        cmd_parts.extend(["--max-images", str(max_images)])
    if epochs is not None:
        cmd_parts.extend(["--max-epochs", str(epochs)])
    if batch_size is not None:
        cmd_parts.extend(["--batch-size", str(batch_size)])
    if english_only is True:
        cmd_parts.append("--english-only")
    elif english_only is False:
        cmd_parts.append("--no-english-only")
    if project is not None:
        cmd_parts.extend(["--project", shlex.quote(project)])
    if name is not None:
        cmd_parts.extend(["--name", shlex.quote(name)])

    overrides_str = _format_overrides(override)
    cmd = " ".join(cmd_parts) + (f" {overrides_str}" if overrides_str else "")
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task(iterable=["override"])
def train_both(
    ctx: Context,
    data_dir: str = DEFAULT_DATA_DIR,
    max_images: int | None = None,
    detector_epochs: int | None = None,
    ocr_epochs: int | None = None,
    detector_batch_size: int | None = None,
    ocr_batch_size: int | None = None,
    english_only: bool | None = None,
    split_dir: str | None = None,
    output_dir: str | None = None,
    config_name: str | None = None,
    config_path: str | None = None,
    override: tuple[str, ...] | None = None,
) -> None:
    """Train both detector and OCR models (~30 min total).

    Args:
        data_dir: Path to CCPD dataset directory.
        max_images: Maximum total images.
        detector_epochs: Detector training epochs.
        ocr_epochs: OCR training epochs.
        detector_batch_size: Detector batch size.
        ocr_batch_size: OCR batch size.
        english_only: Only predict English chars (no Chinese province).
        split_dir: Directory containing train/val split files.
        output_dir: Output directory for YOLO-format data.
        config_name: Hydra config name (defaults to configs/config.yaml).
        config_path: Hydra config directory.
        override: Hydra override strings (repeatable).
    """
    use_config_values = bool(override)
    if not use_config_values:
        max_images = 5000 if max_images is None else max_images
        detector_epochs = 10 if detector_epochs is None else detector_epochs
        ocr_epochs = 15 if ocr_epochs is None else ocr_epochs

    cmd_parts = [
        "uv",
        "run",
        "python",
        "-m",
        f"{PROJECT_NAME}.train",
        "train-both",
        shlex.quote(data_dir),
    ]
    if config_name is not None:
        cmd_parts.extend(["--config-name", shlex.quote(config_name)])
    if config_path is not None:
        cmd_parts.extend(["--config-path", shlex.quote(_absolute_config_path(config_path))])
    if split_dir is not None:
        cmd_parts.extend(["--split-dir", shlex.quote(split_dir)])
    if output_dir is not None:
        cmd_parts.extend(["--output-dir", shlex.quote(output_dir)])
    if max_images is not None:
        cmd_parts.extend(["--max-images", str(max_images)])
    if detector_epochs is not None:
        cmd_parts.extend(["--detector-epochs", str(detector_epochs)])
    if detector_batch_size is not None:
        cmd_parts.extend(["--detector-batch-size", str(detector_batch_size)])
    if ocr_epochs is not None:
        cmd_parts.extend(["--ocr-epochs", str(ocr_epochs)])
    if ocr_batch_size is not None:
        cmd_parts.extend(["--ocr-batch-size", str(ocr_batch_size)])
    if english_only is True:
        cmd_parts.append("--english-only")
    elif english_only is False:
        cmd_parts.append("--no-english-only")

    overrides_str = _format_overrides(override)
    cmd = " ".join(cmd_parts) + (f" {overrides_str}" if overrides_str else "")
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
    job: str = "train-job",
    data_dir: str = DEFAULT_DATA_DIR,
    detector_batch_size: int = 8,
    ocr_batch_size: int = 16,
    max_images: int = 5000,
) -> None:
    """Create a Cloud Run Job for training.

    Args:
        project: GCP project ID.
        repo: Artifact Registry repository name.
        tag: Docker image tag.
        region: Cloud Run region.
        job: Cloud Run Job name.
        data_dir: Dataset directory inside the container.
        detector_batch_size: Detector batch size.
        ocr_batch_size: OCR batch size.
        max_images: Maximum images for OCR training.
    """
    image = f"{region}-docker.pkg.dev/{project}/{repo}/train:{tag}"
    train_command = (
        "cd /app && "
        f"find {data_dir} -type f -print -quit 2>/dev/null | grep -q . && "
        f"uv run -m {PROJECT_NAME}.train train-both {data_dir} "
        f"--detector-batch-size {detector_batch_size} "
        f"--ocr-batch-size {ocr_batch_size} "
        f"--max-images {max_images}"
    )
    cmd = (
        f"gcloud run jobs create {job} "
        f"--region={region} "
        f"--image={image} "
        "--command=sh "
        f'--args=-c,"{train_command}"'
    )
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def train_release(
    ctx: Context,
    project: str = "mlops-license-plate-484109",
    repo: str = "license-plate-repo",
    tag: str = "latest",
    region: str = "europe-west1",
    job: str = "train-job",
    data_dir: str = DEFAULT_DATA_DIR,
    detector_batch_size: int = 8,
    ocr_batch_size: int = 16,
    max_images: int = 500,
) -> None:
    """Build, tag, push, update, and execute the training job.

    Args:
        project: GCP project ID.
        repo: Artifact Registry repository name.
        tag: Docker image tag.
        region: Cloud Run/Artifact Registry region.
        job: Cloud Run Job name.
        data_dir: Dataset directory inside the container.
        detector_batch_size: Detector batch size.
        ocr_batch_size: OCR batch size.
        max_images: Maximum images for OCR training.
    """
    image = f"{region}-docker.pkg.dev/{project}/{repo}/train:{tag}"
    ctx.run(
        f"docker buildx build --platform linux/amd64 -t {image} -f dockerfiles/train.dockerfile . --push",
        echo=True,
        pty=not WINDOWS,
    )
    train_command = (
        "cd /app && "
        f"find {data_dir} -type f -print -quit 2>/dev/null | grep -q . && "
        f"uv run -m {PROJECT_NAME}.train train-both {data_dir} "
        f"--detector-batch-size {detector_batch_size} "
        f"--ocr-batch-size {ocr_batch_size} "
        f"--max-images {max_images}"
    )
    update_cmd = (
        f"gcloud run jobs update {job} "
        f"--region={region} "
        f"--image={image} "
        "--command=sh "
        f'--args=-c,"{train_command}"'
    )
    ctx.run(update_cmd, echo=True, pty=not WINDOWS)
    ctx.run(f"gcloud run jobs execute {job} --region={region}", echo=True, pty=not WINDOWS)


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


# create cloud run using Engine:
# gcloud run jobs create train-job3 --region=europe-west1 --memory=4Gi --set-env-vars WANDB_API_KEY="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" --image=europe-west1-docker.pkg.dev/mlops-license-plate-484109/license-plate-repo/train:latest --command=uv --args=run,-m,ml_ops.train,train-both,data/ccpd_small,--max-images,500
# gcloud run jobs execute train-job3 --region=europe-west1


# gcloud run jobs create train-job --region=europe-west1 --memory=4Gi --set-secrets WANDB_API_KEY=projects/529952243062/secrets/WANDB_API_KEY:latest --image=europe-west1-docker.pkg.dev/mlops-license-plate-484109/license-plate-repo/train:latest --command=sh --args=-c,"uv run dvc pull && uv run -m ml_ops.train train-both data/ccpd_small --max-images 500"
# gcloud run jobs execute train-job4
