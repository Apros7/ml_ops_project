"""Training scripts for license plate detection and OCR."""

import csv
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import typer
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import wandb
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from loguru import logger

from ml_ops.model import PlateDetector, PlateOCR
from ml_ops.data import (
    CCPDDataModule,
    CCPDOCRDataset,
    ENGLISH_CHARS,
    PROVINCES,
    export_yolo_format,
    ENGLISH_NUM_CLASSES,
    NUM_CLASSES,
)

app = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"

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
    logger.info(f"Logging to Weights & Biases: {entity}/{project}")
    return True


def _finish_wandb_run(active: bool) -> None:
    """Finish WandB run if one was started."""

    if active:
        wandb.finish()


def _log_yolo_results_to_wandb(results_file: Path) -> None:
    """Stream YOLO results.csv metrics to the active W&B run if present."""

    if not results_file.exists():
        logger.warning(f"W&B logging skipped, results file not found: {results_file}")
        return

    with results_file.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            payload: dict[str, float | int | str] = {}
            for key, value in row.items():
                if value is None:
                    continue
                if key.lower() == "epoch":
                    try:
                        payload["epoch"] = int(float(value))
                    except ValueError:
                        payload["epoch"] = value
                    continue
                try:
                    payload[key] = float(value)
                except ValueError:
                    payload[key] = value

            if payload:
                wandb.log(payload, step=payload.get("epoch"))


def _export_best_model(src: Path, dst: Path) -> None:
    """Copy a trained model artifact into the project's models folder.

    Args:
        src: Source file path.
        dst: Destination file path.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    logger.info(f"Exported best model: {src} -> {dst}")


def _build_easyocr_character_list(*, english_only: bool) -> str:
    """Build an ordered EasyOCR character set for license plates.

    Args:
        english_only: If True, only include A-Z (no I/O) and digits.

    Returns:
        A string containing all characters in order.
    """
    if english_only:
        return ENGLISH_CHARS

    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    digits = "0123456789"
    ordered = list(PROVINCES) + list(letters) + list(digits)
    return "".join(dict.fromkeys(ordered))


def _normalize_easyocr_plate_text(text: str, *, english_only: bool) -> str:
    """Normalize plate text to match the EasyOCR character set.

    Args:
        text: Plate text from the dataset.
        english_only: Whether the model is trained on the English part only.

    Returns:
        Normalized plate text.
    """
    normalized = text.replace(" ", "").upper()
    if english_only:
        return normalized.replace("I", "1").replace("O", "0").replace("L", "1").replace("|", "1")
    return normalized


def _easyocr_collate_fn(batch: list[dict[str, Any]], *, english_only: bool) -> dict[str, Any]:
    """Collate function for EasyOCR fine-tuning.

    Args:
        batch: List of CCPDOCRDataset samples.
        english_only: Whether the labels contain only the English part of the plate.

    Returns:
        Batch dictionary containing grayscale images and text labels.
    """
    images = torch.stack([item["image"] for item in batch])
    texts = [_normalize_easyocr_plate_text(item["plate_text"], english_only=english_only) for item in batch]

    r, g, b = images[:, 0], images[:, 1], images[:, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = gray.unsqueeze(1)
    gray = (gray - 0.5) / 0.5

    return {"images": gray, "texts": texts}


def _ctc_greedy_decode(indices: torch.Tensor, *, idx_to_char: list[str]) -> list[str]:
    """Greedy-decode CTC predictions.

    Args:
        indices: Tensor of shape (batch, seq_len) with argmax class indices.
        idx_to_char: Index-to-character mapping where index 0 is blank.

    Returns:
        List of decoded strings.
    """
    decoded: list[str] = []
    for seq in indices.tolist():
        chars: list[str] = []
        prev = -1
        for idx in seq:
            if idx != 0 and idx != prev:
                chars.append(idx_to_char[idx])
            prev = idx
        decoded.append("".join(chars))
    return decoded


def _calculate_char_accuracy(pred: str, gt: str) -> float:
    """Calculate character-level accuracy between prediction and ground truth."""
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    correct = sum(1 for p, g in zip(pred, gt) if p == g)
    return correct / max(len(pred), len(gt))


def _train_easyocr_with_cfg(
    cfg: DictConfig,
    *,
    data_dir: Path | None = None,
    split_dir: Path | None = None,
    batch_size: int | None = None,
    max_epochs: int | None = None,
    learning_rate: float | None = None,
    hidden_size: int | None = None,
    img_height: int | None = None,
    img_width: int | None = None,
    max_images: int | None = None,
    num_workers: int | None = None,
    project: Path | None = None,
    name: str | None = None,
    english_only: bool | None = None,
) -> tuple[Path, str]:
    """Fine-tune an EasyOCR recognizer model on CCPD crops.

    Args:
        cfg: Hydra config.
        data_dir: Path to CCPD dataset directory.
        split_dir: Optional directory with train/val split files.
        batch_size: Batch size.
        max_epochs: Maximum epochs.
        learning_rate: Learning rate.
        hidden_size: LSTM hidden size for the EasyOCR recognizer.
        img_height: OCR image height.
        img_width: OCR image width.
        max_images: Maximum total images (train/val split controlled via config).
        num_workers: DataLoader workers.
        project: Output directory for logs/checkpoints.
        name: Experiment name.
        english_only: If True, train on the English part only (6 chars).

    Returns:
        Tuple of (project_path, experiment_name).
    """
    from torch.utils.data import DataLoader
    from easyocr.model.vgg_model import Model as EasyOCRVGGModel
    from easyocr.utils import CTCLabelConverter

    data_cfg = cfg.data
    training_cfg = cfg.training.ocr
    model_cfg = cfg.model.ocr
    wandb_cfg = cfg.wandb_configs

    resolved_data_dir = _normalize_path(data_dir) or _normalize_path(data_cfg.get("data_dir")) or PROJECT_ROOT / "data"
    resolved_split_dir = _normalize_path(split_dir) or _normalize_path(data_cfg.get("split_dir"))
    project_path = _normalize_path(project) or _normalize_path(training_cfg.get("project_dir")) or RUNS_DIR / "easyocr"
    experiment_name = name or training_cfg.get("experiment_name", "easyocr_ocr")

    batch_size_value = batch_size or training_cfg.get("batch_size", 64)
    max_epochs_value = max_epochs or training_cfg.get("max_epochs", 15)
    learning_rate_value = learning_rate or training_cfg.get("learning_rate", 1e-3)
    num_workers_value = num_workers or data_cfg.get("num_workers", 4)

    english_only_value = english_only if english_only is not None else model_cfg.get("english_only", True)
    img_height_value = img_height or model_cfg.get("img_height", 32)
    img_width_value = img_width or model_cfg.get("img_width", 200)
    hidden_size_value = hidden_size or model_cfg.get("hidden_size", 256)
    output_channel_value = int(model_cfg.get("output_channel", 256))

    max_images_value = max_images or data_cfg.get("max_total_images")
    if max_images_value is None:
        raise ValueError(
            "Missing value for max_images. Set it either via the CLI argument "
            "'--max-images' or by specifying 'data.max_total_images' in the Hydra configuration."
        )

    train_split = float(data_cfg.get("train_split", 0.8))
    max_train_images = int(max_images_value * train_split)
    max_val_images = max_images_value - max_train_images

    logger.info(f"Max images: {max_images_value} -> {max_train_images} train, {max_val_images} val")
    logger.info(f"EasyOCR Mode: {'English-only (6 chars)' if english_only_value else 'Full (7 chars with Chinese)'}")

    project_path.mkdir(parents=True, exist_ok=True)
    experiment_dir = project_path / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_dir = resolved_data_dir / "train" if (resolved_data_dir / "train").exists() else resolved_data_dir
    val_dir = resolved_data_dir / "val" if (resolved_data_dir / "val").exists() else resolved_data_dir

    train_split_file = resolved_split_dir / "train.txt" if resolved_split_dir else None
    val_split_file = resolved_split_dir / "val.txt" if resolved_split_dir else None

    train_dataset = CCPDOCRDataset(
        train_dir,
        split_file=train_split_file,
        img_height=img_height_value,
        img_width=img_width_value,
        max_images=max_train_images,
        english_only=english_only_value,
    )
    val_dataset = CCPDOCRDataset(
        val_dir,
        split_file=val_split_file,
        img_height=img_height_value,
        img_width=img_width_value,
        max_images=max_val_images,
        english_only=english_only_value,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_value,
        shuffle=True,
        num_workers=num_workers_value,
        pin_memory=True,
        collate_fn=partial(_easyocr_collate_fn, english_only=english_only_value),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_value,
        shuffle=False,
        num_workers=num_workers_value,
        pin_memory=True,
        collate_fn=partial(_easyocr_collate_fn, english_only=english_only_value),
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA for EasyOCR fine-tuning: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")
        logger.warning("MPS is available, but CTCLoss is not supported on MPS. Using CPU for EasyOCR fine-tuning.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for EasyOCR fine-tuning.")

    character_list = _build_easyocr_character_list(english_only=english_only_value)
    converter = CTCLabelConverter(character_list)
    num_class = len(converter.character)

    model = EasyOCRVGGModel(
        input_channel=1,
        output_channel=output_channel_value,
        hidden_size=hidden_size_value,
        num_class=num_class,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_value, weight_decay=1e-4)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    wandb_active = _start_wandb_run(
        wandb_cfg,
        name=f"easyocr-{experiment_name}",
        metadata={
            "architecture": "EasyOCR-VGG",
            "task": "OCR",
            "english_only": english_only_value,
            "img_height": img_height_value,
            "img_width": img_width_value,
            "hidden_size": hidden_size_value,
            "output_channel": output_channel_value,
            "learning_rate": learning_rate_value,
            "batch_size": batch_size_value,
            "max_epochs": max_epochs_value,
            "max_train_images": max_train_images,
            "max_val_images": max_val_images,
            "dataset": data_cfg.get("name", "CCPD"),
        },
    )

    best_val_exact = -1.0
    epochs_no_improve = 0
    patience_cfg = training_cfg.get("patience", 5)
    patience = int(patience_cfg) if patience_cfg is not None else 0
    early_stopping_enabled = patience > 0
    grad_clip = float(training_cfg.get("gradient_clip_val", 5.0))

    best_local_path = experiment_dir / "ocr_best.pth"

    if not early_stopping_enabled:
        logger.info("Early stopping disabled for OCR training (patience <= 0).")

    for epoch in range(max_epochs_value):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{max_epochs_value} [train]",
            unit="batch",
            leave=False,
        )
        for batch in train_bar:
            images = batch["images"].to(device, non_blocking=(device.type == "cuda"))
            texts = batch["texts"]
            batch_size_actual = images.size(0)

            max_len = max(len(t) for t in texts) if texts else 0
            text_for_pred = torch.zeros(batch_size_actual, max_len + 1, dtype=torch.long, device=device)

            preds = model(images, text_for_pred)
            preds_log_probs = preds.log_softmax(2)
            preds_for_ctc = preds_log_probs.permute(1, 0, 2)

            input_lengths = torch.full(
                (batch_size_actual,),
                preds_for_ctc.size(0),
                dtype=torch.long,
                device=device,
            )
            targets, target_lengths = converter.encode(texts)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            loss = criterion(preds_for_ctc, targets, input_lengths, target_lengths)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            loss_value = float(loss.item())
            train_loss_sum += loss_value
            train_batches += 1
            train_bar.set_postfix(loss=loss_value, avg=train_loss_sum / train_batches)

        train_loss = train_loss_sum / max(train_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        total = 0
        exact = 0
        char_acc_sum = 0.0

        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{max_epochs_value} [val]",
                unit="batch",
                leave=False,
            )
            for batch in val_bar:
                images = batch["images"].to(device, non_blocking=(device.type == "cuda"))
                texts = batch["texts"]
                batch_size_actual = images.size(0)

                max_len = max(len(t) for t in texts) if texts else 0
                text_for_pred = torch.zeros(batch_size_actual, max_len + 1, dtype=torch.long, device=device)

                preds = model(images, text_for_pred)
                preds_log_probs = preds.log_softmax(2)
                preds_for_ctc = preds_log_probs.permute(1, 0, 2)

                input_lengths = torch.full(
                    (batch_size_actual,),
                    preds_for_ctc.size(0),
                    dtype=torch.long,
                    device=device,
                )
                targets, target_lengths = converter.encode(texts)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                loss = criterion(preds_for_ctc, targets, input_lengths, target_lengths)
                loss_value = float(loss.item())
                val_loss_sum += loss_value
                val_batches += 1

                pred_indices = preds_log_probs.argmax(2)
                decoded = _ctc_greedy_decode(pred_indices, idx_to_char=converter.character)

                for pred, gt in zip(decoded, texts):
                    total += 1
                    if pred == gt:
                        exact += 1
                    char_acc_sum += _calculate_char_accuracy(pred, gt)
                val_bar.set_postfix(
                    loss=loss_value,
                    avg=val_loss_sum / val_batches,
                    exact=(exact / total) if total else 0.0,
                )

        val_loss = val_loss_sum / max(val_batches, 1)
        val_exact = exact / total if total > 0 else 0.0
        val_char = char_acc_sum / total if total > 0 else 0.0

        logger.info(
            f"Epoch {epoch+1}/{max_epochs_value} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_exact={val_exact:.4f} | val_char={val_char:.4f}"
        )

        if wandb_active:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_exact_accuracy": val_exact,
                    "val_char_accuracy": val_char,
                },
                step=epoch + 1,
            )

        if val_exact > best_val_exact:
            best_val_exact = val_exact
            epochs_no_improve = 0

            state_dict = {f"module.{k}": v.detach().cpu() for k, v in model.state_dict().items()}
            payload = {
                "state_dict": state_dict,
                "network_params": {
                    "input_channel": 1,
                    "output_channel": output_channel_value,
                    "hidden_size": hidden_size_value,
                },
                "characters": character_list,
                "img_height": img_height_value,
                "img_width": img_width_value,
                "english_only": english_only_value,
            }
            torch.save(payload, best_local_path)
            logger.info(f"New best EasyOCR checkpoint saved: {best_local_path} (val_exact={best_val_exact:.4f})")
        else:
            epochs_no_improve += 1
            if early_stopping_enabled and epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered (patience={patience}). Best val_exact={best_val_exact:.4f}")
                break

    _finish_wandb_run(wandb_active)

    if best_local_path.exists():
        _export_best_model(best_local_path, MODELS_DIR / "ocr_best.pth")
    else:
        logger.warning("No EasyOCR best checkpoint found to export.")

    logger.info(f"\nTraining complete! Results saved to {experiment_dir}")
    logger.info(f"  - Best OCR weights: {MODELS_DIR / 'ocr_best.pth'}")

    return project_path, experiment_name


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
    wandb_cfg = cfg.wandb_configs

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
        raise ValueError(
            "Cannot determine train/val split sizes. Set either (1) both "
            "max_train_images and max_val_images, or (2) max_total_images "
            "together with train_split, via config or CLI overrides."
        )

    project_path.mkdir(parents=True, exist_ok=True)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results will be saved to: {project_path / experiment_name}")
    logger.info("Step 1: Converting CCPD to YOLO format...")

    train_output = resolved_output_dir / "train"
    val_output = resolved_output_dir / "val"

    train_data_dir = resolved_data_dir / "train" if (resolved_data_dir / "train").exists() else resolved_data_dir
    val_data_dir = resolved_data_dir / "val" if (resolved_data_dir / "val").exists() else resolved_data_dir

    if train_data_dir != resolved_data_dir:
        logger.info(f"Detected pre-split dataset: {resolved_data_dir}")
        logger.info(f"  Train folder: {train_data_dir}")
        logger.info(f"  Val folder: {val_data_dir}")

    train_split_file = resolved_split_dir / "train.txt" if resolved_split_dir else None
    val_split_file = resolved_split_dir / "val.txt" if resolved_split_dir else None

    enable_profiling = training_cfg.get("enable_profiling", False)
    export_yolo_format(
        train_data_dir, train_output, train_split_file, max_images=max_train_value, enable_profiling=enable_profiling
    )
    export_yolo_format(
        val_data_dir, val_output, val_split_file, max_images=max_val_value, enable_profiling=enable_profiling
    )

    logger.info("Step 2: Creating data.yaml config...")
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

    logger.info("Step 3: Training YOLOv8...")

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

    if wandb_active:
        results_file = project_path / experiment_name / "results.csv"
        _log_yolo_results_to_wandb(results_file)

    _finish_wandb_run(wandb_active)

    logger.info(f"\nTraining complete! Results saved to {project_path / experiment_name}")
    logger.info(f"  - Training curves: {project_path / experiment_name / 'results.png'}")
    logger.info(f"  - Metrics CSV: {project_path / experiment_name / 'results.csv'}")
    logger.info(f"  - Best weights: {project_path / experiment_name / 'weights' / 'best.pt'}")
    if wandb_active:
        entity = wandb_cfg.get("entity", WANDB_ENTITY)
        project = wandb_cfg.get("project", WANDB_PROJECT)
        logger.info(f"  - W&B Dashboard: https://wandb.ai/{entity}/{project}")

    best_weights = project_path / experiment_name / "weights" / "best.pt"
    if best_weights.exists():
        _export_best_model(best_weights, MODELS_DIR / "yolo_best.pt")
    else:
        logger.warning(f"Could not export best YOLO weights; file not found: {best_weights}")

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
    wandb_cfg = cfg.wandb_configs

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
        raise ValueError(
            "Missing value for max_images. Set it either via the CLI argument "
            "'--max-images' or by specifying 'data.max_total_images' in the Hydra "
            "configuration (for example in configs/)."
        )

    train_split = float(data_cfg.get("train_split", 0.8))
    max_train_images = int(max_images_value * train_split)
    max_val_images = max_images_value - max_train_images

    logger.info(f"Max images: {max_images_value} -> {max_train_images} train, {max_val_images} val")
    logger.info(f"OCR Mode: {'English-only (6 chars)' if english_only_value else 'Full (7 chars with Chinese)'}")

    project_path.mkdir(parents=True, exist_ok=True)
    output_dir = project_path / experiment_name / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results will be saved to: {project_path / experiment_name}")
    logger.info(f"Prediction visualizations will be saved to: {output_dir}")
    logger.info("Setting up data module...")

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

    logger.info("Creating CRNN model...")
    enable_profiling = training_cfg.get("enable_profiling", False)
    profile_every_n_steps = training_cfg.get("profile_every_n_steps", 100)
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
        enable_profiling=enable_profiling,
        profile_every_n_steps=profile_every_n_steps,
    )

    checkpoint_dir = project_path / experiment_name / "checkpoints"
    callbacks: list[Any] = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="ocr-{epoch:02d}-{val_accuracy:.4f}",
            monitor="val_accuracy",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    patience_cfg = training_cfg.get("patience", 5)
    patience = int(patience_cfg) if patience_cfg is not None else 0
    if patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_accuracy",
                patience=patience,
                mode="max",
                verbose=True,
            )
        )
    else:
        logger.info("Early stopping disabled for CRNN OCR training (patience <= 0).")

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
        logger.info(
            f"Logging to Weights & Biases: {wandb_cfg.get('entity', WANDB_ENTITY)}/"
            f"{wandb_cfg.get('project', WANDB_PROJECT)}"
        )

    force_cpu = torch.backends.mps.is_available()
    accelerator, devices = get_accelerator(force_cpu=force_cpu)

    if force_cpu:
        logger.warning("Note: Using CPU for OCR training (CTC loss not supported on MPS)")

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

    logger.info(f"\nTraining complete! Results saved to {project_path / experiment_name}")
    logger.info(f"  - Metrics CSV: {project_path / experiment_name}")
    logger.info(f"  - Checkpoints: {checkpoint_dir}")
    if wandb_logger is not None:
        entity = wandb_cfg.get("entity", WANDB_ENTITY)
        project = wandb_cfg.get("project", WANDB_PROJECT)
        logger.info(f"  - W&B Dashboard: https://wandb.ai/{entity}/{project}")

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
    """Fine-tune an EasyOCR recognizer on CCPD crops using Hydra configs."""

    cfg = load_hydra_config(config_path=config_path, config_name=config_name, overrides=overrides)
    _train_easyocr_with_cfg(
        cfg,
        data_dir=data_dir,
        split_dir=split_dir,
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        img_height=img_height,
        img_width=img_width,
        max_images=max_images,
        num_workers=num_workers,
        project=project,
        name=name,
        english_only=english_only,
    )


@app.command()
def train_crnn(
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
    """Train the in-house CRNN OCR model (Lightning) using Hydra configs.

    This command is kept for comparison and for unit tests; the default `train-ocr`
    command fine-tunes an EasyOCR recognizer.
    """
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

    logger.info(f"Results will be saved to: {RUNS_DIR}")
    logger.info("=" * 50)
    logger.info("PHASE 1: Training License Plate Detector")
    logger.info("=" * 50)

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

    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: Training License Plate OCR")
    logger.info("=" * 50)

    ocr_project, ocr_name = _train_easyocr_with_cfg(
        cfg,
        data_dir=data_dir,
        split_dir=split_dir,
        batch_size=ocr_batch_size,
        max_epochs=ocr_epochs,
        max_images=max_images,
        english_only=english_only,
    )

    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 50)
    logger.info(f"\nAll results saved to: {RUNS_DIR}")
    logger.info(f"  - Detector results: {detector_project / detector_name}")
    logger.info(f"  - OCR results: {ocr_project / ocr_name}")


if __name__ == "__main__":
    app()
