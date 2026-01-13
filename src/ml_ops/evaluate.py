"""Evaluation scripts for license plate detection and OCR."""

from pathlib import Path
from typing import Any

import typer
import torch
from tqdm import tqdm
from loguru import logger

from ml_ops.model import PlateDetector, PlateOCR, LicensePlateRecognizer
from ml_ops.data import CCPDDataModule, parse_ccpd_filename

app = typer.Typer()


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """Calculate Intersection over Union between two bounding boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2].
        box2: Second bounding box [x1, y1, x2, y2].

    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_char_accuracy(pred: str, gt: str) -> float:
    """Calculate character-level accuracy.

    Args:
        pred: Predicted plate text.
        gt: Ground truth plate text.

    Returns:
        Character accuracy between 0 and 1.
    """
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0

    correct = sum(1 for p, g in zip(pred, gt) if p == g)
    return correct / max(len(pred), len(gt))


@app.command()
def evaluate_detector(
    data_dir: Path = typer.Argument(..., help="Path to test images"),
    weights: str = typer.Option("runs/detect/plate_detection/weights/best.pt", help="Path to detector weights"),
    conf_threshold: float = typer.Option(0.25, help="Confidence threshold"),
    iou_threshold: float = typer.Option(0.5, help="IoU threshold for matching"),
    max_samples: int = typer.Option(None, help="Maximum samples to evaluate"),
) -> dict[str, float]:
    """Evaluate the license plate detector.

    Args:
        data_dir: Directory containing test images.
        weights: Path to trained model weights.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for matching predictions to ground truth.
        max_samples: Maximum number of samples to evaluate.

    Returns:
        Dictionary with evaluation metrics.
    """
    detector = PlateDetector(model_name=weights)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(Path(data_dir).rglob(ext))
    image_paths = sorted(image_paths)

    if max_samples:
        image_paths = image_paths[:max_samples]

    tp, fp, fn = 0, 0, 0

    for img_path in tqdm(image_paths, desc="Evaluating detector"):
        try:
            gt_ann = parse_ccpd_filename(img_path.name)
            gt_bbox = gt_ann["bbox"]
        except (ValueError, IndexError):
            continue

        detections = detector.predict(str(img_path), conf=conf_threshold)

        if not detections:
            fn += 1
            continue

        best_iou = 0
        for det in detections:
            iou = calculate_iou(det["bbox"], gt_bbox)
            best_iou = max(best_iou, iou)

        if best_iou >= iou_threshold:
            tp += 1
            fp += len(detections) - 1
        else:
            fn += 1
            fp += len(detections)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }

    logger.info("\n=== Detection Evaluation Results ===")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"TP: {tp}, FP: {fp}, FN: {fn}")

    return results


@app.command()
def evaluate_ocr(
    data_dir: Path = typer.Argument(..., help="Path to CCPD dataset"),
    checkpoint: str = typer.Option(..., help="Path to OCR checkpoint"),
    split_dir: Path = typer.Option(None, help="Directory with split files"),
    batch_size: int = typer.Option(64, help="Batch size"),
    num_workers: int = typer.Option(4, help="Number of workers"),
) -> dict[str, float]:
    """Evaluate the license plate OCR model.

    Args:
        data_dir: Path to CCPD dataset.
        checkpoint: Path to OCR model checkpoint.
        split_dir: Directory containing split files.
        batch_size: Batch size for evaluation.
        num_workers: Number of dataloader workers.

    Returns:
        Dictionary with evaluation metrics.
    """
    model = PlateOCR.load_from_checkpoint(checkpoint)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_module = CCPDDataModule(
        data_dir=data_dir,
        split_dir=split_dir,
        task="ocr",
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    total_samples = 0
    exact_matches = 0
    total_char_accuracy = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating OCR"):
            images = batch["images"].to(device)
            plate_texts = batch["plate_texts"]

            log_probs = model(images)
            predictions = model.decode(log_probs)

            for pred, gt in zip(predictions, plate_texts):
                total_samples += 1
                if pred == gt:
                    exact_matches += 1
                total_char_accuracy += calculate_char_accuracy(pred, gt)

    exact_accuracy = exact_matches / total_samples if total_samples > 0 else 0
    char_accuracy = total_char_accuracy / total_samples if total_samples > 0 else 0

    results = {
        "exact_accuracy": exact_accuracy,
        "char_accuracy": char_accuracy,
        "total_samples": total_samples,
        "exact_matches": exact_matches,
    }

    logger.info("\n=== OCR Evaluation Results ===")
    logger.info(f"Exact Match Accuracy: {exact_accuracy:.4f} ({exact_matches}/{total_samples})")
    logger.info(f"Character Accuracy: {char_accuracy:.4f}")

    return results


@app.command()
def evaluate_pipeline(
    data_dir: Path = typer.Argument(..., help="Path to test images"),
    detector_weights: str = typer.Option("runs/detect/plate_detection/weights/best.pt", help="Detector weights"),
    ocr_checkpoint: str = typer.Option("runs/ocr/plate_ocr/checkpoints/last.ckpt", help="OCR checkpoint"),
    conf_threshold: float = typer.Option(0.25, help="Detection confidence threshold"),
    max_samples: int = typer.Option(None, help="Maximum samples to evaluate"),
) -> dict[str, Any]:
    """Evaluate the complete license plate recognition pipeline.

    Args:
        data_dir: Directory containing test images.
        detector_weights: Path to detector weights.
        ocr_checkpoint: Path to OCR checkpoint.
        conf_threshold: Detection confidence threshold.
        max_samples: Maximum samples to evaluate.

    Returns:
        Dictionary with evaluation metrics.
    """
    recognizer = LicensePlateRecognizer(
        detector_weights=detector_weights,
        ocr_checkpoint=ocr_checkpoint,
    )

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(Path(data_dir).rglob(ext))
    image_paths = sorted(image_paths)

    if max_samples:
        image_paths = image_paths[:max_samples]

    total_samples = 0
    detection_correct = 0
    ocr_correct = 0
    full_pipeline_correct = 0

    for img_path in tqdm(image_paths, desc="Evaluating pipeline"):
        try:
            gt_ann = parse_ccpd_filename(img_path.name)
            gt_bbox = gt_ann["bbox"]
            gt_text = gt_ann["plate_text"]
        except (ValueError, IndexError):
            continue

        total_samples += 1

        results = recognizer.recognize(str(img_path), conf_threshold=conf_threshold)

        if not results:
            continue

        best_result = max(results, key=lambda x: x["confidence"])
        iou = calculate_iou(best_result["bbox"], gt_bbox)

        if iou >= 0.5:
            detection_correct += 1

            if best_result["plate_text"] == gt_text:
                ocr_correct += 1
                full_pipeline_correct += 1

    detection_accuracy = detection_correct / total_samples if total_samples > 0 else 0
    ocr_accuracy = ocr_correct / detection_correct if detection_correct > 0 else 0
    pipeline_accuracy = full_pipeline_correct / total_samples if total_samples > 0 else 0

    results = {
        "detection_accuracy": detection_accuracy,
        "ocr_accuracy_given_detection": ocr_accuracy,
        "full_pipeline_accuracy": pipeline_accuracy,
        "total_samples": total_samples,
        "detection_correct": detection_correct,
        "ocr_correct": ocr_correct,
    }

    logger.info("\n=== Full Pipeline Evaluation Results ===")
    logger.info(f"Detection Accuracy: {detection_accuracy:.4f} ({detection_correct}/{total_samples})")
    logger.info(f"OCR Accuracy (given correct detection): {ocr_accuracy:.4f}")
    logger.info(f"Full Pipeline Accuracy: {pipeline_accuracy:.4f} ({full_pipeline_correct}/{total_samples})")

    return results


if __name__ == "__main__":
    app()
