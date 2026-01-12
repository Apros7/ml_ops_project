"""Visualization utilities for license plate detection and OCR results."""

from pathlib import Path

import cv2
import numpy as np
import typer

from ml_ops.model import LicensePlateRecognizer
from ml_ops.data import parse_ccpd_filename

app = typer.Typer()


def draw_detection(
    image: np.ndarray,
    bbox: list[int],
    plate_text: str,
    confidence: float,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a detection on the image.

    Args:
        image: Input image (BGR format).
        bbox: Bounding box [x1, y1, x2, y2].
        plate_text: Recognized plate text.
        confidence: Detection confidence.
        color: Box color in BGR.
        thickness: Line thickness.

    Returns:
        Image with drawn detection.
    """
    x1, y1, x2, y2 = map(int, bbox)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    label = f"{plate_text} ({confidence:.2f})"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    cv2.rectangle(
        image,
        (x1, y1 - text_height - 10),
        (x1 + text_width + 10, y1),
        color,
        -1,
    )

    cv2.putText(
        image,
        label,
        (x1 + 5, y1 - 5),
        font,
        font_scale,
        (0, 0, 0),
        font_thickness,
    )

    return image


def visualize_predictions(
    image_path: str,
    predictions: list[dict],
    output_path: str | None = None,
    show: bool = True,
) -> np.ndarray:
    """Visualize license plate predictions on an image.

    Args:
        image_path: Path to input image.
        predictions: List of prediction dictionaries with bbox, plate_text, confidence.
        output_path: Optional path to save the visualization.
        show: Whether to display the image.

    Returns:
        Annotated image.
    """
    image = cv2.imread(image_path)

    for pred in predictions:
        image = draw_detection(
            image,
            pred["bbox"],
            pred["plate_text"],
            pred["confidence"],
        )

    if output_path:
        cv2.imwrite(output_path, image)

    if show:
        cv2.imshow("License Plate Recognition", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def visualize_ground_truth(
    image_path: str,
    output_path: str | None = None,
    show: bool = True,
) -> np.ndarray:
    """Visualize ground truth annotations from CCPD filename.

    Args:
        image_path: Path to CCPD image.
        output_path: Optional path to save visualization.
        show: Whether to display the image.

    Returns:
        Annotated image.
    """
    image = cv2.imread(image_path)
    filename = Path(image_path).name

    try:
        ann = parse_ccpd_filename(filename)
        image = draw_detection(
            image,
            ann["bbox"],
            ann["plate_text"],
            1.0,
            color=(255, 0, 0),
        )
    except (ValueError, IndexError) as e:
        print(f"Could not parse filename: {e}")

    if output_path:
        cv2.imwrite(output_path, image)

    if show:
        cv2.imshow("Ground Truth", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def compare_prediction_ground_truth(
    image_path: str,
    predictions: list[dict],
    output_path: str | None = None,
    show: bool = True,
) -> np.ndarray:
    """Compare predictions with ground truth side by side.

    Args:
        image_path: Path to CCPD image.
        predictions: List of prediction dictionaries.
        output_path: Optional path to save visualization.
        show: Whether to display the image.

    Returns:
        Combined comparison image.
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    gt_image = image.copy()
    pred_image = image.copy()

    filename = Path(image_path).name
    try:
        ann = parse_ccpd_filename(filename)
        gt_image = draw_detection(
            gt_image,
            ann["bbox"],
            f"GT: {ann['plate_text']}",
            1.0,
            color=(255, 0, 0),
        )
    except (ValueError, IndexError):
        pass

    for pred in predictions:
        pred_image = draw_detection(
            pred_image,
            pred["bbox"],
            f"Pred: {pred['plate_text']}",
            pred["confidence"],
            color=(0, 255, 0),
        )

    combined = np.hstack([gt_image, pred_image])

    cv2.putText(combined, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(combined, "Prediction", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, combined)

    if show:
        cv2.imshow("Comparison", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return combined


@app.command()
def visualize(
    image_path: str = typer.Argument(..., help="Path to input image"),
    detector_weights: str = typer.Option("yolov8n.pt", help="Path to detector weights"),
    ocr_checkpoint: str = typer.Option(None, help="Path to OCR checkpoint"),
    output: str = typer.Option(None, help="Output path for visualization"),
    compare: bool = typer.Option(False, help="Compare with ground truth"),
    no_display: bool = typer.Option(False, help="Don't display the image"),
) -> None:
    """Visualize license plate recognition results.

    Args:
        image_path: Path to input image.
        detector_weights: Path to YOLO detector weights.
        ocr_checkpoint: Path to OCR model checkpoint.
        output: Output path for saving visualization.
        compare: Whether to show comparison with ground truth.
        no_display: If True, don't display the image window.
    """
    recognizer = LicensePlateRecognizer(
        detector_weights=detector_weights,
        ocr_checkpoint=ocr_checkpoint,
    )

    predictions = recognizer.recognize(image_path)

    print(f"\nFound {len(predictions)} license plate(s):")
    for i, pred in enumerate(predictions, 1):
        print(f"  {i}. {pred['plate_text']} (confidence: {pred['confidence']:.3f})")

    if compare:
        compare_prediction_ground_truth(
            image_path,
            predictions,
            output_path=output,
            show=not no_display,
        )
    else:
        visualize_predictions(
            image_path,
            predictions,
            output_path=output,
            show=not no_display,
        )


@app.command()
def batch_visualize(
    input_dir: str = typer.Argument(..., help="Directory with input images"),
    output_dir: str = typer.Argument(..., help="Output directory for visualizations"),
    detector_weights: str = typer.Option("yolov8n.pt", help="Path to detector weights"),
    ocr_checkpoint: str = typer.Option(None, help="Path to OCR checkpoint"),
    max_images: int = typer.Option(None, help="Maximum number of images to process"),
) -> None:
    """Batch visualize license plate recognition on multiple images.

    Args:
        input_dir: Directory containing input images.
        output_dir: Directory to save visualizations.
        detector_weights: Path to YOLO detector weights.
        ocr_checkpoint: Path to OCR checkpoint.
        max_images: Maximum number of images to process.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    recognizer = LicensePlateRecognizer(
        detector_weights=detector_weights,
        ocr_checkpoint=ocr_checkpoint,
    )

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(input_path.rglob(ext))

    if max_images:
        image_paths = image_paths[:max_images]

    for img_path in image_paths:
        predictions = recognizer.recognize(str(img_path))
        output_file = output_path / f"vis_{img_path.name}"
        compare_prediction_ground_truth(
            str(img_path),
            predictions,
            output_path=str(output_file),
            show=False,
        )
        print(f"Processed: {img_path.name}")

    print(f"\nSaved {len(image_paths)} visualizations to {output_dir}")


if __name__ == "__main__":
    app()
