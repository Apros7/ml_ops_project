"""Evaluate EasyOCR on CCPD license plate images for comparison."""

from pathlib import Path

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import typer
from tqdm import tqdm

from ml_ops.data import parse_ccpd_filename, PROVINCES

app = typer.Typer()

RUNS_DIR = Path(__file__).parent.parent.parent / "runs"

# Valid characters for Chinese license plates
PLATE_ALLOWLIST = "".join(PROVINCES[:-1])  # Province characters (exclude 'O' placeholder)
PLATE_ALLOWLIST += "ABCDEFGHJKLMNPQRSTUVWXYZ"  # Letters (no I, O)
PLATE_ALLOWLIST += "0123456789"  # Digits

# Common OCR confusions for license plates
# In Chinese plates: I and O are NEVER used (they look like 1 and 0)
OCR_CORRECTIONS = {
    "I": "1",  # Capital I -> digit 1
    "O": "0",  # Capital O -> digit 0
    "i": "1",  # lowercase i -> digit 1
    "o": "0",  # lowercase o -> digit 0
    "l": "1",  # lowercase L -> digit 1
    "|": "1",  # pipe -> digit 1
    "Z": "2",  # Sometimes Z looks like 2 (context dependent)
    "S": "5",  # Sometimes S looks like 5 (context dependent)
    "B": "8",  # Sometimes B looks like 8 (context dependent)
}


def postprocess_plate_text(text: str) -> str:
    """Apply license plate-specific corrections to OCR output.

    Chinese license plates never use I or O (they look like 1 and 0).
    Format: [Province][Letter][5 alphanumerics]

    Args:
        text: Raw OCR text.

    Returns:
        Corrected text.
    """
    if not text:
        return text

    # First pass: fix obvious I/O -> 1/0 confusions
    # These are ALWAYS wrong in license plates
    result = []
    for i, char in enumerate(text):
        if char in ("I", "i", "l", "|"):
            result.append("1")
        elif char in ("O", "o") and i > 0:  # O at position 0 might be a failed province read
            result.append("0")
        else:
            result.append(char)

    return "".join(result)


def preprocess_plate_image(image: np.ndarray) -> np.ndarray:
    """Preprocess license plate image for better OCR.

    Args:
        image: BGR image of license plate.

    Returns:
        Preprocessed BGR image.
    """
    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    # Merge and convert back
    lab = cv2.merge([l_channel, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced


@app.command()
def evaluate(
    data_dir: Path = typer.Argument(..., help="Path to CCPD dataset directory (e.g., data/ccpd_base)"),
    max_images: int = typer.Option(100, help="Maximum number of images to evaluate"),
    output_dir: Path = typer.Option(None, help="Output directory for results"),
    use_full_image: bool = typer.Option(False, help="Use full image instead of cropped plate"),
) -> None:
    """Evaluate EasyOCR on CCPD license plate images using high-resolution crops.

    By default, uses the original cropped license plate at full resolution.

    Example:
        uv run python -m ml_ops.eval_easyocr data/ccpd_base --max-images 100
    """
    if output_dir is None:
        output_dir = RUNS_DIR / "easyocr"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing EasyOCR (this may take a moment to download models)...")
    reader = easyocr.Reader(["ch_sim", "en"], gpu=False)

    print(f"Loading images from {data_dir}...")
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(data_dir.rglob(ext))
    image_paths = sorted(image_paths)[:max_images]

    print(f"Evaluating {len(image_paths)} images...")
    if use_full_image:
        print("Mode: FULL IMAGE - EasyOCR will process the entire image")
    else:
        print("Mode: HIGH-RES CROP - EasyOCR will process the license plate at original resolution")

    results = []
    for img_path in tqdm(image_paths, desc="Processing"):
        try:
            annotation = parse_ccpd_filename(img_path.name)
            gt_text = annotation["plate_text"]
            bbox = annotation["bbox"]
        except (ValueError, IndexError):
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        plate_crop = image[y_min:y_max, x_min:x_max]
        if plate_crop.size == 0:
            continue

        if use_full_image:
            ocr_input = image
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            ocr_input = plate_crop
            display_image = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)

        crop_h, crop_w = plate_crop.shape[:2]

        # Preprocess for better OCR if using crop
        if not use_full_image:
            ocr_input = preprocess_plate_image(ocr_input)

        # Use optimized parameters for license plate OCR
        ocr_results = reader.readtext(
            ocr_input,
            allowlist=PLATE_ALLOWLIST,
            detail=1,
            paragraph=False,
            min_size=10,
            text_threshold=0.5,
            low_text=0.3,
            mag_ratio=2.0,  # Magnify small images
        )

        all_texts = []
        if ocr_results:
            for r in ocr_results:
                text = r[1].replace(" ", "").upper()
                all_texts.append(text)

        pred_text = "".join(all_texts) if all_texts else ""

        exact_match = pred_text == gt_text
        partial_match = gt_text in pred_text or any(gt_text in t for t in all_texts)

        chars_correct = sum(1 for a, b in zip(pred_text, gt_text) if a == b)
        char_accuracy = chars_correct / len(gt_text) if gt_text else 0

        results.append(
            {
                "image_path": img_path,
                "display_image": display_image,
                "gt_text": gt_text,
                "pred_text": pred_text,
                "all_texts": all_texts,
                "exact_match": exact_match,
                "partial_match": partial_match,
                "char_accuracy": char_accuracy,
                "crop_size": (crop_h, crop_w),
            }
        )

    exact_correct = sum(1 for r in results if r["exact_match"])
    partial_correct = sum(1 for r in results if r["partial_match"])
    avg_char_acc = sum(r["char_accuracy"] for r in results) / len(results) if results else 0

    if results:
        avg_crop_h = sum(r["crop_size"][0] for r in results) / len(results)
        avg_crop_w = sum(r["crop_size"][1] for r in results) / len(results)
    else:
        avg_crop_h, avg_crop_w = 0, 0

    print("\n" + "=" * 60)
    print("EASYOCR EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total images evaluated: {len(results)}")
    print(f"Average crop size:      {avg_crop_h:.0f} x {avg_crop_w:.0f} pixels")
    print(f"Exact matches:          {exact_correct}/{len(results)} ({exact_correct/len(results)*100:.1f}%)")
    print(f"Partial matches:        {partial_correct}/{len(results)} ({partial_correct/len(results)*100:.1f}%)")
    print(f"Average char accuracy:  {avg_char_acc*100:.1f}%")
    print("=" * 60)

    print("\nSaving visualizations...")
    num_pages = (len(results) + 7) // 8

    for page in range(min(num_pages, 5)):
        start_idx = page * 8
        end_idx = min(start_idx + 8, len(results))
        page_results = results[start_idx:end_idx]

        fig, axes = plt.subplots(2, 4, figsize=(16, 6))
        axes = axes.flatten()

        for i, result in enumerate(page_results):
            axes[i].imshow(result["display_image"])
            pred, gt = result["pred_text"][:15], result["gt_text"]
            img_name = result["image_path"].name[:30]
            color = "green" if result["exact_match"] else ("orange" if result["partial_match"] else "red")
            axes[i].set_title(f"{img_name}\nPred: {pred} | GT: {gt}", fontsize=8, color=color)
            axes[i].axis("off")

        for i in range(len(page_results), 8):
            axes[i].axis("off")

        title = f"EasyOCR Results - Page {page + 1}\n"
        title += f"Exact: {exact_correct}/{len(results)} | Partial: {partial_correct}/{len(results)}"
        plt.suptitle(title, fontsize=12)
        plt.tight_layout()

        save_path = output_dir / f"easyocr_results_page_{page + 1:02d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("EasyOCR Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mode: {'Full image' if use_full_image else 'Cropped plate'}\n")
        f.write(f"Total images: {len(results)}\n")
        f.write(f"Exact matches: {exact_correct} ({exact_correct/len(results)*100:.1f}%)\n")
        f.write(f"Partial matches: {partial_correct} ({partial_correct/len(results)*100:.1f}%)\n")
        f.write(f"Average char accuracy: {avg_char_acc*100:.1f}%\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Sample predictions:\n")
        f.write("-" * 80 + "\n")
        for r in results[:30]:
            status = "✓" if r["exact_match"] else ("~" if r["partial_match"] else "✗")
            f.write(f"{status} GT: {r['gt_text']:10} | Pred: {r['pred_text'][:15]:15} | {r['image_path']}\n")

    print(f"\nSummary saved to: {summary_path}")
    print(f"All results saved to: {output_dir}")

    print("\n" + "=" * 60)
    print(f"FINAL SCORE: {exact_correct} / {len(results)} license plates correctly recognized")
    print("=" * 60)

    print("\nSample image paths (for reference):")
    for r in results[:5]:
        status = "✓" if r["exact_match"] else "✗"
        print(f"  {status} {r['image_path']}")


def run_ocr_strategy(
    reader: easyocr.Reader,
    image: np.ndarray,
    strategy_name: str,
    use_allowlist: bool = False,
    decoder: str = "greedy",
    apply_corrections: bool = True,
) -> tuple[str, str, list[str], list]:
    """Run OCR with a specific strategy.

    Args:
        reader: EasyOCR reader instance.
        image: Image to process (BGR).
        strategy_name: Name of the strategy for logging.
        use_allowlist: Whether to use the plate character allowlist.
        decoder: Decoder type ('greedy' or 'beamsearch').
        apply_corrections: Whether to apply license plate corrections (I->1, O->0).

    Returns:
        Tuple of (corrected_text, raw_text, all_texts, raw_results).
    """
    kwargs = {
        "detail": 1,
        "paragraph": False,
    }

    if use_allowlist:
        kwargs["allowlist"] = PLATE_ALLOWLIST

    if decoder == "beamsearch":
        kwargs["decoder"] = "beamsearch"
        kwargs["beamWidth"] = 10

    ocr_results = reader.readtext(image, **kwargs)

    all_texts = []
    if ocr_results:
        for r in ocr_results:
            text = r[1].replace(" ", "").upper()
            all_texts.append(text)

    raw_text = "".join(all_texts) if all_texts else ""

    # Apply license plate corrections
    if apply_corrections:
        corrected_text = postprocess_plate_text(raw_text)
    else:
        corrected_text = raw_text

    return corrected_text, raw_text, all_texts, ocr_results


@app.command()
def analyze(
    image_path: Path = typer.Argument(..., help="Path to a single CCPD image to analyze"),
    save_output: bool = typer.Option(True, help="Save the visualization to runs/easyocr/"),
) -> None:
    """Analyze a single image with EasyOCR using multiple strategies.

    Tries multiple approaches (full image, crop, upscaled crop, different decoders)
    and shows all results to find the best one.

    Example:
        uv run python -m ml_ops.eval_easyocr analyze path/to/image.jpg
    """
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        raise typer.Exit(1)

    print(f"Analyzing: {image_path}")
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(["ch_sim", "en"], gpu=False)

    try:
        annotation = parse_ccpd_filename(image_path.name)
        gt_text = annotation["plate_text"]
        bbox = annotation["bbox"]
    except (ValueError, IndexError) as e:
        print(f"Error: Could not parse CCPD filename: {e}")
        print("This tool requires images with CCPD-style filenames containing annotations.")
        raise typer.Exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        raise typer.Exit(1)

    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    plate_crop = image[y_min:y_max, x_min:x_max]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB) if plate_crop.size > 0 else None

    # Prepare different image versions
    crop_enhanced = preprocess_plate_image(plate_crop)

    # Upscale crop 3x for better detection
    crop_upscaled = cv2.resize(plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    crop_upscaled_enhanced = preprocess_plate_image(crop_upscaled)

    print(f"\nImage size: {w} x {h}")
    print(f"Plate crop: {plate_crop.shape[1]} x {plate_crop.shape[0]}")
    print(f"Upscaled crop: {crop_upscaled.shape[1]} x {crop_upscaled.shape[0]}")
    print(f"\nGround Truth: {gt_text}")
    print("\n" + "=" * 70)
    print("TRYING MULTIPLE STRATEGIES...")
    print("=" * 70)

    # Try multiple strategies
    strategies = []

    # Strategy 1: Full image, no allowlist (like web demo)
    print("\n[1] Full image, no restrictions...")
    pred, raw, texts, _ = run_ocr_strategy(reader, image, "Full image", use_allowlist=False)
    char_acc = sum(1 for a, b in zip(pred, gt_text) if a == b) / len(gt_text) if gt_text else 0
    strategies.append(("Full image (no filter)", pred, raw, texts, char_acc, image_rgb))
    print(f"    Raw: {raw} → Corrected: {pred} | Char accuracy: {char_acc*100:.1f}%")

    # Strategy 2: Full image with allowlist
    print("\n[2] Full image, with plate allowlist...")
    pred, raw, texts, _ = run_ocr_strategy(reader, image, "Full image + allowlist", use_allowlist=True)
    char_acc = sum(1 for a, b in zip(pred, gt_text) if a == b) / len(gt_text) if gt_text else 0
    strategies.append(("Full image (allowlist)", pred, raw, texts, char_acc, image_rgb))
    print(f"    Raw: {raw} → Corrected: {pred} | Char accuracy: {char_acc*100:.1f}%")

    # Strategy 3: Crop, no preprocessing
    print("\n[3] Original crop, no preprocessing...")
    pred, raw, texts, _ = run_ocr_strategy(reader, plate_crop, "Crop raw", use_allowlist=False)
    char_acc = sum(1 for a, b in zip(pred, gt_text) if a == b) / len(gt_text) if gt_text else 0
    strategies.append(("Crop (raw)", pred, raw, texts, char_acc, crop_rgb))
    print(f"    Raw: {raw} → Corrected: {pred} | Char accuracy: {char_acc*100:.1f}%")

    # Strategy 4: Crop with CLAHE
    print("\n[4] Crop with CLAHE enhancement...")
    pred, raw, texts, _ = run_ocr_strategy(reader, crop_enhanced, "Crop CLAHE", use_allowlist=False)
    char_acc = sum(1 for a, b in zip(pred, gt_text) if a == b) / len(gt_text) if gt_text else 0
    strategies.append(("Crop (CLAHE)", pred, raw, texts, char_acc, cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2RGB)))
    print(f"    Raw: {raw} → Corrected: {pred} | Char accuracy: {char_acc*100:.1f}%")

    # Strategy 5: Upscaled crop
    print("\n[5] Upscaled crop (3x)...")
    pred, raw, texts, _ = run_ocr_strategy(reader, crop_upscaled, "Crop 3x", use_allowlist=False)
    char_acc = sum(1 for a, b in zip(pred, gt_text) if a == b) / len(gt_text) if gt_text else 0
    strategies.append(
        ("Crop (3x upscaled)", pred, raw, texts, char_acc, cv2.cvtColor(crop_upscaled, cv2.COLOR_BGR2RGB))
    )
    print(f"    Raw: {raw} → Corrected: {pred} | Char accuracy: {char_acc*100:.1f}%")

    # Strategy 6: Upscaled + CLAHE
    print("\n[6] Upscaled crop (3x) + CLAHE...")
    pred, raw, texts, _ = run_ocr_strategy(reader, crop_upscaled_enhanced, "Crop 3x CLAHE", use_allowlist=False)
    char_acc = sum(1 for a, b in zip(pred, gt_text) if a == b) / len(gt_text) if gt_text else 0
    strategies.append(
        ("Crop (3x + CLAHE)", pred, raw, texts, char_acc, cv2.cvtColor(crop_upscaled_enhanced, cv2.COLOR_BGR2RGB))
    )
    print(f"    Raw: {raw} → Corrected: {pred} | Char accuracy: {char_acc*100:.1f}%")

    # Strategy 7: Beamsearch on full image
    print("\n[7] Full image with beam search decoder...")
    pred, raw, texts, _ = run_ocr_strategy(reader, image, "Full + beam", use_allowlist=False, decoder="beamsearch")
    char_acc = sum(1 for a, b in zip(pred, gt_text) if a == b) / len(gt_text) if gt_text else 0
    strategies.append(("Full (beamsearch)", pred, raw, texts, char_acc, image_rgb))
    print(f"    Raw: {raw} → Corrected: {pred} | Char accuracy: {char_acc*100:.1f}%")

    # Find best strategy (char_acc is now at index 4)
    best_idx = max(range(len(strategies)), key=lambda i: strategies[i][4])
    best_strategy = strategies[best_idx]

    print("\n" + "=" * 70)
    print(f"BEST RESULT: {best_strategy[0]}")
    print(f"Raw OCR:     {best_strategy[2]}")
    print(f"Corrected:   {best_strategy[1]}")
    print(f"Ground Truth: {gt_text}")
    print(f"Exact Match: {'✓ YES' if best_strategy[1] == gt_text else '✗ NO'}")
    print(f"Char Accuracy: {best_strategy[4]*100:.1f}%")
    print("=" * 70)

    # Create visualization with all strategies
    fig = plt.figure(figsize=(20, 12))

    # Top row: Original images
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.imshow(image_rgb)
    ax1.add_patch(
        plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor="lime", linewidth=2)
    )
    ax1.set_title(f"Full Image ({w}x{h})", fontsize=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(3, 4, 2)
    if crop_rgb is not None:
        ax2.imshow(crop_rgb)
    ax2.set_title(f"Original Crop ({plate_crop.shape[1]}x{plate_crop.shape[0]})", fontsize=10)
    ax2.axis("off")

    ax3 = fig.add_subplot(3, 4, 3)
    ax3.imshow(cv2.cvtColor(crop_upscaled, cv2.COLOR_BGR2RGB))
    ax3.set_title(f"3x Upscaled ({crop_upscaled.shape[1]}x{crop_upscaled.shape[0]})", fontsize=10)
    ax3.axis("off")

    ax4 = fig.add_subplot(3, 4, 4)
    ax4.imshow(cv2.cvtColor(crop_upscaled_enhanced, cv2.COLOR_BGR2RGB))
    ax4.set_title("3x Upscaled + CLAHE", fontsize=10)
    ax4.axis("off")

    # Bottom rows: Results from each strategy
    # Tuple structure: (name, pred, raw, texts, char_acc, image)
    for i, (name, pred, raw, texts, char_acc, _) in enumerate(strategies):
        ax = fig.add_subplot(3, 4, 5 + i)
        exact = pred == gt_text
        color = "green" if exact else ("orange" if char_acc > 0.5 else "red")
        # Show both raw and corrected if different
        if raw != pred:
            ax.text(
                0.5, 0.7, f"Raw: {raw}", fontsize=10, ha="center", va="center", fontfamily="monospace", color="gray"
            )
            ax.text(0.5, 0.5, f"→ {pred}", fontsize=14, ha="center", va="center", fontfamily="monospace", color=color)
        else:
            ax.text(0.5, 0.6, f"{pred}", fontsize=14, ha="center", va="center", fontfamily="monospace", color=color)
        ax.text(0.5, 0.25, f"Acc: {char_acc*100:.0f}%", fontsize=12, ha="center", va="center")
        if i == best_idx:
            ax.set_facecolor("#e6ffe6")
            name = f"★ {name}"
        ax.set_title(name, fontsize=9, fontweight="bold" if i == best_idx else "normal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    plt.suptitle(f"EasyOCR Analysis: GT = {gt_text}\n{image_path.name}", fontsize=14)
    plt.tight_layout()

    if save_output:
        output_dir = RUNS_DIR / "easyocr"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"analysis_{image_path.stem}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nVisualization saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    app()
