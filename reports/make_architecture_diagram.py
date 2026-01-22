"""Generate an architectural diagram for the project's MLOps pipeline.

This script renders a simple, publication-friendly PNG diagram (no external design tools
required) and writes it to both:
  - reports/figures/mlops_architecture.png (for the exam report)
  - docs/source/images/mlops_architecture.png (for MkDocs)

Run:
    uv run python reports/make_architecture_diagram.py

If you don't have the project environment available, `python3` also works as long as
`Pillow` is installed:
    python3 reports/make_architecture_diagram.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DOCS_IMAGES_DIR = PROJECT_ROOT / "docs" / "source" / "images"


@dataclass(frozen=True)
class Box:
    """A labeled box in the diagram."""

    x: int
    y: int
    w: int
    h: int
    title: str
    subtitle: str | None = None
    fill: tuple[int, int, int] = (255, 255, 255)
    outline: tuple[int, int, int] = (40, 40, 40)

    def center(self) -> tuple[int, int]:
        """Return center point."""
        return (self.x + self.w // 2, self.y + self.h // 2)

    def right(self) -> tuple[int, int]:
        """Return midpoint of the right edge."""
        return (self.x + self.w, self.y + self.h // 2)

    def left(self) -> tuple[int, int]:
        """Return midpoint of the left edge."""
        return (self.x, self.y + self.h // 2)

    def top(self) -> tuple[int, int]:
        """Return midpoint of the top edge."""
        return (self.x + self.w // 2, self.y)

    def bottom(self) -> tuple[int, int]:
        """Return midpoint of the bottom edge."""
        return (self.x + self.w // 2, self.y + self.h)


def _try_load_font(size: int, *, bold: bool) -> ImageFont.ImageFont:
    """Load a readable font with fallbacks.

    Args:
        size: Font size in pixels.
        bold: Whether to prefer a bold face.

    Returns:
        A PIL font instance.
    """

    candidates: list[Path] = []
    pil_dir = Path(ImageFont.__file__).resolve().parent
    candidates.append(pil_dir / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"))
    candidates.extend(
        [
            Path(
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
                if bold
                else "/System/Library/Fonts/Supplemental/Arial.ttf"
            ),
            Path("/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf"),
        ]
    )

    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _wrap_lines(text: str, *, max_chars: int) -> list[str]:
    """Wrap text into multiple lines with a simple character budget."""

    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join([*current, word])
        if len(trial) > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def draw_box(
    draw: ImageDraw.ImageDraw, box: Box, *, title_font: ImageFont.ImageFont, body_font: ImageFont.ImageFont
) -> None:
    """Draw a rounded box with centered text."""

    radius = 16
    draw.rounded_rectangle(
        (box.x, box.y, box.x + box.w, box.y + box.h),
        radius=radius,
        fill=box.fill,
        outline=box.outline,
        width=3,
    )

    title_lines = _wrap_lines(box.title, max_chars=26)
    subtitle_lines = _wrap_lines(box.subtitle, max_chars=34) if box.subtitle else []

    title_text = "\n".join(title_lines)
    subtitle_text = "\n".join(subtitle_lines)

    title_bbox = draw.multiline_textbbox((0, 0), title_text, font=title_font, align="center", spacing=4)
    subtitle_bbox = (
        draw.multiline_textbbox((0, 0), subtitle_text, font=body_font, align="center", spacing=4)
        if subtitle_text
        else (0, 0, 0, 0)
    )

    title_h = title_bbox[3] - title_bbox[1]
    subtitle_h = subtitle_bbox[3] - subtitle_bbox[1]
    total_h = title_h + (subtitle_h + 10 if subtitle_text else 0)

    start_y = box.y + (box.h - total_h) // 2

    title_w = title_bbox[2] - title_bbox[0]
    title_x = box.x + (box.w - title_w) // 2
    draw.multiline_text((title_x, start_y), title_text, font=title_font, fill=(20, 20, 20), align="center", spacing=4)

    if subtitle_text:
        subtitle_w = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = box.x + (box.w - subtitle_w) // 2
        draw.multiline_text(
            (subtitle_x, start_y + title_h + 10),
            subtitle_text,
            font=body_font,
            fill=(55, 55, 55),
            align="center",
            spacing=4,
        )


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: tuple[int, int, int] = (60, 60, 60),
    width: int = 4,
    head_len: int = 16,
    head_w: int = 10,
    label: str | None = None,
    label_font: ImageFont.ImageFont | None = None,
) -> None:
    """Draw an arrow with an optional label.

    Args:
        draw: Pillow draw object.
        start: Start point (x, y).
        end: End point (x, y).
        color: Line color.
        width: Line width.
        head_len: Arrow head length.
        head_w: Arrow head width.
        label: Optional label text.
        label_font: Font used for label.
    """

    draw.line([start, end], fill=color, width=width)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    back_x = end[0] - head_len * math.cos(angle)
    back_y = end[1] - head_len * math.sin(angle)

    left_x = back_x + head_w * math.cos(angle + math.pi / 2)
    left_y = back_y + head_w * math.sin(angle + math.pi / 2)
    right_x = back_x + head_w * math.cos(angle - math.pi / 2)
    right_y = back_y + head_w * math.sin(angle - math.pi / 2)

    draw.polygon([(end[0], end[1]), (int(left_x), int(left_y)), (int(right_x), int(right_y))], fill=color)

    if label and label_font:
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad = 6
        x = mid[0] - text_w // 2
        y = mid[1] - text_h // 2 - 14
        draw.rounded_rectangle((x - pad, y - pad, x + text_w + pad, y + text_h + pad), radius=8, fill=(255, 255, 255))
        draw.text((x, y), label, font=label_font, fill=(40, 40, 40))


def _ensure_dirs(paths: Iterable[Path]) -> None:
    """Create output directories if missing."""

    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def render(output_path: Path) -> None:
    """Render the diagram PNG.

    Args:
        output_path: Output file path (PNG).
    """

    width, height = 2200, 1300
    img = Image.new("RGB", (width, height), (248, 249, 250))
    draw = ImageDraw.Draw(img)

    title_font = _try_load_font(52, bold=True)
    section_font = _try_load_font(26, bold=True)
    box_title_font = _try_load_font(24, bold=True)
    box_body_font = _try_load_font(18, bold=False)
    label_font = _try_load_font(16, bold=False)

    # Title
    title = "MLOps Architecture — License Plate Recognition"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (bbox[2] - bbox[0])) // 2, 40), title, font=title_font, fill=(25, 25, 25))

    # Section labels
    draw.text((90, 120), "Data & Versioning", font=section_font, fill=(60, 60, 60))
    draw.text((560, 120), "Code & CI/CD", font=section_font, fill=(60, 60, 60))
    draw.text((1060, 120), "Training & Tracking", font=section_font, fill=(60, 60, 60))
    draw.text((1560, 120), "Serving & Monitoring", font=section_font, fill=(60, 60, 60))

    # Colors
    data_fill = (227, 245, 233)
    cicd_fill = (227, 238, 252)
    train_fill = (255, 241, 224)
    serve_fill = (241, 232, 255)
    obs_fill = (235, 235, 235)

    # Boxes
    kaggle = Box(80, 170, 420, 120, "Public datasets", "Kaggle: CCPD + ALPR", fill=data_fill)
    dvc = Box(80, 330, 420, 120, "DVC-tracked data", "data/ + data.dvc", fill=data_fill)
    gcs = Box(80, 490, 420, 120, "GCS bucket (DVC remote)", "gs://ccpd_base", fill=data_fill)

    github = Box(560, 170, 420, 120, "GitHub repository", "src/ + configs/ + dockerfiles/", fill=cicd_fill)
    gha = Box(560, 330, 420, 120, "GitHub Actions (CI)", "pytest + ruff + dvc pull", fill=cicd_fill)
    cloud_build = Box(560, 490, 420, 120, "Cloud Build (CD)", "build + push containers", fill=cicd_fill)

    registry = Box(1040, 170, 460, 120, "Artifact Registry", "train/api/frontend images", fill=train_fill)
    vertex = Box(
        1040, 330, 460, 150, "Vertex AI / Cloud Run Job", "uv run dvc pull → ml_ops.train train-both", fill=train_fill
    )
    wandb = Box(1040, 520, 460, 120, "Weights & Biases", "metrics + artifacts", fill=train_fill)
    models = Box(1040, 680, 460, 120, "Model artifacts", "models/yolo_best.pt + models/ocr_best.pth", fill=train_fill)

    frontend = Box(1540, 170, 520, 120, "Streamlit frontend", "uploads image → calls backend", fill=serve_fill)
    api = Box(1540, 330, 520, 150, "FastAPI backend (Cloud Run)", "/detect + /recognize + /metrics", fill=serve_fill)
    prom = Box(1540, 520, 520, 120, "Prometheus-style metrics", "backend exposes /metrics", fill=obs_fill)
    users = Box(1540, 680, 520, 120, "Users", "browser UI + API clients", fill=serve_fill)

    boxes = [kaggle, dvc, gcs, github, gha, cloud_build, registry, vertex, wandb, models, frontend, api, prom, users]
    for b in boxes:
        draw_box(draw, b, title_font=box_title_font, body_font=box_body_font)

    # Arrows (data flow)
    draw_arrow(draw, kaggle.bottom(), dvc.top(), label="download/prepare", label_font=label_font)
    draw_arrow(draw, dvc.bottom(), gcs.top(), label="dvc push", label_font=label_font)

    # Arrows (CI/CD)
    draw_arrow(draw, github.bottom(), gha.top(), label="push/PR", label_font=label_font)
    draw_arrow(draw, github.bottom(), cloud_build.top(), label="release", label_font=label_font)
    draw_arrow(draw, cloud_build.right(), registry.left(), label="docker push", label_font=label_font)

    # Arrows (training)
    draw_arrow(draw, registry.bottom(), vertex.top(), label="train image", label_font=label_font)
    draw_arrow(draw, gcs.right(), vertex.left(), label="dvc pull", label_font=label_font)
    draw_arrow(draw, gha.left(), gcs.right(), label="dvc pull", label_font=label_font)
    draw_arrow(draw, vertex.right(), wandb.left(), label="log metrics", label_font=label_font)
    draw_arrow(draw, vertex.bottom(), models.top(), label="export weights", label_font=label_font)

    # Arrows (serving)
    draw_arrow(draw, models.right(), api.left(), label="load weights", label_font=label_font)
    draw_arrow(draw, frontend.bottom(), api.top(), label="HTTP", label_font=label_font)
    draw_arrow(draw, api.bottom(), prom.top(), label="/metrics", label_font=label_font)
    draw_arrow(draw, users.top(), frontend.bottom(), label="UI", label_font=label_font)
    draw_arrow(draw, users.top(), api.bottom(), label="API", label_font=label_font)

    # Footer note
    footer = "Source-of-truth config: Hydra (configs/). Reproducibility: DVC + locked dependencies (uv.lock)."
    f_bbox = draw.textbbox((0, 0), footer, font=label_font)
    draw.text(((width - (f_bbox[2] - f_bbox[0])) // 2, height - 60), footer, font=label_font, fill=(90, 90, 90))

    img.save(output_path, format="PNG", optimize=True)


def main() -> None:
    """CLI entrypoint."""

    _ensure_dirs([REPORT_FIGURES_DIR, DOCS_IMAGES_DIR])
    report_path = REPORT_FIGURES_DIR / "mlops_architecture.png"
    docs_path = DOCS_IMAGES_DIR / "mlops_architecture.png"

    render(report_path)
    render(docs_path)
    print(f"Wrote: {report_path}")
    print(f"Wrote: {docs_path}")


if __name__ == "__main__":
    main()
