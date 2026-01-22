## Evaluation

Evaluation is available via the Typer CLI in `ml_ops.evaluate`.

### Detector metrics (precision/recall/F1)

```bash
uv run python -m ml_ops.evaluate evaluate-detector data/ccpd_tiny/val --weights models/yolo_best.pt
```

### OCR metrics (exact match + character accuracy)

#### Fine-tuned EasyOCR (recommended)

```bash
uv run python -m ml_ops.evaluate evaluate-easyocr data/ccpd_tiny/val --weights models/ocr_best.pth
```

#### CRNN checkpoint (baseline)

```bash
uv run python -m ml_ops.evaluate evaluate-ocr data/ccpd_tiny/val --checkpoint /path/to/checkpoint.ckpt
```

### End-to-end pipeline accuracy (YOLO + EasyOCR)

```bash
uv run python -m ml_ops.evaluate evaluate-pipeline data/ccpd_tiny/val \
  --detector-weights models/yolo_best.pt \
  --ocr-weights models/ocr_best.pth
```
