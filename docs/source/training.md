## Training

Training is exposed via a Typer CLI in `ml_ops.train` and wrapped by `invoke` tasks in `tasks.py`.

### Recommended: train detector + OCR

```bash
uv run invoke train-both --data-dir data/ccpd_tiny
```

Outputs:

- **YOLO runs**: `runs/detect/…`
- **OCR runs**: `runs/ocr/…`
- **Exported artifacts**:
  - `models/yolo_best.pt`
  - `models/ocr_best.pth`

### Train detector only (YOLOv8)

```bash
uv run invoke train-detector --data-dir data/ccpd_tiny
```

### Train OCR (fine-tuned EasyOCR)

This produces the fine-tuned EasyOCR weights used by the end-to-end evaluation pipeline and (by default) the API.

```bash
uv run invoke train-ocr --data-dir data/ccpd_tiny
```

### CRNN OCR (baseline / comparison)

The CRNN model is kept mainly for comparison and tests.

```bash
uv run invoke train-crnn --data-dir data/ccpd_tiny
```

### Distributed training (DDP)

Distributed training is supported for:

- **EasyOCR fine-tuning**: via `torchrun` + `training.ocr.distributed.enabled=true`
- **CRNN (Lightning)**: via Lightning strategies (e.g. `training.ocr.distributed.strategy=ddp`) when multiple GPUs are visible
- **YOLO detector (Ultralytics)**: set `training.detector.device` to multiple GPUs (e.g. `[0,1]`)

#### EasyOCR fine-tuning (torchrun)

```bash
uv run torchrun --standalone --nproc_per_node=2 -m ml_ops.train train-ocr data/ccpd_tiny \
  --override training.ocr.distributed.enabled=true
```

#### YOLO detector (multi-GPU)

```bash
uv run invoke train-detector --data-dir data/ccpd_tiny --override "training.detector.device=[0,1]"
```

#### CRNN OCR (Lightning DDP)

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run invoke train-crnn --data-dir data/ccpd_tiny \
  --override training.ocr.distributed.strategy=ddp
```

### EasyOCR quantization (optional)

Dynamic INT8 quantization can be enabled for the fine-tuned EasyOCR model (default: off). When enabled, the
`models/ocr_best.pth` checkpoint will default to quantized CPU inference in `EasyOCRFineTunedRecognizer`.

```bash
uv run invoke train-ocr --data-dir data/ccpd_tiny --override model.ocr.easyocr.quantize=true
```

### Hydra configs and overrides

Defaults are defined in `configs/config.yaml`. You can override presets without editing files:

```bash
uv run invoke train-detector --data-dir data/ccpd_tiny --override model/detector=yolov8s --override training/detector=fast
uv run invoke train-ocr --data-dir data/ccpd_tiny --override wandb_configs=disabled
```

### Weights & Biases logging

W&B is controlled via `configs/wandb_configs/`. To disable W&B locally:

```bash
uv run invoke train-both --data-dir data/ccpd_tiny --override wandb_configs=disabled
```
