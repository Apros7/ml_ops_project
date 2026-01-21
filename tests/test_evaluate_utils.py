from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ml_ops import evaluate as eval_mod


def _write_dummy_image(path: Path, *, height: int = 80, width: int = 120) -> None:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (10, 20, 30)  # BGR
    assert cv2.imwrite(str(path), img)


def _ccpd_filename(*, bbox: tuple[int, int, int, int] = (10, 20, 50, 60), plate: str = "皖A12345") -> str:
    x1, y1, x2, y2 = bbox
    vertices = f"{x1}&{y1}_{x2}&{y1}_{x2}&{y2}_{x1}&{y2}"
    # plate_indices are not used by evaluate.py directly; parse_ccpd_filename will derive plate_text from indices,
    # but for these tests we only rely on bbox. Use a valid-looking indices block.
    indices = "0_0_0_0_0_0_0"
    _ = plate
    return f"000-0_0-{x1}&{y1}_{x2}&{y2}-{vertices}-{indices}-0-0.jpg"


def test_calculate_iou_basic() -> None:
    assert eval_mod.calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert eval_mod.calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert eval_mod.calculate_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)


def test_calculate_char_accuracy_handles_empty_and_lengths() -> None:
    assert eval_mod.calculate_char_accuracy("", "") == 1.0
    assert eval_mod.calculate_char_accuracy("A", "") == 0.0
    assert eval_mod.calculate_char_accuracy("ABC", "ABCD") == 0.75


def test_evaluate_detector_counts_tp_fp_fn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Three images:
    # - img1: 2 detections, one matches (TP=1, FP += 1)
    # - img2: 1 detection, bad IoU (FN += 1, FP += 1)
    # - img3: no detections (FN += 1)
    img1 = tmp_path / _ccpd_filename(bbox=(10, 10, 30, 30))
    img2 = tmp_path / _ccpd_filename(bbox=(40, 40, 60, 60))
    img3 = tmp_path / _ccpd_filename(bbox=(5, 5, 15, 15))
    _write_dummy_image(img1)
    _write_dummy_image(img2)
    _write_dummy_image(img3)

    class _FakePlateDetector:
        def __init__(self, model_name: str) -> None:
            _ = model_name

        def predict(self, image_path: str, conf: float = 0.25) -> list[dict[str, Any]]:
            _ = conf
            name = Path(image_path).name
            if name == img1.name:
                # One perfect match + one extra FP
                return [{"bbox": [10, 10, 30, 30]}, {"bbox": [0, 0, 5, 5]}]
            if name == img2.name:
                # Bad IoU
                return [{"bbox": [0, 0, 5, 5]}]
            return []

    monkeypatch.setattr(eval_mod, "PlateDetector", _FakePlateDetector)

    results = eval_mod.evaluate_detector(
        data_dir=tmp_path,
        weights="nonexistent.pt",
        conf_threshold=0.25,
        iou_threshold=0.5,
        max_samples=None,
    )

    assert results["true_positives"] == 1
    assert results["false_positives"] == 2  # +1 (extra in img1) +1 (img2 bad)
    assert results["false_negatives"] == 2  # img2 + img3
    assert results["precision"] == pytest.approx(1 / 3)
    assert results["recall"] == pytest.approx(1 / 3)


def test_evaluate_ocr_uses_datamodule_and_decode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FakeOCR:
        def eval(self) -> "_FakeOCR":
            return self

        def to(self, _device: torch.device) -> "_FakeOCR":
            return self

        def __call__(self, images: torch.Tensor) -> torch.Tensor:
            # log_probs shape is irrelevant for fake decode; keep minimal
            _ = images
            return torch.zeros(2, 2, 2)

        def decode(self, _log_probs: torch.Tensor) -> list[str]:
            return ["AAA", "BBB"]

    monkeypatch.setattr(eval_mod.PlateOCR, "load_from_checkpoint", lambda _ckpt: _FakeOCR())

    class _FakeDM:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def setup(self, stage: str | None = None) -> None:
            _ = stage

        def test_dataloader(self) -> DataLoader:
            batch = {"images": torch.zeros(2, 3, 32, 64), "plate_texts": ["AAA", "CCC"]}
            return DataLoader([batch], batch_size=None)

    monkeypatch.setattr(eval_mod, "CCPDDataModule", _FakeDM)

    results = eval_mod.evaluate_ocr(
        data_dir=tmp_path, checkpoint="ckpt.ckpt", split_dir=None, batch_size=2, num_workers=0
    )
    assert results["total_samples"] == 2
    assert results["exact_matches"] == 1
    assert results["exact_accuracy"] == 0.5
    assert 0.0 <= results["char_accuracy"] <= 1.0


def test_evaluate_easyocr_requires_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing.pth"
    with pytest.raises(FileNotFoundError, match="EasyOCR weights not found"):
        eval_mod.evaluate_easyocr(data_dir=tmp_path, weights=str(missing), split_dir=None, batch_size=1, num_workers=0)


def test_evaluate_easyocr_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    weights = tmp_path / "ocr_best.pth"
    weights.write_bytes(b"x")

    class _FakeEasyOCR:
        def __init__(self, _weights: str, device: str = "auto") -> None:
            _ = device
            self.img_height = 32
            self.img_width = 64
            self.english_only = True

        def predict(self, _image: np.ndarray) -> str:
            return "ABCDEF"

    monkeypatch.setattr(eval_mod, "EasyOCRFineTunedRecognizer", _FakeEasyOCR)

    class _FakeDM:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def setup(self, stage: str | None = None) -> None:
            _ = stage

        def test_dataloader(self) -> DataLoader:
            batch = {"images": torch.zeros(2, 3, 32, 64), "plate_texts": ["ABCDEF", "ZZZZZZ"]}
            return DataLoader([batch], batch_size=None)

    monkeypatch.setattr(eval_mod, "CCPDDataModule", _FakeDM)

    results = eval_mod.evaluate_easyocr(
        data_dir=tmp_path, weights=str(weights), split_dir=None, batch_size=2, num_workers=0
    )
    assert results["total_samples"] == 2
    assert results["exact_matches"] == 1
    assert results["exact_accuracy"] == 0.5


def test_evaluate_pipeline_counts_detection_and_ocr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    img = tmp_path / _ccpd_filename(bbox=(10, 10, 30, 30))
    _write_dummy_image(img)

    class _FakeOCR:
        english_only = True

    class _FakeRecognizer:
        def __init__(self, detector_weights: str, ocr_weights: str, device: str = "auto") -> None:
            _ = detector_weights, ocr_weights, device
            self.ocr = _FakeOCR()

        def recognize(self, _image_path: str, conf_threshold: float = 0.25) -> list[dict[str, Any]]:
            _ = conf_threshold
            return [{"bbox": [10, 10, 30, 30], "confidence": 0.9, "plate_text": "A12345"}]

    monkeypatch.setattr(eval_mod, "LicensePlateRecognizerEasyOCR", _FakeRecognizer)
    monkeypatch.setattr(
        eval_mod,
        "parse_ccpd_filename",
        lambda _name: {"bbox": [10, 10, 30, 30], "plate_text": "皖A12345"},
    )

    results = eval_mod.evaluate_pipeline(
        data_dir=tmp_path,
        detector_weights="missing.pt",
        ocr_weights="ocr.pth",
        conf_threshold=0.25,
        max_samples=None,
    )

    assert results["total_samples"] == 1
    assert results["detection_correct"] == 1
    assert results["ocr_correct"] == 1
    assert results["full_pipeline_accuracy"] == 1.0
