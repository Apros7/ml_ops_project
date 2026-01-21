from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from ml_ops import data as data_mod
from ml_ops import model as model_mod


class _FakeConf:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def item(self) -> float:
        return self._value


class _FakeCls:
    def __init__(self, value: int) -> None:
        self._value = int(value)

    def item(self) -> int:
        return self._value


class _FakeCoord:
    def __init__(self, coords: list[float]) -> None:
        self._coords = coords

    def tolist(self) -> list[float]:
        return self._coords


class _FakeBox:
    def __init__(self, coords: list[float], confidence: float = 0.9, cls_id: int = 0) -> None:
        self.xyxy = [_FakeCoord(coords)]
        self.conf = [_FakeConf(confidence)]
        self.cls = [_FakeCls(cls_id)]


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


class _FakeYOLO:
    def __init__(self, _weights: str) -> None:
        self._next_results: list[_FakeResult] = []

    def set_results(self, results: list[_FakeResult]) -> None:
        self._next_results = results

    def __call__(self, _image: object, conf: float = 0.25) -> list[_FakeResult]:
        _ = conf
        return self._next_results


def _write_dummy_image(path: Path, *, height: int = 40, width: int = 60) -> None:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (10, 20, 30)  # BGR
    assert cv2.imwrite(str(path), img)


def test_plate_detector_predict_parses_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeYOLO("x")
    fake.set_results([_FakeResult([_FakeBox([0, 1, 10, 11], confidence=0.8, cls_id=0)])])

    def _fake_ctor(_weights: str) -> _FakeYOLO:
        return fake

    monkeypatch.setattr(model_mod, "YOLO", _fake_ctor)

    det = model_mod.PlateDetector(model_name="weights.pt")
    preds = det.predict("img.jpg", conf=0.25)

    assert preds == [{"bbox": [0, 1, 10, 11], "confidence": 0.8, "class": 0}]


def test_plateocr_decode_english_only_removes_blanks_and_repeats() -> None:
    ocr = model_mod.PlateOCR(img_height=32, img_width=64, hidden_size=16, num_layers=1, dropout=0.0, english_only=True)
    blank = data_mod.ENGLISH_BLANK_IDX
    num = data_mod.ENGLISH_NUM_CLASSES

    # Indices: blank, A, A, blank, B, B -> "AB"
    indices = [blank, 0, 0, blank, 1, 1]
    seq_len = len(indices)
    log_probs = torch.full((seq_len, 1, num + 1), -100.0)
    for t, idx in enumerate(indices):
        log_probs[t, 0, idx] = 0.0

    decoded = ocr.decode(log_probs)
    assert decoded == ["AB"]


def test_plateocr_edit_distance_basic() -> None:
    ocr = model_mod.PlateOCR(img_height=32, img_width=64, hidden_size=16, num_layers=1, dropout=0.0, english_only=True)
    assert ocr._calculate_edit_distance("ABC", "ABC") == 0
    assert ocr._calculate_edit_distance("", "A") == 1
    assert ocr._calculate_edit_distance("A", "") == 1
    assert ocr._calculate_edit_distance("ABC", "AXC") == 1


def test_license_plate_recognizer_recognize_clamps_bbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FakePlateOCR:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.hparams = {"img_height": 32, "img_width": 64}

        def to(self, _device: torch.device) -> "_FakePlateOCR":
            return self

        def eval(self) -> "_FakePlateOCR":
            return self

        def __call__(self, _x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(2, 1, 2)

        def decode(self, _log_probs: torch.Tensor) -> list[str]:
            return ["OK"]

    fake_yolo = _FakeYOLO("x")
    fake_yolo.set_results([_FakeResult([_FakeBox([-5, -5, 50, 50], confidence=0.9)])])

    monkeypatch.setattr(model_mod, "YOLO", lambda _w: fake_yolo)
    monkeypatch.setattr(model_mod, "PlateOCR", _FakePlateOCR)

    img_path = tmp_path / "img.jpg"
    _write_dummy_image(img_path, height=40, width=60)

    rec = model_mod.LicensePlateRecognizer(detector_weights="weights.pt", ocr_checkpoint=None, device="cpu")
    out = rec.recognize(str(img_path), conf_threshold=0.25)

    assert len(out) == 1
    assert out[0]["bbox"] == [0, 0, 50, 40]  # clamped to image bounds (w=60, h=40)
    assert out[0]["plate_text"] == "OK"
    assert out[0]["confidence"] == 0.9


def test_license_plate_recognizer_easyocr_returns_empty_when_image_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _FakeEasyOCR:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def predict(self, _img: object) -> str:
            return "X"

    fake_yolo = _FakeYOLO("x")
    fake_yolo.set_results([_FakeResult([_FakeBox([0, 0, 10, 10], confidence=0.9)])])

    monkeypatch.setattr(model_mod, "YOLO", lambda _w: fake_yolo)
    monkeypatch.setattr(model_mod, "EasyOCRFineTunedRecognizer", _FakeEasyOCR)

    # Ensure imread returns None for the given path
    missing = tmp_path / "missing.jpg"
    import cv2 as _cv2

    monkeypatch.setattr(_cv2, "imread", lambda *_args, **_kwargs: None)

    rec = model_mod.LicensePlateRecognizerEasyOCR(detector_weights="weights.pt", ocr_weights="w.pth", device="cpu")
    assert rec.recognize(str(missing), conf_threshold=0.25) == []


def test_easyocr_finetuned_preprocess_type_check_and_shape() -> None:
    rec = object.__new__(model_mod.EasyOCRFineTunedRecognizer)
    rec.img_height = 32
    rec.img_width = 64
    rec.device = torch.device("cpu")

    with pytest.raises(TypeError, match="numpy\\.ndarray"):
        rec._preprocess("not-an-array")  # type: ignore[arg-type]

    image_rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    x = rec._preprocess(image_rgb)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, 1, 32, 64)


def test_easyocr_finetuned_ctc_decode_removes_blanks_and_repeats() -> None:
    rec = object.__new__(model_mod.EasyOCRFineTunedRecognizer)
    rec.converter = type("_Conv", (), {"character": ["", "A", "B"]})()
    decoded = rec._ctc_decode(torch.tensor([[0, 1, 1, 0, 2, 2]]))
    assert decoded == ["AB"]
