from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from ml_ops import data as data_mod


def _write_dummy_image(path: Path, *, height: int = 80, width: int = 120) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)  # BGR
    assert cv2.imwrite(str(path), image)


def _ccpd_filename(
    *,
    bbox: tuple[int, int, int, int] = (10, 20, 50, 60),
    plate_indices: tuple[int, int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0, 0),
) -> str:
    x1, y1, x2, y2 = bbox
    vertices = f"{x1}&{y1}_{x2}&{y1}_{x2}&{y2}_{x1}&{y2}"
    indices = "_".join(str(i) for i in plate_indices)
    return f"000-0_0-{x1}&{y1}_{x2}&{y2}-{vertices}-{indices}-0-0.jpg"


def test_english_plate_text_to_indices_handles_i_o_and_unknowns() -> None:
    indices = data_mod.english_plate_text_to_indices("皖IOA?BC")
    assert len(indices) == 6
    # I -> 1, O -> 0
    assert indices[0] == data_mod.ENGLISH_CHAR_TO_IDX["1"]
    assert indices[1] == data_mod.ENGLISH_CHAR_TO_IDX["0"]
    assert indices[2] == data_mod.ENGLISH_CHAR_TO_IDX["A"]
    # unknown -> 0 (fallback)
    assert indices[3] == 0


def test_english_indices_to_plate_text_skips_out_of_range() -> None:
    text = data_mod.english_indices_to_plate_text([0, 1, data_mod.ENGLISH_NUM_CLASSES + 5])
    assert text == "AB"


def test_parse_ccpd_filename_round_trip_plate_text_and_bbox() -> None:
    filename = _ccpd_filename(bbox=(10, 20, 50, 60), plate_indices=(0, 0, 1, 2, 3, 4, 5))
    ann = data_mod.parse_ccpd_filename(filename)
    assert ann["bbox"] == [10, 20, 50, 60]
    assert len(ann["vertices"]) == 4
    assert len(ann["plate_indices"]) == 7
    assert len(ann["plate_text"]) == 7


def test_parse_ccpd_filename_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Invalid CCPD filename format"):
        data_mod.parse_ccpd_filename("not-a-ccpd.jpg")


def test_plate_text_indices_round_trip() -> None:
    plate = f"{data_mod.PROVINCES[0]}A12345"
    idx = data_mod.plate_text_to_indices(plate)
    assert data_mod.indices_to_plate_text(idx) == plate


def test_indices_to_plate_text_skips_O_in_first_two_positions() -> None:
    # 'O' is a placeholder in the province/alpha sets and is intentionally skipped at positions 0 and 1.
    plate = "OO12345"
    idx = data_mod.plate_text_to_indices(plate)
    assert data_mod.indices_to_plate_text(idx).startswith("12345")


def test_ccpd_detection_dataset_from_tmp_dir(tmp_path: Path) -> None:
    img_path = tmp_path / _ccpd_filename(bbox=(10, 20, 50, 60))
    _write_dummy_image(img_path, height=80, width=120)

    dataset = data_mod.CCPDDetectionDataset(data_dir=tmp_path)
    assert len(dataset) == 1

    sample = dataset[0]
    image = sample["image"]
    target = sample["target"]
    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert target.shape == (5,)
    assert target.dtype == torch.float32
    assert target[0].item() == 0
    assert torch.all(target[1:] >= 0) and torch.all(target[1:] <= 1)


def test_ccpd_ocr_dataset_english_only_and_full(tmp_path: Path) -> None:
    img_path = tmp_path / _ccpd_filename(bbox=(10, 20, 50, 60), plate_indices=(0, 0, 0, 0, 0, 0, 0))
    _write_dummy_image(img_path, height=80, width=120)

    ds_full = data_mod.CCPDOCRDataset(data_dir=tmp_path, img_height=32, img_width=64, english_only=False)
    sample_full = ds_full[0]
    assert sample_full["image"].shape == (3, 32, 64)
    assert sample_full["label"].shape[0] == 7
    assert len(sample_full["plate_text"]) == 7

    ds_en = data_mod.CCPDOCRDataset(data_dir=tmp_path, img_height=32, img_width=64, english_only=True)
    sample_en = ds_en[0]
    assert sample_en["image"].shape == (3, 32, 64)
    assert sample_en["label"].shape[0] == 6
    assert len(sample_en["plate_text"]) == 6


def test_ocr_collate_fn_pads_synthetic_batch() -> None:
    batch = [
        {
            "image": torch.zeros(3, 32, 64),
            "label": torch.tensor([1, 2]),
            "label_length": torch.tensor(2),
            "plate_text": "AB",
        },
        {
            "image": torch.ones(3, 32, 64),
            "label": torch.tensor([3, 4, 5]),
            "label_length": torch.tensor(3),
            "plate_text": "CDE",
        },
    ]
    out = data_mod.ocr_collate_fn(batch)
    assert out["images"].shape == (2, 3, 32, 64)
    assert out["labels"].shape == (2, 3)
    assert torch.equal(out["label_lengths"], torch.tensor([2, 3]))
    assert out["plate_texts"] == ["AB", "CDE"]
    assert torch.equal(out["labels"][0], torch.tensor([1, 2, 0]))


def test_ccpd_data_module_setup_and_dataloader(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    val_dir.mkdir()
    test_dir.mkdir()

    for i in range(3):
        _write_dummy_image(train_dir / _ccpd_filename(bbox=(10, 20, 50, 60), plate_indices=(0, 0, i, 0, 0, 0, 0)))
    _write_dummy_image(val_dir / _ccpd_filename(bbox=(10, 20, 50, 60)))
    _write_dummy_image(test_dir / _ccpd_filename(bbox=(10, 20, 50, 60)))

    dm = data_mod.CCPDDataModule(
        data_dir=tmp_path,
        task="ocr",
        batch_size=2,
        num_workers=0,
        img_height=32,
        img_width=64,
        max_train_images=2,
        max_val_images=1,
        english_only=True,
    )
    dm.setup(stage=None)
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None
    assert len(dm.train_dataset) == 2
    assert len(dm.val_dataset) == 1
    assert len(dm.test_dataset) == 1

    batch = next(iter(dm.train_dataloader()))
    assert set(batch.keys()) == {"images", "labels", "label_lengths", "plate_texts"}


def test_export_yolo_format_creates_labels_and_images(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    fname = _ccpd_filename(bbox=(10, 20, 50, 60))
    img_path = input_dir / fname
    _write_dummy_image(img_path, height=80, width=120)

    output_dir = tmp_path / "yolo"
    data_mod.export_yolo_format(input_dir, output_dir, max_images=1, enable_profiling=False)

    copied = output_dir / "images" / fname
    label_file = output_dir / "labels" / f"{Path(fname).stem}.txt"
    assert copied.exists()
    assert label_file.exists()

    line = label_file.read_text().strip()
    cls, x_center, y_center, w, h = line.split()
    assert cls == "0"
    assert float(x_center) == pytest.approx(((10 + 50) / 2) / 120, rel=1e-6)
    assert float(y_center) == pytest.approx(((20 + 60) / 2) / 80, rel=1e-6)
    assert float(w) == pytest.approx((50 - 10) / 120, rel=1e-6)
    assert float(h) == pytest.approx((60 - 20) / 80, rel=1e-6)


def test_preprocess_calls_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called: list[tuple[Path, Path]] = []

    def _fake_export(data_path: Path, output_folder: Path, *_args: object, **_kwargs: object) -> None:
        called.append((data_path, output_folder))

    monkeypatch.setattr(data_mod, "export_yolo_format", _fake_export)
    data_mod.preprocess(tmp_path / "raw", tmp_path / "out")
    assert called == [(tmp_path / "raw", tmp_path / "out")]
