from pathlib import Path

import pytest
import torch

from ml_ops.data import CCPDDetectionDataset, CCPDOCRDataset, ocr_collate_fn


DATA_DIR = Path("data/ccpd_tiny")
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")


def count_images(path: Path) -> int:
    return sum(len(list(path.rglob(ext))) for ext in IMAGE_EXTS)


DATA_AVAILABLE = DATA_DIR.exists() and count_images(DATA_DIR) > 0


@pytest.mark.skipif(not DATA_AVAILABLE, reason="CCPD tiny data not available")
def test_ccpd_dataset() -> None:
    dataset = CCPDDetectionDataset(data_dir=DATA_DIR)
    expected_count = count_images(DATA_DIR)

    assert len(dataset) == expected_count
    assert len(dataset) > 0

    sample = dataset[0]
    image = sample["image"]
    target = sample["target"]

    assert image.ndim == 3 and image.shape[0] == 3
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert target.shape == (5,)
    assert target.dtype == torch.float32
    assert target[0].item() == 0
    assert torch.all(target[1:] >= 0) and torch.all(target[1:] <= 1)


@pytest.mark.skipif(not DATA_AVAILABLE, reason="CCPD tiny data not available")
def test_ocr_dataset() -> None:
    dataset = CCPDOCRDataset(data_dir=DATA_DIR, img_height=32, img_width=200)

    assert len(dataset) > 0

    sample = dataset[0]
    image = sample["image"]
    label = sample["label"]
    label_length = sample["label_length"]
    plate_text = sample["plate_text"]

    assert image.shape == (3, 32, 200)
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert label.dtype == torch.long
    assert label_length.item() == len(label)
    assert len(plate_text) == label_length.item()


@pytest.mark.skipif(not DATA_AVAILABLE, reason="CCPD tiny data not available")
def test_ocr_collate_fn_pads_labels() -> None:
    dataset = CCPDOCRDataset(data_dir=DATA_DIR, img_height=32, img_width=200, max_images=5)

    if len(dataset) < 2:
        pytest.skip("Not enough OCR samples to test collate.")

    batch = [dataset[0], dataset[1]]
    collated = ocr_collate_fn(batch)

    assert collated["images"].shape[0] == len(batch)
    assert collated["labels"].shape[0] == len(batch)

    expected_max_len = max(item["label"].shape[0] for item in batch)
    assert collated["labels"].shape[1] == expected_max_len
    assert torch.equal(collated["label_lengths"], torch.stack([item["label_length"] for item in batch]))
