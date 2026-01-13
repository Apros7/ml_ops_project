import torch

from ml_ops.data import ENGLISH_NUM_CLASSES, NUM_CLASSES
from ml_ops.model import CRNN, PlateOCR


def test_crnn_forward_shape_default() -> None:
    batch_size = 2
    img_height, img_width = 32, 200
    model = CRNN(img_height=img_height, img_width=img_width)

    x = torch.randn(batch_size, 3, img_height, img_width)
    y = model(x)  # seq_len, batch, num_classes+1

    assert y.shape[1] == batch_size
    assert y.shape[2] == int(NUM_CLASSES + 1)


def test_crnn_forward_shape_english_only() -> None:
    batch_size = 2
    img_height, img_width = 32, 200
    model = CRNN(img_height=img_height, img_width=img_width, english_only=True)

    x = torch.randn(batch_size, 3, img_height, img_width)
    y = model(x)

    assert y.shape[1] == batch_size
    assert y.shape[2] == int(ENGLISH_NUM_CLASSES + 1)


def test_plateocr_forward_matches_expected_shape() -> None:
    batch_size = 2
    img_height, img_width = 32, 200
    model = PlateOCR(img_height=img_height, img_width=img_width, english_only=True)

    x = torch.randn(batch_size, 3, img_height, img_width)
    y = model(x)

    assert y.shape[1] == batch_size
    assert y.shape[2] == int(ENGLISH_NUM_CLASSES + 1)


if __name__ == "__main__":
    test_crnn_forward_shape_default()
    test_crnn_forward_shape_english_only()
    test_plateocr_forward_matches_expected_shape()
