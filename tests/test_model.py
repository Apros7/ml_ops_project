"""Tests for model architectures."""

import pytest
import torch

from ml_ops.model import CRNN, PlateOCR
from ml_ops.data import NUM_CLASSES, BLANK_IDX


class TestCRNN:
    """Tests for CRNN architecture."""

    def test_crnn_output_shape(self):
        """Test that CRNN produces correct output shape."""
        model = CRNN(img_height=48, img_width=168, hidden_size=256)
        x = torch.randn(2, 3, 48, 168)
        output = model(x)

        assert output.dim() == 3
        assert output.size(1) == 2
        assert output.size(2) == NUM_CLASSES + 1

    def test_crnn_different_batch_sizes(self):
        """Test CRNN with different batch sizes."""
        model = CRNN()

        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 3, 48, 168)
            output = model(x)
            assert output.size(1) == batch_size

    def test_crnn_output_is_log_softmax(self):
        """Test that CRNN output is log softmax (sums to ~0 in log space)."""
        model = CRNN()
        x = torch.randn(2, 3, 48, 168)
        output = model(x)

        probs = torch.exp(output)
        sums = probs.sum(dim=2)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_crnn_custom_params(self):
        """Test CRNN with custom parameters."""
        model = CRNN(
            img_height=32,
            img_width=100,
            hidden_size=128,
            num_layers=1,
            dropout=0.2,
        )
        x = torch.randn(2, 3, 32, 100)
        output = model(x)

        assert output.size(1) == 2
        assert output.size(2) == NUM_CLASSES + 1


class TestPlateOCR:
    """Tests for PlateOCR Lightning module."""

    def test_plate_ocr_forward(self):
        """Test PlateOCR forward pass."""
        model = PlateOCR(img_height=48, img_width=168)
        x = torch.randn(2, 3, 48, 168)
        output = model(x)

        assert output.dim() == 3
        assert output.size(1) == 2

    def test_plate_ocr_decode(self):
        """Test CTC decoding."""
        model = PlateOCR()
        x = torch.randn(2, 3, 48, 168)
        log_probs = model(x)

        decoded = model.decode(log_probs)

        assert isinstance(decoded, list)
        assert len(decoded) == 2
        assert all(isinstance(d, str) for d in decoded)

    def test_plate_ocr_training_step(self):
        """Test training step returns loss."""
        model = PlateOCR()

        batch = {
            "images": torch.randn(4, 3, 48, 168),
            "labels": torch.randint(0, NUM_CLASSES, (4, 7)),
            "label_lengths": torch.tensor([7, 7, 7, 7]),
            "plate_texts": ["皖A12345", "京B88888", "沪C12D34", "粤A00001"],
        }

        loss = model.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() > 0


class TestBlankIndex:
    """Tests for CTC blank index configuration."""

    def test_blank_index_value(self):
        """Test that blank index is NUM_CLASSES."""
        assert BLANK_IDX == NUM_CLASSES

    def test_blank_index_not_in_char_range(self):
        """Test blank index is outside character range."""
        assert BLANK_IDX >= NUM_CLASSES
