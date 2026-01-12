"""Tests for data loading and preprocessing."""

import pytest

from ml_ops.data import (
    parse_ccpd_filename,
    plate_text_to_indices,
    indices_to_plate_text,
    PROVINCES,
    ALPHABETS,
    ADS,
    NUM_CLASSES,
)


class TestCCPDParser:
    """Tests for CCPD filename parsing."""

    def test_parse_valid_filename(self):
        """Test parsing a valid CCPD filename."""
        filename = "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
        result = parse_ccpd_filename(filename)

        assert "bbox" in result
        assert "vertices" in result
        assert "plate_text" in result
        assert "plate_indices" in result

        assert result["bbox"] == [154, 383, 386, 473]
        assert len(result["vertices"]) == 4
        assert result["plate_indices"] == [0, 0, 22, 27, 27, 33, 16]

    def test_parse_invalid_filename(self):
        """Test that invalid filenames raise ValueError."""
        with pytest.raises(ValueError):
            parse_ccpd_filename("invalid_filename.jpg")

    def test_parse_extracts_plate_text(self):
        """Test that plate text is correctly extracted."""
        filename = "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
        result = parse_ccpd_filename(filename)

        plate_text = result["plate_text"]
        assert len(plate_text) == 7
        assert plate_text[0] in PROVINCES
        assert plate_text[1] in ALPHABETS


class TestCharacterEncoding:
    """Tests for character encoding/decoding."""

    def test_plate_text_to_indices(self):
        """Test converting plate text to indices."""
        plate_text = "皖A12345"
        indices = plate_text_to_indices(plate_text)

        assert len(indices) == 7
        assert all(isinstance(idx, int) for idx in indices)
        assert all(0 <= idx < NUM_CLASSES for idx in indices)

    def test_indices_to_plate_text(self):
        """Test converting indices back to plate text."""
        original_text = "皖A12345"
        indices = plate_text_to_indices(original_text)
        recovered_text = indices_to_plate_text(indices)

        assert recovered_text == original_text

    def test_roundtrip_encoding(self):
        """Test that encoding and decoding are inverse operations."""
        test_plates = ["京B88888", "沪C12D34", "粤A00001"]

        for plate in test_plates:
            indices = plate_text_to_indices(plate)
            recovered = indices_to_plate_text(indices)
            assert recovered == plate, f"Failed for plate: {plate}"


class TestCharacterSets:
    """Tests for character set definitions."""

    def test_provinces_count(self):
        """Test that provinces list has expected length."""
        assert len(PROVINCES) == 34

    def test_alphabets_count(self):
        """Test that alphabets list has expected length."""
        assert len(ALPHABETS) == 25

    def test_ads_count(self):
        """Test that ads list has expected length."""
        assert len(ADS) == 35

    def test_no_letter_i_or_o_in_alphabets(self):
        """Test that I is not in alphabets (commonly confused with 1)."""
        assert "I" not in ALPHABETS

    def test_num_classes(self):
        """Test total number of character classes."""
        assert NUM_CLASSES > 0
