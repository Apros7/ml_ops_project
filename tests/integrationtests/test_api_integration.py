"""Integration tests for the license plate recognition API."""

import io
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ml_ops.api import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    yield TestClient(app)


@pytest.fixture
def sample_image() -> bytes:
    """Load a sample test image from data directory."""
    data_dir = Path("data/ccpd_tiny/val")
    if data_dir.exists():
        image_files = list(data_dir.glob("*.jpg"))
        if image_files:
            return image_files[0].read_bytes()

    img = Image.new("RGB", (640, 480), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    return img_bytes.getvalue()


@pytest.fixture
def empty_image() -> bytes:
    """Create an empty/blank image for testing."""
    img = Image.new("RGB", (100, 100), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    return img_bytes.getvalue()


@pytest.fixture
def invalid_file() -> bytes:
    """Create invalid file content (not an image)."""
    return b"This is not an image file"


def test_health_endpoint(client: TestClient) -> None:
    """Test the /health endpoint returns 200 and correct structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "detector_loaded" in data
    assert "ocr_loaded" in data
    assert isinstance(data["status"], str)
    assert isinstance(data["detector_loaded"], bool)
    assert isinstance(data["ocr_loaded"], bool)


def test_metrics_endpoint(client: TestClient) -> None:
    """Test the /metrics endpoint returns 200 and contains Prometheus format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "http_requests_total" in content or "# HELP" in content
    assert "text/plain" in response.headers.get("content-type", "")


def test_recognize_valid_image(client: TestClient, sample_image: bytes) -> None:
    """Test POST image to /recognize endpoint, verify 200 and response structure."""
    files = {"file": ("test.jpg", io.BytesIO(sample_image), "image/jpeg")}
    data = {"conf_threshold": 0.25}
    response = client.post("/recognize", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    assert "success" in result
    assert "num_plates" in result
    assert "plates" in result
    assert isinstance(result["success"], bool)
    assert isinstance(result["num_plates"], int)
    assert isinstance(result["plates"], list)


def test_recognize_empty_image(client: TestClient, empty_image: bytes) -> None:
    """Test that blank image returns 0 detections."""
    files = {"file": ("empty.jpg", io.BytesIO(empty_image), "image/jpeg")}
    data = {"conf_threshold": 0.25}
    response = client.post("/recognize", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["num_plates"] == 0
    assert result["plates"] == []


def test_detect_endpoint(client: TestClient, sample_image: bytes) -> None:
    """Test /detect endpoint works."""
    files = {"file": ("test.jpg", io.BytesIO(sample_image), "image/jpeg")}
    data = {"conf_threshold": 0.25}
    response = client.post("/detect", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    assert "success" in result
    assert "num_detections" in result
    assert "detections" in result
    assert isinstance(result["success"], bool)
    assert isinstance(result["num_detections"], int)
    assert isinstance(result["detections"], list)


def test_confidence_threshold(client: TestClient, sample_image: bytes) -> None:
    """Test different threshold values (0.1, 0.5, 0.9)."""
    files = {"file": ("test.jpg", io.BytesIO(sample_image), "image/jpeg")}

    for threshold in [0.1, 0.5, 0.9]:
        data = {"conf_threshold": threshold}
        response = client.post("/recognize", files=files, data=data)
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert isinstance(result["num_plates"], int)


def test_invalid_file_type(client: TestClient, invalid_file: bytes) -> None:
    """Test that sending non-image file returns 400 error."""
    files = {"file": ("test.txt", io.BytesIO(invalid_file), "text/plain")}
    data = {"conf_threshold": 0.25}
    response = client.post("/recognize", files=files, data=data)

    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]


def test_missing_file(client: TestClient) -> None:
    """Test that sending request without file returns 422."""
    data = {"conf_threshold": 0.25}
    response = client.post("/recognize", data=data)

    assert response.status_code == 422


def test_concurrent_requests(client: TestClient, sample_image: bytes) -> None:
    """Test sending 10 requests simultaneously, all should succeed."""
    import concurrent.futures

    def make_request() -> tuple[int, dict]:
        files = {"file": ("test.jpg", io.BytesIO(sample_image), "image/jpeg")}
        data = {"conf_threshold": 0.25}
        response = client.post("/recognize", files=files, data=data)
        return response.status_code, response.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    assert len(results) == 10
    for status_code, result in results:
        assert status_code == 200
        assert result["success"] is True


def test_response_structure(client: TestClient, sample_image: bytes) -> None:
    """Test that each plate has bbox, confidence, plate_text fields."""
    files = {"file": ("test.jpg", io.BytesIO(sample_image), "image/jpeg")}
    data = {"conf_threshold": 0.25}
    response = client.post("/recognize", files=files, data=data)

    assert response.status_code == 200
    result = response.json()

    if result["num_plates"] > 0:
        for plate in result["plates"]:
            assert "bbox" in plate
            assert "confidence" in plate
            assert "plate_text" in plate
            assert isinstance(plate["bbox"], list)
            assert len(plate["bbox"]) == 4
            assert isinstance(plate["confidence"], (int, float))
            assert isinstance(plate["plate_text"], str)
