import io
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from ml_ops.api import app


class _FakeConf:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeCoord:
    def __init__(self, coords: list[float]) -> None:
        self._coords = coords

    def tolist(self) -> list[float]:
        return self._coords


class _FakeBox:
    def __init__(self, coords: list[float], confidence: float) -> None:
        self.xyxy = [_FakeCoord(coords)]
        self.conf = [_FakeConf(confidence)]


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


class _FakeRecognizer:
    def __init__(self, recognize_fn: Callable[..., list[dict[str, Any]]]) -> None:
        self._recognize_fn = recognize_fn
        self.ocr = object()

    def recognize(self, image_path: str, conf_threshold: float = 0.25) -> list[dict[str, Any]]:
        return self._recognize_fn(image_path, conf_threshold)

    def detector(self, image_path: str, conf: float = 0.25) -> list[_FakeResult]:
        _ = image_path, conf
        return [_FakeResult([_FakeBox([0, 0, 10, 10], 0.9)])]


@pytest.fixture(autouse=True)
def reset_recognizer_cache() -> None:
    # Ensure a clean recognizer between tests
    from ml_ops import api

    api.recognizer = None


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    def make_fake_recognizer() -> _FakeRecognizer:
        return _FakeRecognizer(
            recognize_fn=lambda *_args, **_kwargs: [{"bbox": [1, 2, 3, 4], "confidence": 0.8, "plate_text": "ABC123"}]
        )

    monkeypatch.setattr("ml_ops.api.get_recognizer", make_fake_recognizer)
    return TestClient(app)


def _fake_image_bytes() -> bytes:
    # Minimal valid PNG header + IHDR chunk
    return bytes.fromhex("89504E470D0A1A0A0000000D4948445200000001000000010802000000907724")


def test_health_check(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["detector_loaded"] is True


def test_recognize_happy_path(client: TestClient) -> None:
    files = {"file": ("test.png", io.BytesIO(_fake_image_bytes()), "image/png")}
    response = client.post("/recognize", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["num_plates"] == 1
    assert data["plates"][0]["plate_text"] == "ABC123"


def test_recognize_rejects_non_image(client: TestClient) -> None:
    files = {"file": ("test.txt", io.BytesIO(b"not-an-image"), "text/plain")}
    response = client.post("/recognize", files=files)

    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]


def test_detect_happy_path(client: TestClient) -> None:
    files = {"file": ("test.png", io.BytesIO(_fake_image_bytes()), "image/png")}
    response = client.post("/detect", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["num_detections"] == 1
    assert len(data["detections"]) == 1


def test_detect_rejects_non_image(client: TestClient) -> None:
    files = {"file": ("test.txt", io.BytesIO(b"not-an-image"), "text/plain")}
    response = client.post("/detect", files=files)

    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]
