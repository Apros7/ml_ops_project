"""Locust load testing file for license plate recognition API."""

import io
import random
from pathlib import Path

from locust import HttpUser, between, task
from PIL import Image


class LicensePlateAPIUser(HttpUser):
    """Locust user class for load testing the license plate API."""

    wait_time = between(1, 3)

    def on_start(self) -> None:
        """Load test images on user start."""
        self.test_images: list[bytes] = []
        
        data_dir = Path("data/ccpd_tiny/val")
        if data_dir.exists():
            image_files = list(data_dir.glob("*.jpg"))[:10]
            if image_files:
                self.test_images = [img_path.read_bytes() for img_path in image_files]
        
        if not self.test_images:
            for _ in range(10):
                img = Image.new("RGB", (640, 480), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="JPEG")
                self.test_images.append(img_bytes.getvalue())

    @task(3)
    def recognize_plate(self) -> None:
        """POST random image to /recognize endpoint."""
        if not self.test_images:
            return
        
        image_bytes = random.choice(self.test_images)
        files = {"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")}
        data = {"conf_threshold": 0.25}
        
        with self.client.post("/recognize", files=files, data=data, catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    response.success()
                else:
                    response.failure(f"Recognition failed: {result}")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(1)
    def detect_only(self) -> None:
        """POST random image to /detect endpoint."""
        if not self.test_images:
            return
        
        image_bytes = random.choice(self.test_images)
        files = {"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")}
        data = {"conf_threshold": 0.25}
        
        with self.client.post("/detect", files=files, data=data, catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    response.success()
                else:
                    response.failure(f"Detection failed: {result}")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(1)
    def health_check(self) -> None:
        """GET /health endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "status" in data:
                    response.success()
                else:
                    response.failure("Missing status field")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
