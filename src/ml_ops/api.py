"""FastAPI endpoint for license plate recognition."""

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

from ml_ops.logging_config import setup_logging

log_level = os.getenv("LOG_LEVEL", "INFO")
logger = setup_logging(log_level, "logs/api.log")

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
)

DETECTION_COUNT = Counter(
    "license_plates_detected_total",
    "Total number of license plates detected",
)

ERROR_COUNT = Counter(
    "http_errors_total",
    "Total number of HTTP errors",
    ["endpoint", "error_type"],
)

ACTIVE_REQUESTS = Gauge(
    "http_active_requests",
    "Number of active HTTP requests",
)

CPU_USAGE = Gauge(
    "system_cpu_percent",
    "CPU usage percentage",
)

MEMORY_USAGE = Gauge(
    "system_memory_percent",
    "Memory usage percentage",
)

MEMORY_USED = Gauge(
    "system_memory_used_bytes",
    "Memory used in bytes",
)

MEMORY_TOTAL = Gauge(
    "system_memory_total_bytes",
    "Total memory in bytes",
)

DISK_USAGE = Gauge(
    "system_disk_percent",
    "Disk usage percentage",
)

DISK_USED = Gauge(
    "system_disk_used_bytes",
    "Disk used in bytes",
)

DISK_TOTAL = Gauge(
    "system_disk_total_bytes",
    "Total disk space in bytes",
)


async def update_system_metrics() -> None:
    """Periodically update system metrics."""
    # First call to establish baseline for CPU (returns 0.0 on first call)
    psutil.cpu_percent(interval=None)
    await asyncio.sleep(1)  # Wait a bit before first real measurement

    while True:
        try:
            # Get actual CPU usage (non-blocking after baseline is set)
            cpu_percent = psutil.cpu_percent(interval=None)
            if cpu_percent == 0.0:
                # Fallback: use blocking call if still 0.0
                cpu_percent = psutil.cpu_percent(interval=0.1)
            CPU_USAGE.set(cpu_percent)

            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.percent)
            MEMORY_USED.set(memory.used)
            MEMORY_TOTAL.set(memory.total)

            disk = psutil.disk_usage("/")
            DISK_USAGE.set(disk.percent)
            DISK_USED.set(disk.used)
            DISK_TOTAL.set(disk.total)
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")

        await asyncio.sleep(0.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    logger.info("Starting API")
    
    background_task = asyncio.create_task(update_system_metrics())
    
    yield
    
    background_task.cancel()
    try:
        await background_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down")


app = FastAPI(
    title="License Plate Recognition API",
    description="Detect and recognize license plates in images",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/metrics", make_asgi_app())


@app.get("/", response_class=HTMLResponse)
async def index():
    """Redirect to Streamlit frontend or show API info page."""
    streamlit_url = os.getenv("STREAMLIT_URL", "http://localhost:8501")
    return f"""
    <html>
      <head>
        <title>License Plate Recognition API</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
          body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f8f9fa;
          }}
          .container {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
          }}
          h1 {{
            color: #1f77b4;
            margin-bottom: 10px;
          }}
          .button {{
            display: inline-block;
            background-color: #1f77b4;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 6px;
            margin: 10px 5px;
            transition: background-color 0.3s;
          }}
          .button:hover {{
            background-color: #1565c0;
          }}
          .button-secondary {{
            background-color: #2ecc71;
          }}
          .button-secondary:hover {{
            background-color: #27ae60;
          }}
          .info {{
            background-color: #e3f2fd;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            margin: 20px 0;
          }}
          code {{
            background-color: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <h1>🚗 License Plate Recognition API</h1>
          <p>FastAPI backend service for license plate detection and OCR.</p>
          
          <div class="info">
            <strong>📱 Modern UI Available:</strong><br>
            For a beautiful, interactive interface with visual feedback, 
            visit the Streamlit frontend or run it locally:
          </div>
          
          <a href="{streamlit_url}" class="button button-secondary" target="_blank">
            🎨 Open Streamlit Frontend
          </a>
          
          <a href="/docs" class="button">
            📚 API Documentation (Swagger)
          </a>
          
          <a href="/health" class="button">
            ❤️ Health Check
          </a>

          <hr style="margin: 30px 0;">

          <h2>API Endpoints</h2>
          <ul>
            <li><strong>POST /recognize</strong> - Detect and recognize license plates (full pipeline)</li>
            <li><strong>POST /detect</strong> - Detect license plates only (no OCR)</li>
            <li><strong>GET /health</strong> - Check API health and model status</li>
            <li><strong>GET /docs</strong> - Interactive API documentation</li>
          </ul>

          <h2>Quick Start</h2>
          <pre><code>curl -X POST "http://localhost:8000/recognize" \\
  -F "file=@image.jpg" \\
  -F "conf_threshold=0.25"</code></pre>

          <p style="color: #666; margin-top: 30px; font-size: 0.9em;">
            💡 Tip: Use the Streamlit frontend for the best user experience with 
            visual annotations and detailed results.
          </p>
        </div>
      </body>
    </html>
    """


recognizer = None


class PlateDetection(BaseModel):
    """Single plate detection result."""

    bbox: list[float]
    confidence: float
    plate_text: str


class RecognitionResponse(BaseModel):
    """Response model for recognition endpoint."""

    success: bool
    num_plates: int
    plates: list[PlateDetection]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    detector_loaded: bool
    ocr_loaded: bool


def get_recognizer():
    """Get or initialize the recognizer."""
    global recognizer
    if recognizer is None:
        logger.info("Loading models...")
        from ml_ops.model import LicensePlateRecognizer

        detector_weights = Path("models/yolo_best.pth")
        ocr_weights = Path("models/ocr_best.pth")

        detector_weights_str = str(detector_weights) if detector_weights.exists() else "yolov8n.pt"
        ocr_weights_str = str(ocr_weights) if ocr_weights.exists() else None

        recognizer = LicensePlateRecognizer(
            detector_weights=detector_weights_str,
            ocr_weights=ocr_weights_str,
        )
        logger.info("Models loaded")
    return recognizer


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and model status."""
    try:
        rec = get_recognizer()
        return HealthResponse(
            status="healthy",
            detector_loaded=rec.detector is not None,
            ocr_loaded=rec.ocr is not None,
        )
    except Exception as e:
        return HealthResponse(
            status=f"unhealthy: {str(e)}",
            detector_loaded=False,
            ocr_loaded=False,
        )


@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_plate(
    file: UploadFile = File(..., description="Image file to process"),
    conf_threshold: float = 0.25,
) -> RecognitionResponse:
    """Recognize license plates in an uploaded image.

    Args:
        file: Uploaded image file.
        conf_threshold: Detection confidence threshold.

    Returns:
        Recognition results with detected plates.
    """
    endpoint = "/recognize"
    REQUEST_COUNT.labels(endpoint=endpoint, status_code="pending").inc()
    ACTIVE_REQUESTS.inc()

    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            REQUEST_COUNT.labels(endpoint=endpoint, status_code="400").inc()
            ERROR_COUNT.labels(endpoint=endpoint, error_type="invalid_file_type").inc()
            raise HTTPException(status_code=400, detail="File must be an image")

        logger.info(f"Processing recognition request: {file.filename}")
        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        with REQUEST_LATENCY.labels(endpoint=endpoint).time():
            rec = get_recognizer()
            results = rec.recognize(tmp_path, conf_threshold=conf_threshold)

        Path(tmp_path).unlink()

        plates = [
            PlateDetection(
                bbox=r["bbox"],
                confidence=r["confidence"],
                plate_text=r["plate_text"],
            )
            for r in results
        ]

        num_plates = len(plates)
        logger.info(f"Found {num_plates} plates")

        DETECTION_COUNT.inc(num_plates)
        REQUEST_COUNT.labels(endpoint=endpoint, status_code="200").inc()

        return RecognitionResponse(
            success=True,
            num_plates=num_plates,
            plates=plates,
        )

    except HTTPException as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status_code=str(e.status_code)).inc()
        ERROR_COUNT.labels(endpoint=endpoint, error_type="http_exception").inc()
        raise
    except Exception as e:
        logger.error(f"Recognition error: {str(e)}", exc_info=True)
        REQUEST_COUNT.labels(endpoint=endpoint, status_code="500").inc()
        ERROR_COUNT.labels(endpoint=endpoint, error_type="exception").inc()
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/detect")
async def detect_plates(
    file: UploadFile = File(..., description="Image file to process"),
    conf_threshold: float = 0.25,
) -> JSONResponse:
    """Detect license plates without OCR.

    Args:
        file: Uploaded image file.
        conf_threshold: Detection confidence threshold.

    Returns:
        Detection results with bounding boxes only.
    """
    endpoint = "/detect"
    REQUEST_COUNT.labels(endpoint=endpoint, status_code="pending").inc()
    ACTIVE_REQUESTS.inc()

    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            REQUEST_COUNT.labels(endpoint=endpoint, status_code="400").inc()
            ERROR_COUNT.labels(endpoint=endpoint, error_type="invalid_file_type").inc()
            raise HTTPException(status_code=400, detail="File must be an image")

        logger.info(f"Processing detection request: {file.filename}")
        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        with REQUEST_LATENCY.labels(endpoint=endpoint).time():
            rec = get_recognizer()
            results = rec.detector(tmp_path, conf=conf_threshold)

        Path(tmp_path).unlink()

        detections = []
        for result in results:
            for box in result.boxes:
                detections.append(
                    {
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": box.conf[0].item(),
                    }
                )

        num_detections = len(detections)
        logger.info(f"Found {num_detections} detections")

        DETECTION_COUNT.inc(num_detections)
        REQUEST_COUNT.labels(endpoint=endpoint, status_code="200").inc()

        return JSONResponse(
            content={
                "success": True,
                "num_detections": num_detections,
                "detections": detections,
            }
        )

    except HTTPException as e:
        REQUEST_COUNT.labels(endpoint=endpoint, status_code=str(e.status_code)).inc()
        ERROR_COUNT.labels(endpoint=endpoint, error_type="http_exception").inc()
        raise
    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        REQUEST_COUNT.labels(endpoint=endpoint, status_code="500").inc()
        ERROR_COUNT.labels(endpoint=endpoint, error_type="exception").inc()
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        ACTIVE_REQUESTS.dec()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
