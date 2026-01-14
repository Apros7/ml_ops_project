"""FastAPI endpoint for license plate recognition."""

import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

app = FastAPI(
    title="License Plate Recognition API",
    description="Detect and recognize license plates in images",
    version="1.0.0",
)

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
      <head>
        <title>License Plate Recognition</title>
      </head>
      <body>
        <h1>License Plate Recognition</h1>
        <h2>Recognize (detect + OCR)</h2>
        <form action="/recognize" enctype="multipart/form-data" method="post">
          <label>Image:</label>
          <input name="file" type="file" accept="image/*" required />
          <br/>
          <label>Confidence threshold:</label>
          <input name="conf_threshold" type="number" step="0.01" value="0.25" />
          <br/><br/>
          <button type="submit">Upload & Recognize</button>
        </form>

        <hr/>

        <h2>Detect only (no OCR)</h2>
        <form action="/detect" enctype="multipart/form-data" method="post">
          <label>Image:</label>
          <input name="file" type="file" accept="image/*" required />
          <br/>
          <label>Confidence threshold:</label>
          <input name="conf_threshold" type="number" step="0.01" value="0.25" />
          <br/><br/>
          <button type="submit">Upload & Detect</button>
        </form>

        <p>For full interactive docs, see <a href="/docs">/docs</a>.</p>
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
        from ml_ops.model import LicensePlateRecognizer

        detector_weights = Path("models/detector/best.pt")
        ocr_checkpoint = Path("models/ocr/last.ckpt")

        if not detector_weights.exists():
            detector_weights = "yolov8n.pt"
        else:
            detector_weights = str(detector_weights)

        ocr_ckpt = str(ocr_checkpoint) if ocr_checkpoint.exists() else None

        recognizer = LicensePlateRecognizer(
            detector_weights=detector_weights,
            ocr_checkpoint=ocr_ckpt,
        )
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
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

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

        return RecognitionResponse(
            success=True,
            num_plates=len(plates),
            plates=plates,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")


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
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

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

        return JSONResponse(
            content={
                "success": True,
                "num_detections": len(detections),
                "detections": detections,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
