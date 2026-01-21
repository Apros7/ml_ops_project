# """Modern Streamlit frontend for License Plate Recognition System.

# This module provides a user-friendly web interface for the license plate
# recognition API with visual feedback, annotations, and comprehensive results display.

# Usage:
#     Local development:
#         streamlit run src/ml_ops/frontend.py --server.port 8501

#     Production (with Cloud Run backend):
#         export BACKEND_URL=https://license-plate-api-xxxx.run.app
#         streamlit run src/ml_ops/frontend.py

# Environment Variables:
#     BACKEND_URL: URL of the FastAPI backend service
#                  (default: http://localhost:8000 for local development)

# Example API Response Format:
#     Recognition:
#         {
#             "success": true,
#             "num_plates": 2,
#             "plates": [
#                 {
#                     "bbox": [100, 200, 300, 250],
#                     "confidence": 0.95,
#                     "plate_text": "ABC1234"
#                 }
#             ]
#         }
    
#     Detection:
#         {
#             "success": true,
#             "num_detections": 2,
#             "detections": [
#                 {
#                     "bbox": [100, 200, 300, 250],
#                     "confidence": 0.95
#                 }
#             ]
#         }
# """

# import io
# import logging
# import os
# import time
# from typing import Any

# import cv2
# import numpy as np
# import pandas as pd
# import requests
# import streamlit as st
# from PIL import Image

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# PRIMARY_COLOR = "#1f77b4"
# SUCCESS_COLOR = "#2ecc71"
# WARNING_COLOR = "#f39c12"
# ERROR_COLOR = "#e74c3c"
# BACKGROUND_COLOR = "#f8f9fa"

# st.set_page_config(
#     page_title="License Plate Recognition System",
#     page_icon="🚗",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# if "total_predictions" not in st.session_state:
#     st.session_state.total_predictions = 0
# if "successful_predictions" not in st.session_state:
#     st.session_state.successful_predictions = 0


# @st.cache_resource
# def get_backend_url() -> str:
#     """Get backend URL from environment or default to localhost.

#     Returns:
#         Backend URL string.
#     """
#     backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
#     logger.info(f"Using backend URL: {backend_url}")
#     return backend_url


# @st.cache_data(ttl=30)
# def check_backend_health(backend_url: str) -> dict[str, Any]:
#     """Check backend health status.

#     Args:
#         backend_url: Base URL of the backend API.

#     Returns:
#         Dictionary with health status and model info.
#     """
#     try:
#         response = requests.get(f"{backend_url}/health", timeout=5)
#         if response.status_code == 200:
#             data = response.json()
#             return {
#                 "healthy": data.get("status") == "healthy",
#                 "detector_loaded": data.get("detector_loaded", False),
#                 "ocr_loaded": data.get("ocr_loaded", False),
#                 "status": data.get("status", "unknown"),
#             }
#         return {"healthy": False, "detector_loaded": False, "ocr_loaded": False, "status": "unhealthy"}
#     except requests.RequestException as e:
#         logger.error(f"Health check failed: {e}")
#         return {"healthy": False, "detector_loaded": False, "ocr_loaded": False, "status": f"error: {str(e)}"}


# @st.cache_data(ttl=0.5)
# def get_system_metrics(backend_url: str) -> dict[str, float]:
#     """Fetch system metrics from Prometheus endpoint.

#     Args:
#         backend_url: Base URL of the backend API.

#     Returns:
#         Dictionary with CPU, memory, and disk usage percentages.
#     """
#     try:
#         response = requests.get(f"{backend_url}/metrics", timeout=2)
#         if response.status_code == 200:
#             metrics_text = response.text
#             metrics = {"cpu": 0.0, "memory": 0.0, "disk": 0.0}
            
#             for line in metrics_text.split("\n"):
#                 line = line.strip()
#                 if not line or line.startswith("#"):
#                     continue
                
#                 # Handle Prometheus format: metric_name{labels} value or metric_name value
#                 parts = line.split()
#                 if len(parts) >= 2:
#                     metric_name = parts[0].split("{")[0]  # Remove labels if present
#                     try:
#                         value = float(parts[-1])  # Last part is the value
                        
#                         if metric_name == "system_cpu_percent":
#                             metrics["cpu"] = value
#                         elif metric_name == "system_memory_percent":
#                             metrics["memory"] = value
#                         elif metric_name == "system_disk_percent":
#                             metrics["disk"] = value
#                     except (ValueError, IndexError):
#                         pass
            
#             return metrics
#     except requests.RequestException:
#         pass
#     return {"cpu": 0.0, "memory": 0.0, "disk": 0.0}


# def detect_plates(image_bytes: bytes, conf_threshold: float, backend_url: str) -> dict[str, Any]:
#     """Send detection request to backend.

#     Args:
#         image_bytes: Image file bytes.
#         conf_threshold: Confidence threshold for detection.
#         backend_url: Base URL of the backend API.

#     Returns:
#         Parsed JSON response from detection endpoint.

#     Raises:
#         requests.RequestException: If request fails.
#     """
#     files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
#     data = {"conf_threshold": conf_threshold}
#     response = requests.post(f"{backend_url}/detect", files=files, data=data, timeout=10)
#     response.raise_for_status()
#     return response.json()


# def recognize_plates(image_bytes: bytes, conf_threshold: float, backend_url: str) -> dict[str, Any]:
#     """Send recognition request to backend.

#     Args:
#         image_bytes: Image file bytes.
#         conf_threshold: Confidence threshold for detection.
#         backend_url: Base URL of the backend API.

#     Returns:
#         Parsed JSON response from recognition endpoint.

#     Raises:
#         requests.RequestException: If request fails.
#     """
#     files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
#     data = {"conf_threshold": conf_threshold}
#     response = requests.post(f"{backend_url}/recognize", files=files, data=data, timeout=15)
#     response.raise_for_status()
#     return response.json()


# def annotate_image(
#     image: Image.Image,
#     predictions: list[dict[str, Any]],
#     annotation_type: str = "recognition",
# ) -> Image.Image:
#     """Annotate image with bounding boxes and labels.

#     Args:
#         image: PIL Image to annotate.
#         predictions: List of prediction dictionaries with bbox, confidence, and optionally plate_text.
#         annotation_type: Type of annotation - "detection" (green boxes) or "recognition" (blue boxes with text).

#     Returns:
#         Annotated PIL Image.
#     """
#     img_array = np.array(image)
#     if len(img_array.shape) == 2:
#         img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
#     elif img_array.shape[2] == 4:
#         img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

#     annotated = img_array.copy()
#     h, w = annotated.shape[:2]

#     for pred in predictions:
#         bbox = pred["bbox"]
#         confidence = pred["confidence"]
#         x1, y1, x2, y2 = map(int, bbox)

#         x1 = max(0, min(x1, w - 1))
#         y1 = max(0, min(y1, h - 1))
#         x2 = max(x1 + 1, min(x2, w))
#         y2 = max(y1 + 1, min(y2, h))

#         if annotation_type == "detection":
#             color = (0, 255, 0)
#             thickness = 3
#             text = f"{confidence * 100:.1f}%"
#             text_y = y1 - 10 if y1 > 30 else y2 + 20
#         else:
#             if confidence > 0.9:
#                 color = (0, 255, 0)
#             elif confidence > 0.7:
#                 color = (0, 165, 255)
#             else:
#                 color = (0, 140, 255)

#             thickness = 3
#             plate_text = pred.get("plate_text", "N/A")
#             text = f"{plate_text}\n{confidence * 100:.1f}%"
#             text_y = y1 - 25 if y1 > 50 else y2 + 25

#         cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.8 if annotation_type == "recognition" else 0.5
#         font_thickness = 2 if annotation_type == "recognition" else 1

#         if annotation_type == "recognition":
#             lines = text.split("\n")
#             for i, line in enumerate(lines):
#                 (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
#                 y_pos = text_y + i * (text_height + baseline + 5)

#                 padding = 5
#                 cv2.rectangle(
#                     annotated,
#                     (x1, y_pos - text_height - padding),
#                     (x1 + text_width + 2 * padding, y_pos + baseline + padding),
#                     (0, 0, 0),
#                     -1,
#                 )

#                 cv2.putText(annotated, line, (x1 + padding, y_pos), font, font_scale, color, font_thickness)
#         else:
#             (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
#             padding = 5
#             cv2.rectangle(
#                 annotated,
#                 (x1, text_y - text_height - padding),
#                 (x1 + text_width + 2 * padding, text_y + baseline + padding),
#                 (0, 0, 0),
#                 -1,
#             )
#             cv2.putText(annotated, text, (x1 + padding, text_y), font, font_scale, color, font_thickness)

#     return Image.fromarray(annotated)


# def apply_custom_css() -> None:
#     """Apply custom CSS styling."""
#     st.markdown(
#         f"""
#         <style>
#         .main {{
#             background-color: {BACKGROUND_COLOR};
#         }}
#         .stButton>button {{
#             border-radius: 8px;
#             padding: 0.5rem 1rem;
#             font-weight: 500;
#             transition: all 0.3s;
#         }}
#         .stButton>button:hover {{
#             transform: translateY(-2px);
#             box-shadow: 0 4px 8px rgba(0,0,0,0.2);
#         }}
#         .metric-card {{
#             background-color: white;
#             padding: 1rem;
#             border-radius: 8px;
#             box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#             margin: 0.5rem 0;
#         }}
#         h1 {{
#             color: {PRIMARY_COLOR};
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )


# def main() -> None:
#     """Main Streamlit application."""
#     apply_custom_css()

#     backend_url = get_backend_url()
#     health = check_backend_health(backend_url)

#     st.title("🚗 License Plate Recognition System")
#     st.markdown("Upload an image to detect and recognize license plates using AI-powered computer vision.")

#     with st.sidebar:
#         st.header("⚙️ Settings")
#         st.markdown("---")

#         if health["healthy"]:
#             st.success("✅ Backend Connected")
#             if health["detector_loaded"]:
#                 st.info("🔍 Detector: Ready")
#             if health["ocr_loaded"]:
#                 st.info("📝 OCR: Ready")
#         else:
#             st.error("❌ Backend Unavailable")
#             st.warning(f"Status: {health['status']}")
#             if st.button("🔄 Retry Connection"):
#                 st.cache_data.clear()
#                 st.rerun()

#         st.markdown("---")
#         conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

#         st.markdown("---")
#         st.header("📊 Statistics")
#         st.metric("Total Predictions", st.session_state.total_predictions)
#         success_rate = (
#             (st.session_state.successful_predictions / st.session_state.total_predictions * 100)
#             if st.session_state.total_predictions > 0
#             else 0
#         )
#         st.metric("Success Rate", f"{success_rate:.1f}%")

#         st.markdown("---")
#         if health["healthy"]:
#             st.header("💻 System Load")
#             metrics = get_system_metrics(backend_url)
            
#             cpu_value = metrics.get("cpu", 0.0)
#             memory_value = metrics.get("memory", 0.0)
#             disk_value = metrics.get("disk", 0.0)
            
#             cpu_color = SUCCESS_COLOR if cpu_value < 70 else WARNING_COLOR if cpu_value < 90 else ERROR_COLOR
#             memory_color = SUCCESS_COLOR if memory_value < 70 else WARNING_COLOR if memory_value < 90 else ERROR_COLOR
#             disk_color = SUCCESS_COLOR if disk_value < 70 else WARNING_COLOR if disk_value < 90 else ERROR_COLOR
            
#             st.metric("CPU Usage", f"{cpu_value:.1f}%")
#             st.progress(cpu_value / 100)
            
#             st.metric("Memory Usage", f"{memory_value:.1f}%")
#             st.progress(memory_value / 100)
            
#             st.metric("Disk Usage", f"{disk_value:.1f}%")
#             st.progress(disk_value / 100)

#         st.markdown("---")
#         with st.expander("ℹ️ Instructions"):
#             st.markdown(
#                 """
#                 1. Upload an image containing license plates
#                 2. Choose between:
#                    - **Detect Only**: Find plates without OCR
#                    - **Detect & Recognize**: Full pipeline with text recognition
#                 3. View annotated results and download if needed
#                 4. Adjust confidence threshold to filter detections
#                 """
#             )

#     uploaded_file = st.file_uploader(
#         "Upload Image",
#         type=["jpg", "jpeg", "png"],
#         help="Upload an image file containing license plates",
#     )

#     if uploaded_file is not None:
#         uploaded_file.seek(0)
#         image_bytes = uploaded_file.read()
#         image = Image.open(io.BytesIO(image_bytes))

#         col1, col2 = st.columns(2)
#         with col1:
#             st.subheader("📷 Original Image")
#             st.image(image, use_container_width=True)
#             st.caption(f"Dimensions: {image.size[0]} × {image.size[1]} px")
#             st.caption(f"File size: {len(image_bytes) / 1024:.2f} KB")

#         if not health["healthy"]:
#             st.error("⚠️ Backend is not available. Please check the connection.")
#             return

#         col21, col22 = st.columns(2)

#         with col21:
#             detect_btn = st.button("🔍 Detect Plates Only", use_container_width=True, type="primary")

#         with col22:
#             recognize_btn = st.button("📝 Detect & Recognize Text", use_container_width=True, type="primary")

#         if detect_btn or recognize_btn:
#             if detect_btn:
#                 with st.spinner("🔍 Analyzing image... Detecting license plates..."):
#                     start_time = time.time()
#                     try:
#                         result = detect_plates(image_bytes, conf_threshold, backend_url)
#                         processing_time = time.time() - start_time

#                         st.session_state.total_predictions += 1
#                         if result.get("success") and result.get("num_detections", 0) > 0:
#                             st.session_state.successful_predictions += 1

#                         detections = result.get("detections", [])
#                         num_detections = result.get("num_detections", 0)

#                         col2.subheader("🎯 Detection Results")

#                         if num_detections == 0:
#                             st.warning("⚠️ No license plates detected. Try lowering the confidence threshold.")
#                         else:
#                             st.success(f"✅ Found {num_detections} license plate(s)")

#                             annotated_image = annotate_image(image, detections, annotation_type="detection")
#                             col2.image(annotated_image, use_container_width=True)

#                             avg_confidence = sum(d["confidence"] for d in detections) / len(detections)

#                             metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
#                             with metrics_col1:
#                                 st.metric("Plates Detected", num_detections)
#                             with metrics_col2:
#                                 st.metric("Avg Confidence", f"{avg_confidence * 100:.1f}%")
#                             with metrics_col3:
#                                 st.metric("Processing Time", f"{processing_time:.2f}s")

#                     except requests.RequestException as e:
#                         st.error(f"❌ Detection failed: {str(e)}")
#                         logger.error(f"Detection error: {e}")

#             elif recognize_btn:
#                 with st.spinner("📝 Analyzing image... Detecting and recognizing text..."):
#                     start_time = time.time()
#                     try:
#                         result = recognize_plates(image_bytes, conf_threshold, backend_url)
#                         processing_time = time.time() - start_time

#                         st.session_state.total_predictions += 1
#                         if result.get("success") and result.get("num_plates", 0) > 0:
#                             st.session_state.successful_predictions += 1

#                         plates = result.get("plates", [])
#                         num_plates = result.get("num_plates", 0)

#                         col2.subheader("🎯 Recognition Results")

#                         if num_plates == 0:
#                             st.warning("⚠️ No license plates detected. Try lowering the confidence threshold.")
#                         else:
#                             st.success(f"✅ Found {num_plates} license plate(s)")

#                             annotated_image = annotate_image(image, plates, annotation_type="recognition")
#                             col2.image(annotated_image, use_container_width=True)

#                             img_buffer = io.BytesIO()
#                             annotated_image.save(img_buffer, format="PNG")
#                             img_buffer.seek(0)

#                             st.download_button(
#                                 label="📥 Download Annotated Image",
#                                 data=img_buffer,
#                                 file_name=f"annotated_{uploaded_file.name}",
#                                 mime="image/png",
#                             )

#                             st.markdown("### 📋 Detailed Results")
#                             results_data = []
#                             for i, plate in enumerate(plates, 1):
#                                 results_data.append(
#                                     {
#                                         "Plate #": i,
#                                         "Recognized Text": plate.get("plate_text", "N/A"),
#                                         "Confidence": f"{plate['confidence'] * 100:.2f}%",
#                                         "Bounding Box": f"[{plate['bbox'][0]:.0f}, {plate['bbox'][1]:.0f}, {plate['bbox'][2]:.0f}, {plate['bbox'][3]:.0f}]",
#                                     }
#                                 )

#                             df = pd.DataFrame(results_data)
#                             st.dataframe(df, use_container_width=True, hide_index=True)

#                             avg_confidence = sum(p["confidence"] for p in plates) / len(plates)
#                             metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
#                             with metrics_col1:
#                                 st.metric("Plates Found", num_plates)
#                             with metrics_col2:
#                                 st.metric("Avg Confidence", f"{avg_confidence * 100:.1f}%")
#                             with metrics_col3:
#                                 st.metric("Processing Time", f"{processing_time:.2f}s")

#                     except requests.RequestException as e:
#                         st.error(f"❌ Recognition failed: {str(e)}")
#                         logger.error(f"Recognition error: {e}")


# if __name__ == "__main__":
#     if "last_refresh" not in st.session_state:
#         st.session_state.last_refresh = time.time()
    
#     # Auto-refresh every 0.5 seconds
#     current_time = time.time()
#     if current_time - st.session_state.last_refresh >= 0.5:
#         st.session_state.last_refresh = current_time
    
#     main()
    
#     # Schedule next refresh
#     time.sleep(0.1)  # Small delay to prevent excessive CPU usage
#     st.rerun()



"""
Modern Streamlit frontend for License Plate Recognition System.

This frontend ONLY sends images to a backend API.
It does NOT load models or run inference locally.

Environment Variables:
    API_URL: URL of the FastAPI backend service
             (default: http://localhost:8000)
"""

import io
import logging
import os
import time
from typing import Any

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------
PRIMARY_COLOR = "#1f77b4"
SUCCESS_COLOR = "#2ecc71"
WARNING_COLOR = "#f39c12"
ERROR_COLOR = "#e74c3c"
BACKGROUND_COLOR = "#f8f9fa"

# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="License Plate Recognition System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "total_predictions" not in st.session_state:
    st.session_state.total_predictions = 0

if "successful_predictions" not in st.session_state:
    st.session_state.successful_predictions = 0

# ---------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------
@st.cache_resource
def get_api_url() -> str:
    """Get API URL from environment or default to localhost."""
    api_url = os.getenv("API_URL", "http://localhost:8000")
    logger.info(f"Using API URL: {api_url}")
    return api_url


@st.cache_data(ttl=30)
def check_backend_health(api_url: str) -> dict[str, Any]:
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return {
                "healthy": data.get("status") == "healthy",
                "detector_loaded": data.get("detector_loaded", False),
                "ocr_loaded": data.get("ocr_loaded", False),
                "status": data.get("status", "unknown"),
            }
    except requests.RequestException as e:
        logger.error(e)

    return {
        "healthy": False,
        "detector_loaded": False,
        "ocr_loaded": False,
        "status": "unreachable",
    }


@st.cache_data(ttl=1)
def get_system_metrics(api_url: str) -> dict[str, float]:
    try:
        r = requests.get(f"{api_url}/metrics", timeout=2)
        if r.status_code == 200:
            metrics = {"cpu": 0.0, "memory": 0.0, "disk": 0.0}
            for line in r.text.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0].split("{")[0]
                    value = float(parts[-1])
                    if name == "system_cpu_percent":
                        metrics["cpu"] = value
                    elif name == "system_memory_percent":
                        metrics["memory"] = value
                    elif name == "system_disk_percent":
                        metrics["disk"] = value
            return metrics
    except Exception:
        pass

    return {"cpu": 0.0, "memory": 0.0, "disk": 0.0}


# ---------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------
def detect_plates(image_bytes: bytes, conf: float, api_url: str) -> dict[str, Any]:
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    data = {"conf_threshold": conf}
    r = requests.post(f"{api_url}/detect", files=files, data=data, timeout=15)
    r.raise_for_status()
    return r.json()


def recognize_plates(image_bytes: bytes, conf: float, api_url: str) -> dict[str, Any]:
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    data = {"conf_threshold": conf}
    r = requests.post(f"{api_url}/recognize", files=files, data=data, timeout=20)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------
# Image annotation
# ---------------------------------------------------------------------
def annotate_image(image: Image.Image, predictions: list[dict], mode: str) -> Image.Image:
    img = np.array(image.convert("RGB"))

    for p in predictions:
        x1, y1, x2, y2 = map(int, p["bbox"])
        conf = p["confidence"]

        if mode == "recognition":
            label = f"{p.get('plate_text', 'N/A')} ({conf:.2f})"
            color = (0, 255, 0)
        else:
            label = f"{conf:.2f}"
            color = (255, 0, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            img,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    return Image.fromarray(img)


# ---------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------
def apply_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        .main {{ background-color: {BACKGROUND_COLOR}; }}
        h1 {{ color: {PRIMARY_COLOR}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------
def main() -> None:
    apply_custom_css()

    api_url = get_api_url()
    health = check_backend_health(api_url)

    st.title("🚗 License Plate Recognition System")
    st.markdown("Upload an image to detect and recognize license plates.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        if health["healthy"]:
            st.success("Backend Connected")
            st.info(f"Detector: {'Ready' if health['detector_loaded'] else 'Not loaded'}")
            st.info(f"OCR: {'Ready' if health['ocr_loaded'] else 'Not loaded'}")
        else:
            st.error("Backend Unavailable")

        conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded and health["healthy"]:
        image_bytes = uploaded.read()
        image = Image.open(io.BytesIO(image_bytes))

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original Image", use_container_width=True)

        d_btn, r_btn = st.columns(2)

        if d_btn.button("🔍 Detect Plates"):
            with st.spinner("Detecting..."):
                res = detect_plates(image_bytes, conf, api_url)
                st.session_state.total_predictions += 1
                annotated = annotate_image(image, res.get("detections", []), "detect")
                col2.image(annotated, use_container_width=True)

        if r_btn.button("📝 Detect & Recognize"):
            with st.spinner("Recognizing..."):
                res = recognize_plates(image_bytes, conf, api_url)
                st.session_state.total_predictions += 1
                annotated = annotate_image(image, res.get("plates", []), "recognition")
                col2.image(annotated, use_container_width=True)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
