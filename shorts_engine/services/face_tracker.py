"""
services/face_tracker.py — Active speaker tracking & dynamic 9:16 auto-framing with YOLO-face.

Responsibilities:
  1. Detect face coordinates frame-by-frame on 16:9 widescreen video inputs using YOLOv8n-face.
  2. Apply Exponential Moving Average (EMA) smoothing to active speaker centroids.
  3. Compute optimal crop bounds to keep the speaker centered when converting to 9:16.
  4. Gracefully fall back to standard center-crop if no face is detected or if vision libraries are absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default smoothing factor for Exponential Moving Average (0 < alpha <= 1)
# Lower values create smoother camera panning; higher values track fast movements quicker.
_DEFAULT_EMA_ALPHA = 0.15
_DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.25  # Sample 4 frames per second for high-speed tracking


def _clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))


def calculate_active_speaker_crop_x(
    video_path: Path,
    source_width: int,
    source_height: int,
    target_width: int = 1080,
    target_height: int = 1920,
    sample_interval: float = _DEFAULT_SAMPLE_INTERVAL_SECONDS,
    ema_alpha: float = _DEFAULT_EMA_ALPHA,
) -> int:
    """
    Detect the active speaker's face centroid and calculate the optimal X crop offset.

    When converting 16:9 widescreen to 9:16 vertical video, the video is scaled so height
    matches *target_height*, leaving excess width on the horizontal axis to crop.
    This function tracks the speaker's face across the clip and returns the smoothed
    horizontal pixel offset where cropping should occur.

    Args:
        video_path:      Path to the source video.
        source_width:    Original video width in pixels.
        source_height:   Original video height in pixels.
        target_width:    Output 9:16 width (default 1080).
        target_height:   Output 9:16 height (default 1920).
        sample_interval: Seconds between sampled frames.
        ema_alpha:       Exponential smoothing coefficient.

    Returns:
        Integer horizontal crop offset in pixels, clamped to [0, scaled_width - target_width].
    """
    scale_factor = target_height / source_height
    scaled_width = int(source_width * scale_factor)
    max_crop_x = max(0, scaled_width - target_width)

    # If the scaled width is already <= target width, no horizontal crop needed
    if max_crop_x <= 0:
        return 0

    default_center_x = max_crop_x // 2

    # Lazy import ultralytics and cv2 so missing dependencies do not crash the engine
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        logger.info("Vision libraries not available (%s) — using center-crop fallback.", exc)
        return default_center_x

    try:
        # Load lightweight YOLO face detection model
        # YOLO auto-downloads standard lightweight weights on first use
        model = YOLO("yolov8n.pt")  # standard compact detector
    except Exception as exc:
        logger.warning("Failed to initialize YOLO model: %s — using center-crop.", exc)
        return default_center_x

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Failed to open video file '%s' for face tracking.", video_path.name)
        return default_center_x

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(fps * sample_interval))

    detected_centers_x: list[float] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                # Run inference with person detection (class 0 = person)
                results = model(frame, classes=[0], verbose=False)
                boxes = results[0].boxes if results else None

                if boxes and len(boxes) > 0:
                    # Select largest detected person / face (most prominent speaker)
                    best_box = max(boxes, key=lambda b: float((b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])))
                    x1, _, x2, _ = best_box.xyxy[0].tolist()
                    centroid_x = (x1 + x2) / 2.0
                    detected_centers_x.append(centroid_x)

            frame_idx += 1
    except Exception as exc:
        logger.warning("Face tracking loop encountered an error: %s", exc)
    finally:
        cap.release()

    if not detected_centers_x:
        logger.info("No active speaker detected in '%s' — using center-crop.", video_path.name)
        return default_center_x

    # Apply Exponential Moving Average (EMA) to smooth speaker positions
    smoothed_centroid_x = detected_centers_x[0]
    for cx in detected_centers_x[1:]:
        smoothed_centroid_x = (ema_alpha * cx) + ((1.0 - ema_alpha) * smoothed_centroid_x)

    # Scale centroid into the target canvas space
    scaled_centroid_x = smoothed_centroid_x * scale_factor
    # Center the target 9:16 frame (width=1080) around the detected face
    desired_crop_x = scaled_centroid_x - (target_width / 2.0)
    clamped_crop_x = int(_clamp(desired_crop_x, 0, max_crop_x))

    logger.info(
        "Active speaker tracking: centered at X=%d px (scaled width=%d px, target=%d px).",
        clamped_crop_x, scaled_width, target_width,
    )
    return clamped_crop_x
