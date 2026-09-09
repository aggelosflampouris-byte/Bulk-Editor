"""
services/face_tracker.py — Active speaker tracking & dynamic 9:16 auto-framing with YOLO.

Responsibilities:
  1. Detect active speaker coordinates frame-by-frame on 16:9 widescreen video inputs using YOLO.
  2. Maintain continuous dynamic tracking across the timeline to keep the speaker centered at all times.
  3. Detect shot cuts and position changes, generating time-based dynamic crop expressions for FFmpeg.
  4. Flag whether a human speaker is recognized on screen so B-roll overlays can be suppressed
     in favor of showing the speaker continuously.
  5. Gracefully fall back to standard center-crop if no speaker is detected or if vision libraries are absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default smoothing factor for Exponential Moving Average (0 < alpha <= 1)
_DEFAULT_EMA_ALPHA = 0.20
_DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.25  # Sample 4 frames per second for high-speed tracking
_MIN_SPEAKER_PRESENCE_RATIO = 0.15       # At least 15% of frames must contain a speaker to flag has_speaker


@dataclass(frozen=True)
class SpeakerTrackingResult:
    """
    Outcome of active speaker tracking across a video clip.
    """
    has_speaker: bool
    speaker_presence_ratio: float  # Fraction of sampled frames containing a speaker (0.0 to 1.0)
    static_crop_x: int             # Single smoothed X offset for fallback
    crop_expression: str           # Dynamic FFmpeg crop expression to center speaker across time
    shot_crop_offsets: list[tuple[float, float, int]]  # List of (t_start, t_end, crop_x) segments


def _clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))


def track_active_speaker(
    video_path: Path,
    source_width: int,
    source_height: int,
    target_width: int = 1080,
    target_height: int = 1920,
    sample_interval: float = _DEFAULT_SAMPLE_INTERVAL_SECONDS,
    ema_alpha: float = _DEFAULT_EMA_ALPHA,
) -> SpeakerTrackingResult:
    """
    Track the active speaker across the entire video timeline and construct dynamic
    crop bounds to keep the speaker centered in the 9:16 frame at all times.

    Args:
        video_path:      Path to the source video clip.
        source_width:    Original video width in pixels.
        source_height:   Original video height in pixels.
        target_width:    Output 9:16 width (default 1080).
        target_height:   Output 9:16 height (default 1920).
        sample_interval: Seconds between sampled frames (default 0.25s).
        ema_alpha:       Exponential smoothing coefficient.

    Returns:
        SpeakerTrackingResult containing presence flag, static fallback X, and dynamic FFmpeg crop expression.
    """
    scale_factor = target_height / source_height
    scaled_width = int(source_width * scale_factor)
    max_crop_x = max(0, scaled_width - target_width)

    # If the scaled width is already <= target width, no horizontal crop needed
    if max_crop_x <= 0:
        return SpeakerTrackingResult(
            has_speaker=False,
            speaker_presence_ratio=0.0,
            static_crop_x=0,
            crop_expression="0",
            shot_crop_offsets=[(0.0, 99999.0, 0)],
        )

    default_center_x = max_crop_x // 2

    # Lazy import ultralytics and cv2 so missing dependencies do not crash the engine
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        logger.info("Vision libraries not available (%s) — using center-crop fallback.", exc)
        return SpeakerTrackingResult(
            has_speaker=False,
            speaker_presence_ratio=0.0,
            static_crop_x=default_center_x,
            crop_expression=str(default_center_x),
            shot_crop_offsets=[(0.0, 99999.0, default_center_x)],
        )

    try:
        model = YOLO("yolov8n.pt")
    except Exception as exc:
        logger.warning("Failed to initialize YOLO model: %s — using center-crop fallback.", exc)
        return SpeakerTrackingResult(
            has_speaker=False,
            speaker_presence_ratio=0.0,
            static_crop_x=default_center_x,
            crop_expression=str(default_center_x),
            shot_crop_offsets=[(0.0, 99999.0, default_center_x)],
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Failed to open video file '%s' for face tracking.", video_path.name)
        return SpeakerTrackingResult(
            has_speaker=False,
            speaker_presence_ratio=0.0,
            static_crop_x=default_center_x,
            crop_expression=str(default_center_x),
            shot_crop_offsets=[(0.0, 99999.0, default_center_x)],
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(fps * sample_interval))

    samples: list[tuple[float, float]] = []  # [(timestamp_sec, centroid_x)]
    total_sampled = 0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                total_sampled += 1
                t_sec = frame_idx / fps

                # Run inference with person detection (class 0 = person)
                results = model(frame, classes=[0], verbose=False)
                boxes = results[0].boxes if results else None

                if boxes and len(boxes) > 0:
                    # Select most prominent speaker (largest bounding box)
                    best_box = max(
                        boxes,
                        key=lambda b: float((b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                    )
                    x1, _, x2, _ = best_box.xyxy[0].tolist()
                    centroid_x = (x1 + x2) / 2.0
                    samples.append((t_sec, centroid_x))

            frame_idx += 1
    except Exception as exc:
        logger.warning("Speaker tracking loop encountered an error: %s", exc)
    finally:
        cap.release()

    presence_ratio = (len(samples) / total_sampled) if total_sampled > 0 else 0.0

    if not samples or presence_ratio < _MIN_SPEAKER_PRESENCE_RATIO:
        logger.info(
            "No active speaker recognized in '%s' (presence=%.1f%%) — using center-crop.",
            video_path.name, presence_ratio * 100.0,
        )
        return SpeakerTrackingResult(
            has_speaker=False,
            speaker_presence_ratio=presence_ratio,
            static_crop_x=default_center_x,
            crop_expression=str(default_center_x),
            shot_crop_offsets=[(0.0, 99999.0, default_center_x)],
        )

    # ── Continuous dynamic tracking: segment into camera shots / positions ────
    # If the speaker shifts significantly (> 120px in source) or after a cut,
    # establish a new camera shot centering the speaker.
    raw_shots: list[list[tuple[float, float]]] = []
    current_shot: list[tuple[float, float]] = [samples[0]]

    for prev_s, curr_s in zip(samples[:-1], samples[1:]):
        time_gap = curr_s[0] - prev_s[0]
        pos_jump = abs(curr_s[1] - prev_s[1])

        if pos_jump > 120.0 or time_gap > 2.0:
            # Camera cut or major speaker repositioning detected
            raw_shots.append(current_shot)
            current_shot = [curr_s]
        else:
            current_shot.append(curr_s)

    if current_shot:
        raw_shots.append(current_shot)

    # Compute smoothed crop_x for each shot
    shot_segments: list[tuple[float, float, int]] = []
    all_shot_offsets: list[int] = []

    for i, shot in enumerate(raw_shots):
        t_start = 0.0 if i == 0 else shot[0][0]
        t_end = shot[-1][0] if i < len(raw_shots) - 1 else 99999.0

        # Compute smoothed centroid for this shot segment
        smoothed_cx = shot[0][1]
        for _, cx in shot[1:]:
            smoothed_cx = (ema_alpha * cx) + ((1.0 - ema_alpha) * smoothed_cx)

        scaled_cx = smoothed_cx * scale_factor
        desired_crop_x = scaled_cx - (target_width / 2.0)
        shot_crop_x = int(_clamp(desired_crop_x, 0, max_crop_x))

        shot_segments.append((t_start, t_end, shot_crop_x))
        all_shot_offsets.append(shot_crop_x)

    # Merge consecutive shots if crop offset difference is negligible (< 35px deadzone)
    merged_shots: list[tuple[float, float, int]] = [shot_segments[0]]
    for s in shot_segments[1:]:
        prev_start, prev_end, prev_x = merged_shots[-1]
        if abs(s[2] - prev_x) < 35:
            # Merge into previous segment
            merged_shots[-1] = (prev_start, s[1], prev_x)
        else:
            merged_shots[-1] = (prev_start, s[0], prev_x)
            merged_shots.append(s)

    # Ensure last shot extends to the end of video
    if merged_shots:
        last_s = merged_shots[-1]
        merged_shots[-1] = (last_s[0], 99999.0, last_s[2])

    # Construct dynamic FFmpeg nested if expression:
    # if(lt(t, t1), x0, if(lt(t, t2), x1, ... x_final))
    if len(merged_shots) == 1:
        crop_expression = str(merged_shots[0][2])
    else:
        expr = str(merged_shots[-1][2])
        for s in reversed(merged_shots[:-1]):
            t_boundary = round(s[1], 2)
            crop_val = s[2]
            expr = f"if(lt(t,{t_boundary}),{crop_val},{expr})"
        crop_expression = expr

    # Overall static crop X fallback (median across all shots)
    sorted_offsets = sorted(all_shot_offsets)
    static_crop_x = sorted_offsets[len(sorted_offsets) // 2] if sorted_offsets else default_center_x

    logger.info(
        "Active speaker recognized (presence=%.1f%% across %d shots). Dynamic crop: %s",
        presence_ratio * 100.0, len(merged_shots), crop_expression[:100],
    )

    return SpeakerTrackingResult(
        has_speaker=True,
        speaker_presence_ratio=presence_ratio,
        static_crop_x=static_crop_x,
        crop_expression=crop_expression,
        shot_crop_offsets=merged_shots,
    )


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
    Backward-compatible helper returning single static crop X offset.
    """
    result = track_active_speaker(
        video_path=video_path,
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        sample_interval=sample_interval,
        ema_alpha=ema_alpha,
    )
    return result.static_crop_x
