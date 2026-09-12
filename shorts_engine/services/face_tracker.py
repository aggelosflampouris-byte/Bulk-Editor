import logging
from dataclasses import dataclass
from pathlib import Path

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
    static_crop_y: int             # Single smoothed Y offset for fallback
    crop_expression: str           # Dynamic FFmpeg crop X expression
    crop_y_expression: str         # Dynamic FFmpeg crop Y expression
    shot_crop_offsets: list[tuple[float, float, int, int]]  # List of (t_start, t_end, crop_x, crop_y) segments


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
    """
    scale_factor = max(target_width / source_width, target_height / source_height)
    scaled_width = int(source_width * scale_factor)
    scaled_height = int(source_height * scale_factor)
    
    max_crop_x = max(0, scaled_width - target_width)
    max_crop_y = max(0, scaled_height - target_height)
    
    default_center_x = max_crop_x // 2
    default_center_y = max_crop_y // 2

    if max_crop_x <= 0 and max_crop_y <= 0:
        return SpeakerTrackingResult(
            has_speaker=False,
            speaker_presence_ratio=0.0,
            static_crop_x=0,
            static_crop_y=0,
            crop_expression="0",
            crop_y_expression="0",
            shot_crop_offsets=[(0.0, 99999.0, 0, 0)],
        )

    # Memory Optimization: Check cache first
    try:
        from services.cache_manager import load_cache_pickle, save_cache_pickle
    except ImportError:
        from shorts_engine.services.cache_manager import load_cache_pickle, save_cache_pickle

    cache_key = f"{video_path.name}_{source_width}x{source_height}_{target_width}x{target_height}"
    cached_result = load_cache_pickle("face_tracking", cache_key)
    if cached_result is not None:
        return cached_result

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        logger.info("Vision libraries not available (%s) — using center-crop fallback.", exc)
        return SpeakerTrackingResult(
            has_speaker=False,
            speaker_presence_ratio=0.0,
            static_crop_x=default_center_x,
            static_crop_y=default_center_y,
            crop_expression=str(default_center_x),
            crop_y_expression=str(default_center_y),
            shot_crop_offsets=[(0.0, 99999.0, default_center_x, default_center_y)],
        )

    try:
        model = YOLO("yolov8n.pt")
    except Exception as exc:
        logger.warning("Failed to initialize YOLO model: %s — using center-crop fallback.", exc)
        return SpeakerTrackingResult(
            has_speaker=False,
            speaker_presence_ratio=0.0,
            static_crop_x=default_center_x,
            static_crop_y=default_center_y,
            crop_expression=str(default_center_x),
            crop_y_expression=str(default_center_y),
            shot_crop_offsets=[(0.0, 99999.0, default_center_x, default_center_y)],
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Failed to open video file '%s' for face tracking.", video_path.name)
        return SpeakerTrackingResult(
            has_speaker=False,
            speaker_presence_ratio=0.0,
            static_crop_x=default_center_x,
            static_crop_y=default_center_y,
            crop_expression=str(default_center_x),
            crop_y_expression=str(default_center_y),
            shot_crop_offsets=[(0.0, 99999.0, default_center_x, default_center_y)],
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(fps * sample_interval))

    samples: list[tuple[float, float, float]] = []
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
                results = model(frame, classes=[0], verbose=False)
                boxes = results[0].boxes if results else None
                if boxes and len(boxes) > 0:
                    best_box = max(
                        boxes,
                        key=lambda b: float((b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                    )
                    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
                    centroid_x = (x1 + x2) / 2.0
                    centroid_y = (y1 + y2) / 2.0
                    samples.append((t_sec, centroid_x, centroid_y))
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
            static_crop_y=default_center_y,
            crop_expression=str(default_center_x),
            crop_y_expression=str(default_center_y),
            shot_crop_offsets=[(0.0, 99999.0, default_center_x, default_center_y)],
        )

    raw_shots: list[list[tuple[float, float, float]]] = []
    current_shot: list[tuple[float, float, float]] = [samples[0]]

    for prev_s, curr_s in zip(samples[:-1], samples[1:]):
        time_gap = curr_s[0] - prev_s[0]
        pos_jump_x = abs(curr_s[1] - prev_s[1])
        pos_jump_y = abs(curr_s[2] - prev_s[2])

        if max(pos_jump_x, pos_jump_y) > 120.0 or time_gap > 2.0:
            raw_shots.append(current_shot)
            current_shot = [curr_s]
        else:
            current_shot.append(curr_s)

    if current_shot:
        raw_shots.append(current_shot)

    shot_segments: list[tuple[float, float, int, int]] = []
    all_shot_offsets_x: list[int] = []
    all_shot_offsets_y: list[int] = []

    for i, shot in enumerate(raw_shots):
        t_start = 0.0 if i == 0 else shot[0][0]
        t_end = shot[-1][0] if i < len(raw_shots) - 1 else 99999.0

        smoothed_cx = shot[0][1]
        smoothed_cy = shot[0][2]
        for _, cx, cy in shot[1:]:
            smoothed_cx = (ema_alpha * cx) + ((1.0 - ema_alpha) * smoothed_cx)
            smoothed_cy = (ema_alpha * cy) + ((1.0 - ema_alpha) * smoothed_cy)

        scaled_cx = smoothed_cx * scale_factor
        scaled_cy = smoothed_cy * scale_factor
        
        desired_crop_x = scaled_cx - (target_width / 2.0)
        desired_crop_y = scaled_cy - (target_height / 2.0)
        
        shot_crop_x = int(_clamp(desired_crop_x, 0, max_crop_x))
        shot_crop_y = int(_clamp(desired_crop_y, 0, max_crop_y))

        shot_segments.append((t_start, t_end, shot_crop_x, shot_crop_y))
        all_shot_offsets_x.append(shot_crop_x)
        all_shot_offsets_y.append(shot_crop_y)

    merged_shots: list[tuple[float, float, int, int]] = [shot_segments[0]]
    for s in shot_segments[1:]:
        prev_start, prev_end, prev_x, prev_y = merged_shots[-1]
        if abs(s[2] - prev_x) < 35 and abs(s[3] - prev_y) < 35:
            merged_shots[-1] = (prev_start, s[1], prev_x, prev_y)
        else:
            merged_shots[-1] = (prev_start, s[0], prev_x, prev_y)
            merged_shots.append(s)

    if merged_shots:
        last_s = merged_shots[-1]
        merged_shots[-1] = (last_s[0], 99999.0, last_s[2], last_s[3])

    def _build_expr(idx: int) -> str:
        if len(merged_shots) == 1:
            return str(merged_shots[0][idx])
        expr = str(merged_shots[-1][idx])
        for s in reversed(merged_shots[:-1]):
            t_boundary = round(s[1], 2)
            expr = f"if(lt(t,{t_boundary}),{s[idx]},{expr})"
        return expr

    crop_expression = _build_expr(2)
    crop_y_expression = _build_expr(3)

    sorted_x = sorted(all_shot_offsets_x)
    sorted_y = sorted(all_shot_offsets_y)
    static_crop_x = sorted_x[len(sorted_x) // 2] if sorted_x else default_center_x
    static_crop_y = sorted_y[len(sorted_y) // 2] if sorted_y else default_center_y

    logger.info(
        "Active speaker recognized (presence=%.1f%% across %d shots). Dynamic crop: X=%s, Y=%s",
        presence_ratio * 100.0, len(merged_shots), crop_expression[:50], crop_y_expression[:50]
    )

    result = SpeakerTrackingResult(
        has_speaker=True,
        speaker_presence_ratio=presence_ratio,
        static_crop_x=static_crop_x,
        static_crop_y=static_crop_y,
        crop_expression=crop_expression,
        crop_y_expression=crop_y_expression,
        shot_crop_offsets=merged_shots,
    )
    save_cache_pickle("face_tracking", cache_key, result)
    return result
