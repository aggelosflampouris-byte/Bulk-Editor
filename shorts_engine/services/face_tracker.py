import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default smoothing factor for Exponential Moving Average (0 < alpha <= 1)
_DEFAULT_EMA_ALPHA = 0.10
# Sample one frame every N seconds — increased to 0.5s (2 fps) for shorter thermal load.
# For short clips (<60s) we use 0.25s to maintain accuracy on rapid speaker movement.
_DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.5
_SHORT_CLIP_SAMPLE_INTERVAL_SECONDS = 0.25
_SHORT_CLIP_THRESHOLD_SECONDS = 60.0
_MIN_SPEAKER_PRESENCE_RATIO = 0.15       # At least 15% of frames must contain a speaker to flag has_speaker


def _best_torch_device() -> str:
    """
    Return the best available torch compute device string.

    Priority: CUDA (NVIDIA/AMD ROCm) → MPS (Apple Silicon) → CPU.
    Failures are silently caught so that YOLO still runs on CPU if torch
    is unavailable or no GPU is present.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # MPS = Apple Metal Performance Shaders (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


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
        from shorts_engine.services.cache_manager import (
            load_cache_pickle,
            save_cache_pickle,
        )

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

    infer_device = _best_torch_device()
    try:
        model = YOLO("yolov8n-pose.pt")
        model.to(infer_device)
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
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    video_duration_sec = (total_frames / fps) if fps > 0 else 0.0

    # Adaptive sampling: finer resolution for short clips (<60 s) to preserve
    # accuracy on rapid speaker movement; coarser for long clips to reduce
    # CPU/GPU thermal load on the laptop.
    if sample_interval == _DEFAULT_SAMPLE_INTERVAL_SECONDS:
        effective_interval = (
            _SHORT_CLIP_SAMPLE_INTERVAL_SECONDS
            if video_duration_sec < _SHORT_CLIP_THRESHOLD_SECONDS
            else _DEFAULT_SAMPLE_INTERVAL_SECONDS
        )
    else:
        effective_interval = sample_interval

    frame_step = max(1, int(fps * effective_interval))
    logger.debug(
        "Face tracking '%s': duration=%.1fs, fps=%.1f, frame_step=%d (%.2fs interval, device=%s)",
        video_path.name, video_duration_sec, fps, frame_step, effective_interval, infer_device,
    )


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
                results = model(frame, classes=[0], conf=0.6, verbose=False, device=infer_device)

                boxes = results[0].boxes if results else None
                if boxes and len(boxes) > 0:
                    best_idx = max(
                        range(len(boxes)),
                        key=lambda i: float((boxes[i].xyxy[0][2] - boxes[i].xyxy[0][0]) * (boxes[i].xyxy[0][3] - boxes[i].xyxy[0][1]))
                    )
                    best_box = boxes[best_idx]
                    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
                    centroid_x = (x1 + x2) / 2.0
                    centroid_y = (y1 + y2) / 2.0
                    
                    # Try to refine centroid using facial keypoints (Nose, L/R Eye, L/R Ear)
                    keypoints = results[0].keypoints
                    if keypoints is not None and len(keypoints) > best_idx:
                        kp = keypoints[best_idx]
                        if kp.xy is not None and len(kp.xy) > 0:
                            face_kps = kp.xy[0][:5]
                            confs = kp.conf[0][:5] if kp.conf is not None else None
                            
                            valid_x, valid_y = [], []
                            for k_idx, pt in enumerate(face_kps):
                                conf = float(confs[k_idx]) if confs is not None else 1.0
                                if conf > 0.5 and pt[0] > 0 and pt[1] > 0:
                                    valid_x.append(float(pt[0]))
                                    valid_y.append(float(pt[1]))
                            
                            if valid_x and valid_y:
                                centroid_x = sum(valid_x) / len(valid_x)
                                centroid_y = sum(valid_y) / len(valid_y)
                                
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
