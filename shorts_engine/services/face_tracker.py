import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default smoothing factor for Exponential Moving Average (0 < alpha <= 1)
_DEFAULT_EMA_ALPHA = 0.20
# Sample one frame every N seconds — 0.5s (2 fps) for balanced tracking and thermal load.
# For short clips (<60s) we use 0.25s (4 fps) to follow fast speaker movement.
_DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.5
_SHORT_CLIP_SAMPLE_INTERVAL_SECONDS = 0.25
_SHORT_CLIP_THRESHOLD_SECONDS = 60.0
_MIN_SPEAKER_PRESENCE_RATIO = 0.15       # At least 15% of frames must contain a speaker to flag has_speaker


def _best_torch_device() -> str:
    """
    Return the best available torch compute device string.

    Priority: CUDA (NVIDIA/AMD ROCm) → MPS (Apple Silicon) → CPU.
    Failures are safely logged so that YOLO still runs on CPU if torch
    is unavailable or no GPU is present.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # MPS = Apple Metal Performance Shaders (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("Torch device probe failed (%s) — falling back to CPU", exc)
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
    crop bounds to identify and follow the speaker's face in the 9:16 frame while they speak.
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
    if infer_device == "cpu":
        try:
            import os

            import torch
            cores = os.cpu_count() or 4
            # Cap PyTorch intra-op threads to prevent thermal saturation on ThinkPad/laptop CPUs
            optimal_torch_threads = max(1, min(4, cores // 2))
            torch.set_num_threads(optimal_torch_threads)
            logger.debug("Configured PyTorch CPU inference threads=%d for face tracking", optimal_torch_threads)
        except (ImportError, RuntimeError, AttributeError) as exc:
            logger.debug("Failed configuring torch CPU threads (%s)", exc)

    model = None
    for model_name in ("yolov8n-pose.pt", "yolov8n.pt"):
        try:
            m = YOLO(model_name)
            m.to(infer_device)
            model = m
            break
        except (RuntimeError, ValueError, OSError, ImportError) as exc:
            logger.debug("YOLO model '%s' failed to load: %s", model_name, exc)

    if model is None:
        logger.warning("Failed to initialize YOLO model — using center-crop fallback.")
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

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    samples: list[tuple[float, float, float]] = []
    total_sampled = 0
    frame_idx = 0
    last_speaker_cx: float | None = None
    last_speaker_cy: float | None = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                total_sampled += 1
                t_sec = frame_idx / fps
                # Use conf=0.25 to reliably detect speakers even in dim / studio LED lighting
                if has_torch:
                    with torch.inference_mode():
                        results = model(frame, classes=[0], conf=0.25, verbose=False, device=infer_device)
                else:
                    results = model(frame, classes=[0], conf=0.25, verbose=False, device=infer_device)

                boxes = results[0].boxes if results else None
                if boxes and len(boxes) > 0:
                    best_idx = 0
                    best_score = -1.0
                    for b_i, b in enumerate(boxes):
                        bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                        b_area = float((bx2 - bx1) * (by2 - by1))
                        b_cx = (bx1 + bx2) / 2.0
                        b_cy = by1 + (by2 - by1) * 0.18

                        # Spatial continuity bonus: track the same speaker across frames
                        score = b_area
                        if last_speaker_cx is not None and last_speaker_cy is not None:
                            dist = ((b_cx - last_speaker_cx) ** 2 + (b_cy - last_speaker_cy) ** 2) ** 0.5
                            max_dist = source_width * 0.35
                            if dist < max_dist:
                                score *= 1.0 + 1.8 * (1.0 - (dist / max_dist))

                        # Center framing preference
                        center_dist = abs(b_cx - (source_width / 2.0))
                        score *= 1.0 - 0.20 * (center_dist / (source_width / 2.0))

                        if score > best_score:
                            best_score = score
                            best_idx = b_i

                    best_box = boxes[best_idx]
                    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
                    centroid_x = (x1 + x2) / 2.0
                    # Face is positioned in the upper ~18% of the human bounding box
                    centroid_y = y1 + (y2 - y1) * 0.18

                    # Refine centroid using facial keypoints (Nose, L/R Eye, L/R Ear)
                    keypoints = getattr(results[0], "keypoints", None)
                    if keypoints is not None and len(keypoints) > best_idx:
                        kp = keypoints[best_idx]
                        if kp.xy is not None and len(kp.xy) > 0:
                            face_kps = kp.xy[0][:5]
                            confs = kp.conf[0][:5] if kp.conf is not None else None

                            valid_x, valid_y = [], []
                            for k_idx, pt in enumerate(face_kps):
                                k_conf = float(confs[k_idx]) if confs is not None else 1.0
                                if k_conf > 0.25 and pt[0] > 0 and pt[1] > 0:
                                    valid_x.append(float(pt[0]))
                                    valid_y.append(float(pt[1]))

                            # Eye-level prioritization for optimal portrait framing
                            if len(face_kps) >= 3 and confs is not None and confs[1] > 0.3 and confs[2] > 0.3:
                                eye_cx = float((face_kps[1][0] + face_kps[2][0]) / 2.0)
                                eye_cy = float((face_kps[1][1] + face_kps[2][1]) / 2.0)
                                if eye_cx > 0 and eye_cy > 0:
                                    centroid_x = eye_cx
                                    centroid_y = eye_cy
                            elif valid_x and valid_y:
                                centroid_x = sum(valid_x) / len(valid_x)
                                centroid_y = sum(valid_y) / len(valid_y)

                    last_speaker_cx = centroid_x
                    last_speaker_cy = centroid_y
                    samples.append((t_sec, centroid_x, centroid_y))
            frame_idx += 1
    except (RuntimeError, ValueError, OSError, cv2.error) as exc:
        logger.warning("Speaker tracking loop encountered an error: %s", exc)
    finally:
        cap.release()
        import gc
        gc.collect()

    presence_ratio = (len(samples) / total_sampled) if total_sampled > 0 else 0.0

    if not samples:
        logger.info(
            "No active speaker recognized in '%s' — using center-crop.",
            video_path.name,
        )
        return SpeakerTrackingResult(
            has_speaker=False,
            speaker_presence_ratio=0.0,
            static_crop_x=default_center_x,
            static_crop_y=default_center_y,
            crop_expression=str(default_center_x),
            crop_y_expression=str(default_center_y),
            shot_crop_offsets=[(0.0, 99999.0, default_center_x, default_center_y)],
        )

    # Compute detected median crop coordinates from samples to prevent empty background crop
    sorted_s_x = sorted(s[1] for s in samples)
    sorted_s_y = sorted(s[2] for s in samples)
    median_cx = sorted_s_x[len(sorted_s_x) // 2]
    median_cy = sorted_s_y[len(sorted_s_y) // 2]
    # Eye-line at 35% from top of 9:16 portrait viewport for ideal portrait headroom
    detected_crop_x = int(_clamp((median_cx * scale_factor) - (target_width / 2.0), 0, max_crop_x))
    detected_crop_y = int(_clamp((median_cy * scale_factor) - (target_height * 0.35), 0, max_crop_y))

    if presence_ratio < _MIN_SPEAKER_PRESENCE_RATIO:
        logger.info(
            "Low speaker presence in '%s' (%.1f%%) — using detected median crop X=%d, Y=%d.",
            video_path.name, presence_ratio * 100.0, detected_crop_x, detected_crop_y,
        )
        return SpeakerTrackingResult(
            has_speaker=False,
            speaker_presence_ratio=presence_ratio,
            static_crop_x=detected_crop_x,
            static_crop_y=detected_crop_y,
            crop_expression=str(detected_crop_x),
            crop_y_expression=str(detected_crop_y),
            shot_crop_offsets=[(0.0, 99999.0, detected_crop_x, detected_crop_y)],
        )

    # Convert samples to desired crop bounds per frame:
    # Rule of thirds: Eye-line at 35% from top of 9:16 portrait viewport
    target_crops: list[tuple[float, float, float]] = []
    for t_sec, cx, cy in samples:
        sc_x = cx * scale_factor
        sc_y = cy * scale_factor
        des_x = _clamp(sc_x - (target_width / 2.0), 0, max_crop_x)
        des_y = _clamp(sc_y - (target_height * 0.35), 0, max_crop_y)
        target_crops.append((t_sec, des_x, des_y))

    # Segment into distinct camera shots (cuts or large scene jumps)
    raw_shots: list[list[tuple[float, float, float]]] = []
    current_shot: list[tuple[float, float, float]] = [target_crops[0]]

    for prev_s, curr_s in itertools.pairwise(target_crops):
        time_gap = curr_s[0] - prev_s[0]
        pos_jump_x = abs(curr_s[1] - prev_s[1])
        pos_jump_y = abs(curr_s[2] - prev_s[2])

        if max(pos_jump_x, pos_jump_y) > 180.0 or time_gap > 2.0:
            raw_shots.append(current_shot)
            current_shot = [curr_s]
        else:
            current_shot.append(curr_s)

    if current_shot:
        raw_shots.append(current_shot)

    shot_segments: list[tuple[float, float, int, int]] = []
    all_shot_offsets_x: list[int] = []
    all_shot_offsets_y: list[int] = []
    shot_expressions_x: list[tuple[float, str]] = []
    shot_expressions_y: list[tuple[float, str]] = []

    for i, shot in enumerate(raw_shots):
        t_shot_start = 0.0 if i == 0 else shot[0][0]
        t_shot_end = shot[-1][0] if i < len(raw_shots) - 1 else 99999.0

        # Exponential moving average filter within the shot
        smoothed_cx = shot[0][1]
        smoothed_cy = shot[0][2]
        smoothed_pts: list[tuple[float, int, int]] = [
            (shot[0][0], int(_clamp(smoothed_cx, 0, max_crop_x)), int(_clamp(smoothed_cy, 0, max_crop_y)))
        ]

        for pt_t, raw_cx, raw_cy in shot[1:]:
            smoothed_cx = (ema_alpha * raw_cx) + ((1.0 - ema_alpha) * smoothed_cx)
            smoothed_cy = (ema_alpha * raw_cy) + ((1.0 - ema_alpha) * smoothed_cy)
            smoothed_pts.append((
                pt_t,
                int(_clamp(smoothed_cx, 0, max_crop_x)),
                int(_clamp(smoothed_cy, 0, max_crop_y)),
            ))

        # Shot-level representative offsets
        median_shot_x = sorted([p[1] for p in smoothed_pts])[len(smoothed_pts) // 2]
        median_shot_y = sorted([p[2] for p in smoothed_pts])[len(smoothed_pts) // 2]
        shot_segments.append((t_shot_start, t_shot_end, median_shot_x, median_shot_y))
        all_shot_offsets_x.append(median_shot_x)
        all_shot_offsets_y.append(median_shot_y)

        # Movement span check: if range of motion in shot is subtle (< 32px), lock to median for rock-solid framing
        range_x = max(p[1] for p in smoothed_pts) - min(p[1] for p in smoothed_pts)
        range_y = max(p[2] for p in smoothed_pts) - min(p[2] for p in smoothed_pts)

        if range_x < 32 and range_y < 32:
            kps_x: list[tuple[float, int]] = [(smoothed_pts[0][0], median_shot_x), (smoothed_pts[-1][0], median_shot_x)]
            kps_y: list[tuple[float, int]] = [(smoothed_pts[0][0], median_shot_y), (smoothed_pts[-1][0], median_shot_y)]
        else:
            # Build dynamic follow keyframes with 24px adaptive deadzone to track intentional motion without jitter
            kps_x = [(smoothed_pts[0][0], smoothed_pts[0][1])]
            kps_y = [(smoothed_pts[0][0], smoothed_pts[0][2])]

            for pt_t, cur_x, cur_y in smoothed_pts[1:]:
                if abs(cur_x - kps_x[-1][1]) >= 24:
                    kps_x.append((pt_t, cur_x))
                if abs(cur_y - kps_y[-1][1]) >= 24:
                    kps_y.append((pt_t, cur_y))

            # Ensure shot boundary end keyframe is present
            last_pt = smoothed_pts[-1]
            if kps_x[-1][0] < last_pt[0]:
                kps_x.append((last_pt[0], last_pt[1]))
            if kps_y[-1][0] < last_pt[0]:
                kps_y.append((last_pt[0], last_pt[2]))

            # Decimate if excessive nodes (cap to max 8 nodes per shot)
            if len(kps_x) > 8:
                step = max(1, len(kps_x) // 8)
                kps_x = [kps_x[idx] for idx in range(0, len(kps_x), step)]
                if kps_x[-1] != (last_pt[0], last_pt[1]):
                    kps_x.append((last_pt[0], last_pt[1]))

            if len(kps_y) > 8:
                step = max(1, len(kps_y) // 8)
                kps_y = [kps_y[idx] for idx in range(0, len(kps_y), step)]
                if kps_y[-1] != (last_pt[0], last_pt[2]):
                    kps_y.append((last_pt[0], last_pt[2]))

        # Construct piecewise follow expression for this shot
        def _build_piecewise(kps: list[tuple[float, int]]) -> str:
            if not kps:
                return "0"
            if len(kps) == 1:
                return str(kps[0][1])
            pieces: list[tuple[float, str]] = []
            for (t_a, pos_a), (t_b, pos_b) in itertools.pairwise(kps):
                dur = round(t_b - t_a, 2)
                if dur <= 0.05 or pos_a == pos_b:
                    piece_expr = f"{pos_a}"
                else:
                    diff = pos_b - pos_a
                    piece_expr = f"{pos_a}+({diff})*((t-{round(t_a, 2)})/{dur})"
                pieces.append((t_b, piece_expr))
            sub_expr = str(kps[-1][1])
            for t_end, p_expr in reversed(pieces):
                sub_expr = f"if(lt(t,{round(t_end, 2)}),{p_expr},{sub_expr})"
            return sub_expr

        shot_expr_x = _build_piecewise(kps_x)
        shot_expr_y = _build_piecewise(kps_y)

        shot_expressions_x.append((t_shot_end, shot_expr_x))
        shot_expressions_y.append((t_shot_end, shot_expr_y))

    def _combine_shot_exprs(shot_exprs: list[tuple[float, str]]) -> str:
        if not shot_exprs:
            return "0"
        if len(shot_exprs) == 1:
            return shot_exprs[0][1]
        overall = shot_exprs[-1][1]
        for t_end, s_expr in reversed(shot_exprs[:-1]):
            overall = f"if(lt(t,{round(t_end, 2)}),{s_expr},{overall})"
        return overall

    crop_expression = _combine_shot_exprs(shot_expressions_x)
    crop_y_expression = _combine_shot_exprs(shot_expressions_y)

    sorted_x = sorted(all_shot_offsets_x)
    sorted_y = sorted(all_shot_offsets_y)
    static_crop_x = sorted_x[len(sorted_x) // 2] if sorted_x else default_center_x
    static_crop_y = sorted_y[len(sorted_y) // 2] if sorted_y else default_center_y

    logger.info(
        "Active speaker recognized (presence=%.1f%% across %d shots). Dynamic follow: X=%s, Y=%s",
        presence_ratio * 100.0, len(shot_segments), crop_expression[:60], crop_y_expression[:60]
    )

    result = SpeakerTrackingResult(
        has_speaker=True,
        speaker_presence_ratio=presence_ratio,
        static_crop_x=static_crop_x,
        static_crop_y=static_crop_y,
        crop_expression=crop_expression,
        crop_y_expression=crop_y_expression,
        shot_crop_offsets=shot_segments,
    )
    save_cache_pickle("face_tracking", cache_key, result)
    return result
