import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    Outcome of active speaker and focal object tracking across a video clip.
    """
    has_speaker: bool
    speaker_presence_ratio: float  # Fraction of sampled frames containing a speaker (0.0 to 1.0)
    static_crop_x: int             # Single smoothed X offset for fallback
    static_crop_y: int             # Single smoothed Y offset for fallback
    crop_expression: str           # Dynamic FFmpeg crop X expression
    crop_y_expression: str         # Dynamic FFmpeg crop Y expression
    shot_crop_offsets: list[tuple[float, float, int, int]]  # List of (t_start, t_end, crop_x, crop_y) segments
    subject_type: str = "speaker_face"  # "speaker_face" | "salient_motion_focus" | "center"


def _clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))


def _extract_mouth_roi(
    frame_shape: tuple[int, ...],
    box_xyxy: list[float],
    kp_xy: Any | None = None,
    kp_conf: Any | None = None,
) -> tuple[int, int, int, int]:
    """
    Extract a bounding box (x1, y1, x2, y2) around the mouth/jaw region.

    If YOLO-pose facial keypoints (Nose=0, Left Eye=1, Right Eye=2) are available with
    good confidence (>0.25), use facial geometry to precisely isolate the mouth.
    Otherwise, fall back to the lower anatomical 35% of the human head bounding box.
    """
    bx1, by1, bx2, by2 = [int(v) for v in box_xyxy]
    img_h, img_w = frame_shape[:2]

    try:
        if kp_xy is not None and len(kp_xy) >= 3:
            conf_ok = True
            if kp_conf is not None and len(kp_conf) >= 3:
                conf_ok = float(kp_conf[0]) > 0.25 and float(kp_conf[1]) > 0.25

            if conf_ok:
                nx, ny = float(kp_xy[0][0]), float(kp_xy[0][1])
                lex, ley = float(kp_xy[1][0]), float(kp_xy[1][1])
                rex, rey = float(kp_xy[2][0]), float(kp_xy[2][1])

                eye_y = (ley + rey) / 2.0
                eye_dist = max(12.0, abs(lex - rex))
                nose_dist = max(10.0, ny - eye_y)

                mouth_cx = int(nx)
                mouth_cy = int(ny + nose_dist * 0.90)
                half_w = int(eye_dist * 0.70)
                half_h = int(nose_dist * 0.85)

                mx1 = max(0, mouth_cx - half_w)
                mx2 = min(img_w, mouth_cx + half_w)
                my1 = max(0, mouth_cy - half_h)
                my2 = min(img_h, mouth_cy + half_h)
                if mx2 > mx1 and my2 > my1:
                    return mx1, my1, mx2, my2
    except (IndexError, TypeError, ValueError, AttributeError):
        pass

    # Anatomical fallback: head is roughly upper 32% of human bounding box
    head_h = (by2 - by1) * 0.32
    mouth_cy = by1 + head_h * 0.78
    half_h = max(8.0, head_h * 0.28)
    half_w = max(12.0, (bx2 - bx1) * 0.18)
    cx = (bx1 + bx2) / 2.0

    mx1 = max(0, int(cx - half_w))
    mx2 = min(img_w, int(cx + half_w))
    my1 = max(0, int(mouth_cy - half_h))
    my2 = min(img_h, int(mouth_cy + half_h))
    return mx1, my1, max(mx1 + 1, mx2), max(my1 + 1, my2)


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

    cache_key = f"{video_path.name}_{source_width}x{source_height}_{target_width}x{target_height}_v2"
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
    speaker_detected_count = 0
    prev_small_gray = None

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
                keypoints = getattr(results[0], "keypoints", None)
                if boxes and len(boxes) > 0:
                    speaker_detected_count += 1
                    if len(boxes) == 1:
                        best_idx = 0
                    else:
                        # Multi-person scene (e.g. split-screen interview, studio panel, podcast dialogue):
                        # Measure active lip/mouth motion over a short forward sub-frame window to identify
                        # the person who is actively articulating speech, rather than a silent listener.
                        rois = []
                        for b_i, b in enumerate(boxes):
                            kp_xy = keypoints[b_i].xy[0] if keypoints is not None and len(keypoints) > b_i else None
                            kp_conf = (
                                keypoints[b_i].conf[0]
                                if keypoints is not None
                                and len(keypoints) > b_i
                                and keypoints[b_i].conf is not None
                                else None
                            )
                            rois.append(_extract_mouth_roi(frame.shape, b.xyxy[0].tolist(), kp_xy, kp_conf))

                        diffs = [0.0] * len(boxes)
                        sub_frames = []
                        # Advance 2-3 consecutive sub-frames to sample fine-grained articulation dynamics
                        for _ in range(3):
                            ret_s, f_s = cap.read()
                            if not ret_s:
                                break
                            frame_idx += 1
                            sub_frames.append(f_s)

                        if sub_frames:
                            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            for sf in sub_frames:
                                curr_gray = cv2.cvtColor(sf, cv2.COLOR_BGR2GRAY)
                                diff = cv2.absdiff(curr_gray, prev_gray)
                                for b_i, (mx1, my1, mx2, my2) in enumerate(rois):
                                    roi_diff = diff[my1:my2, mx1:mx2]
                                    if roi_diff.size > 0:
                                        diffs[b_i] += float(cv2.mean(roi_diff)[0])
                                prev_gray = curr_gray

                        best_idx = 0
                        best_score = -1.0
                        for b_i, b in enumerate(boxes):
                            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                            b_area = float((bx2 - bx1) * (by2 - by1))
                            b_cx = (bx1 + bx2) / 2.0
                            b_cy = by1 + (by2 - by1) * 0.18
                            m_diff = diffs[b_i] if b_i < len(diffs) else 0.0

                            # Non-linear boost for speech articulation (moving lips vs static listener)
                            speaking_weight = (m_diff + 0.1) ** 1.6
                            score = b_area * speaking_weight

                            # Spatial continuity: maintain moderate lock ONLY if candidate is actively talking
                            if last_speaker_cx is not None and last_speaker_cy is not None:
                                dist = ((b_cx - last_speaker_cx) ** 2 + (b_cy - last_speaker_cy) ** 2) ** 0.5
                                max_dist = source_width * 0.35
                                if dist < max_dist and m_diff > 1.2:
                                    score *= 1.0 + 0.5 * (1.0 - (dist / max_dist))

                            if score > best_score:
                                best_score = score
                                best_idx = b_i

                    best_box = boxes[best_idx]
                    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
                    centroid_x = (x1 + x2) / 2.0
                    # Face is positioned in the upper ~18% of the human bounding box
                    centroid_y = y1 + (y2 - y1) * 0.18

                    # Refine centroid using facial keypoints (Nose, L/R Eye, L/R Ear)
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
                else:
                    # No human face/speaker in frame: track salient visual motion & action
                    # (e.g. car engine bay, hands pointing at parts, tools, product unboxing)
                    small_gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
                    if prev_small_gray is not None:
                        diff = cv2.absdiff(small_gray, prev_small_gray)
                        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
                        moments = cv2.moments(thresh)
                        if moments["m00"] > 300:
                            cx_small = moments["m10"] / moments["m00"]
                            cy_small = moments["m01"] / moments["m00"]
                            focal_cx = (cx_small / 320.0) * source_width
                            focal_cy = (cy_small / 180.0) * source_height
                            last_speaker_cx = focal_cx
                            last_speaker_cy = focal_cy
                            samples.append((t_sec, focal_cx, focal_cy))
                        elif last_speaker_cx is not None and last_speaker_cy is not None:
                            samples.append((t_sec, last_speaker_cx, last_speaker_cy))
                    prev_small_gray = small_gray
            frame_idx += 1
    except (RuntimeError, ValueError, OSError, cv2.error) as exc:
        logger.warning("Speaker tracking loop encountered an error: %s", exc)
    finally:
        cap.release()
        import gc
        gc.collect()

    presence_ratio = (speaker_detected_count / total_sampled) if total_sampled > 0 else 0.0

    if not samples:
        logger.info(
            "No active speaker or salient focal motion recognized in '%s' — using center-crop.",
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
            subject_type="center",
        )

    # Compute detected median crop coordinates from samples to prevent empty background crop
    sorted_s_x = sorted(s[1] for s in samples)
    sorted_s_y = sorted(s[2] for s in samples)
    median_cx = sorted_s_x[len(sorted_s_x) // 2]
    median_cy = sorted_s_y[len(sorted_s_y) // 2]
    # Speaker head center framing in 9:16 portrait viewport
    detected_crop_x = int(_clamp((median_cx * scale_factor) - (target_width / 2.0), 0, max_crop_x))
    detected_crop_y = int(_clamp((median_cy * scale_factor) - (target_height / 2.0), 0, max_crop_y))

    detected_subject = "speaker_face" if speaker_detected_count > 0 else "salient_motion_focus"

    if presence_ratio < _MIN_SPEAKER_PRESENCE_RATIO:
        logger.info(
            "Low speaker presence in '%s' (%.1f%%, mode=%s) — using detected median crop X=%d, Y=%d.",
            video_path.name, presence_ratio * 100.0, detected_subject, detected_crop_x, detected_crop_y,
        )
        return SpeakerTrackingResult(
            has_speaker=False,
            speaker_presence_ratio=presence_ratio,
            static_crop_x=detected_crop_x,
            static_crop_y=detected_crop_y,
            crop_expression=str(detected_crop_x),
            crop_y_expression=str(detected_crop_y),
            shot_crop_offsets=[(0.0, 99999.0, detected_crop_x, detected_crop_y)],
            subject_type=detected_subject,
        )

    # Convert samples to desired crop bounds per frame:
    # Speaker's head is kept at the center of the screen (50% horizontal and 50% vertical)
    target_crops: list[tuple[float, float, float]] = []
    for t_sec, cx, cy in samples:
        sc_x = cx * scale_factor
        sc_y = cy * scale_factor
        des_x = _clamp(sc_x - (target_width / 2.0), 0, max_crop_x)
        des_y = _clamp(sc_y - (target_height / 2.0), 0, max_crop_y)
        target_crops.append((t_sec, des_x, des_y))

    # 1. Outlier Rejection: eliminate transient 1-sample spikes (e.g. YOLO latching onto a background tool or passerby)
    cleaned_crops = list(target_crops)
    if len(cleaned_crops) >= 3:
        for i in range(1, len(cleaned_crops) - 1):
            prev_x = cleaned_crops[i - 1][1]
            curr_x = cleaned_crops[i][1]
            next_x = cleaned_crops[i + 1][1]
            if abs(curr_x - prev_x) > 220.0 and abs(curr_x - next_x) > 220.0 and abs(prev_x - next_x) < 140.0:
                cleaned_crops[i] = (cleaned_crops[i][0], (prev_x + next_x) / 2.0, cleaned_crops[i][2])

            prev_y = cleaned_crops[i - 1][2]
            curr_y = cleaned_crops[i][2]
            next_y = cleaned_crops[i + 1][2]
            if abs(curr_y - prev_y) > 180.0 and abs(curr_y - next_y) > 180.0 and abs(prev_y - next_y) < 100.0:
                cleaned_crops[i] = (cleaned_crops[i][0], cleaned_crops[i][1], (prev_y + next_y) / 2.0)

    # 2. Continuous Exponential Moving Average (EMA) with physical velocity clamping
    # Guarantees the camera smoothly glides and never teleports or introduces jump cuts
    smoothed_pts: list[tuple[float, int, int]] = []
    cur_x = cleaned_crops[0][1]
    cur_y = cleaned_crops[0][2]
    smoothed_pts.append((
        cleaned_crops[0][0],
        int(_clamp(cur_x, 0, max_crop_x)),
        int(_clamp(cur_y, 0, max_crop_y)),
    ))

    for prev_c, curr_c in itertools.pairwise(cleaned_crops):
        dt = max(0.01, curr_c[0] - prev_c[0])
        target_x = curr_c[1]
        target_y = curr_c[2]

        next_x = (ema_alpha * target_x) + ((1.0 - ema_alpha) * cur_x)
        next_y = (ema_alpha * target_y) + ((1.0 - ema_alpha) * cur_y)

        # Max velocity limit: ~280px/s X and ~180px/s Y prevents jarring whip pans or jump cuts
        max_step_x = 280.0 * dt
        max_step_y = 180.0 * dt
        diff_x = next_x - cur_x
        diff_y = next_y - cur_y
        clamped_diff_x = _clamp(diff_x, -max_step_x, max_step_x)
        clamped_diff_y = _clamp(diff_y, -max_step_y, max_step_y)

        cur_x += clamped_diff_x
        cur_y += clamped_diff_y

        smoothed_pts.append((
            curr_c[0],
            int(_clamp(cur_x, 0, max_crop_x)),
            int(_clamp(cur_y, 0, max_crop_y)),
        ))

    all_x = [p[1] for p in smoothed_pts]
    all_y = [p[2] for p in smoothed_pts]
    median_x = sorted(all_x)[len(all_x) // 2]
    median_y = sorted(all_y)[len(all_y) // 2]
    range_x = max(all_x) - min(all_x)
    range_y = max(all_y) - min(all_y)

    # 3. Deadzone & Steady Shot Stability: if motion is subtle (< 48px), lock steadily to median
    if range_x < 48 and range_y < 48:
        kps_x = [(smoothed_pts[0][0], median_x), (smoothed_pts[-1][0], median_x)]
        kps_y = [(smoothed_pts[0][0], median_y), (smoothed_pts[-1][0], median_y)]
    else:
        # Build continuous keyframes separated by at least 0.5s and 28px intentional movement
        kps_x = [(smoothed_pts[0][0], smoothed_pts[0][1])]
        kps_y = [(smoothed_pts[0][0], smoothed_pts[0][2])]

        for pt_t, p_x, p_y in smoothed_pts[1:]:
            if abs(p_x - kps_x[-1][1]) >= 28 and (pt_t - kps_x[-1][0]) >= 0.45:
                kps_x.append((pt_t, p_x))
            if abs(p_y - kps_y[-1][1]) >= 28 and (pt_t - kps_y[-1][0]) >= 0.45:
                kps_y.append((pt_t, p_y))

        last_pt = smoothed_pts[-1]
        if kps_x[-1][0] < last_pt[0]:
            kps_x.append((last_pt[0], last_pt[1]))
        if kps_y[-1][0] < last_pt[0]:
            kps_y.append((last_pt[0], last_pt[2]))

        # Decimate if excessive nodes to keep FFmpeg expression clean and fast
        if len(kps_x) > 10:
            step = max(1, len(kps_x) // 10)
            kps_x = [kps_x[idx] for idx in range(0, len(kps_x), step)]
            if kps_x[-1] != (last_pt[0], last_pt[1]):
                kps_x.append((last_pt[0], last_pt[1]))

        if len(kps_y) > 10:
            step = max(1, len(kps_y) // 10)
            kps_y = [kps_y[idx] for idx in range(0, len(kps_y), step)]
            if kps_y[-1] != (last_pt[0], last_pt[2]):
                kps_y.append((last_pt[0], last_pt[2]))

    # 4. Construct continuous, mathematically seamless piecewise follow expression (ZERO jump cuts)
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

    crop_expression = _build_piecewise(kps_x)
    crop_y_expression = _build_piecewise(kps_y)

    static_crop_x = median_x
    static_crop_y = median_y
    shot_segments = [(0.0, smoothed_pts[-1][0], median_x, median_y)]

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
