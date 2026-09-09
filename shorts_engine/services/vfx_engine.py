"""
services/vfx_engine.py — Transcript- and YOLO-driven VFX / colour grading.

Responsibilities:
  1. Sample the video with the existing YOLOv8n model to build a SceneAnalysis
     (person frame ratio, dominant object class).
  2. Analyse the transcript text for high-energy, cinematic, or suspense patterns.
  3. Combine both signals to select one of five VfxPreset values.
  4. Apply the chosen preset to the video via a single FFmpeg vf pass,
     stream-copying audio to avoid any quality loss.

All FFmpeg calls are routed through run_ffmpeg() from video_engine for consistent
error handling.  YOLO/cv2 dependencies are soft — if missing the module falls back
to the SUBTLE preset with a warning log.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Sample interval ────────────────────────────────────────────────────────────

# How many seconds between sampled frames for scene analysis.
# 1.0 fps is enough to categorise the scene without being slow.
_SCENE_SAMPLE_INTERVAL_SECONDS: float = 1.0

# Minimum ratio of person-detected frames to trigger warmth grade.
_PERSON_DOMINANT_THRESHOLD: float = 0.60

# ── Greek high-energy keyword sets ────────────────────────────────────────────

# Words that trigger a vibrance/punchy grade (excitement, shock, hype).
_HIGH_ENERGY_KEYWORDS: frozenset[str] = frozenset({
    "αδύνατο", "απίστευτο", "τρελό", "σοκ", "εκπληκτικό",
    "wow", "ναι", "τέλειο", "απόλυτο", "εντυπωσιακό",
    "ωραίο", "φοβερό", "κατάπληκτος", "πάρτι", "φωτιά",
    "εκρηκτικό", "μεγάλο", "υπέροχο", "ιστορία", "αλήθεια",
})

# Words that trigger a dramatic/suspense grade (tension, mystery, gravity).
_DRAMATIC_KEYWORDS: frozenset[str] = frozenset({
    "προσοχή", "κίνδυνος", "επικίνδυνο", "σοβαρό", "τραγωδία",
    "θάνατος", "κρίση", "πρόβλημα", "αποκάλυψη", "μυστικό",
    "αποτυχία", "απογοήτευση", "σκοτάδι", "φόβος", "τέλος",
    "χάθηκε", "δραματικό", "σκληρό", "άγριο",
})


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class SceneAnalysis:
    """
    Summary of what YOLO detected across the sampled frames of a video.

    Attributes:
        person_frame_ratio: Fraction of sampled frames in which at least one
            person was detected (0.0 - 1.0).
        detected_classes:   Set of COCO class names seen at least once.
        dominant_class:     The most frequently detected non-person class,
            or None if only persons were detected.
    """

    person_frame_ratio: float = 0.0
    detected_classes: set[str] = field(default_factory=set)
    dominant_class: Optional[str] = None


class VfxPreset(Enum):
    """
    Named VFX / colour-grading presets applied via FFmpeg vf chains.

    Each member stores the FFmpeg vf filter string as its value.
    """

    # Mild lift — always safe default.
    SUBTLE = "eq=contrast=1.03:saturation=1.10"

    # Punchy, saturated look for high-energy moments.
    VIBRANCE = (
        "eq=contrast=1.05:saturation=1.30:brightness=0.02,"
        "vignette=PI/4"
    )

    # Warm skin-tone grade for person-heavy clips.
    WARMTH = (
        "eq=contrast=1.04:saturation=1.15,"
        "colorbalance=rs=0.05:gs=0.02:bs=-0.08:rm=0.03:gm=0.00:bm=-0.05"
    )

    # Cool cinematic teal/shadow for action / object-scene clips.
    CINEMATIC = (
        "eq=contrast=1.10:saturation=0.85,"
        "colorbalance=rs=-0.05:gs=0.00:bs=0.08"
    )

    # Desaturated, high-contrast dramatic look for suspense.
    DRAMATIC = (
        "eq=contrast=1.15:saturation=0.75:gamma=0.95,"
        "vignette=PI/5"
    )


# ── Scene Analysis ─────────────────────────────────────────────────────────────


def analyse_scene_objects(
    video_path: Path,
    model_path: str = "yolov8n.pt",
    sample_interval: float = _SCENE_SAMPLE_INTERVAL_SECONDS,
) -> SceneAnalysis:
    """
    Run YOLOv8 on evenly sampled frames to characterise the video scene.

    Samples one frame per *sample_interval* seconds (default: 1 fps) and
    records which COCO classes appear.  Results are aggregated into a
    SceneAnalysis that the preset chooser uses for decision-making.

    Args:
        video_path:      Path to the video file.
        model_path:      Path or filename of the YOLO weights (auto-downloaded
                         if not found locally — same behaviour as face_tracker).
        sample_interval: Seconds between sampled frames.

    Returns:
        A SceneAnalysis instance.  All fields are 0 / empty if YOLO or cv2
        are unavailable, triggering the SUBTLE fallback in choose_vfx_preset.
    """
    analysis = SceneAnalysis()

    # Lazy imports so the module loads even without vision libraries.
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        logger.info(
            "VFX scene analysis skipped — vision libraries unavailable: %s", exc
        )
        return analysis

    try:
        model = YOLO(model_path)
    except Exception as exc:
        logger.warning("VFX: failed to load YOLO model '%s': %s", model_path, exc)
        return analysis

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("VFX: could not open '%s' for scene analysis.", video_path.name)
        return analysis

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(fps * sample_interval))

    total_sampled = 0
    person_frames = 0
    class_counts: dict[str, int] = {}
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                results = model(frame, verbose=False)
                boxes = results[0].boxes if results else None
                has_person = False

                if boxes and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = model.names.get(cls_id, str(cls_id))
                        analysis.detected_classes.add(cls_name)
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                        if cls_name == "person":
                            has_person = True

                if has_person:
                    person_frames += 1
                total_sampled += 1

            frame_idx += 1
    except Exception as exc:
        logger.warning("VFX scene analysis loop error: %s", exc)
    finally:
        cap.release()

    if total_sampled > 0:
        analysis.person_frame_ratio = person_frames / total_sampled

    # Find most-common non-person class as dominant context.
    non_person = {k: v for k, v in class_counts.items() if k != "person"}
    if non_person:
        analysis.dominant_class = max(non_person, key=non_person.__getitem__)

    logger.info(
        "VFX scene analysis: person_ratio=%.2f, detected=%s, dominant=%s",
        analysis.person_frame_ratio,
        analysis.detected_classes,
        analysis.dominant_class,
    )
    return analysis


# ── Preset Selection ───────────────────────────────────────────────────────────


def choose_vfx_preset(
    transcript_text: str,
    scene: SceneAnalysis,
) -> VfxPreset:
    """
    Choose the most appropriate VFX preset based on transcript + scene data.

    Decision priority (first matching rule wins):
      1. Transcript contains dramatic/suspense keywords  -> DRAMATIC
      2. Transcript contains high-energy/hype keywords   -> VIBRANCE
      3. YOLO: person detected in >=60% of frames        -> WARMTH
      4. YOLO: no person at all (action/object scene)    -> CINEMATIC
      5. Fallback                                        -> SUBTLE

    Args:
        transcript_text: Full corrected transcript of the clip.
        scene:           SceneAnalysis from analyse_scene_objects.

    Returns:
        A VfxPreset enum member.
    """
    # Normalise to lowercase for keyword matching (handles accented Greek).
    normalised = transcript_text.lower()

    # Split into tokens for whole-word matching where possible.
    tokens: set[str] = set(re.split(r"\W+", normalised))

    # Rule 1: Dramatic / suspense keywords.
    if tokens & _DRAMATIC_KEYWORDS:
        matched = tokens & _DRAMATIC_KEYWORDS
        logger.info("VFX preset -> DRAMATIC (matched: %s)", matched)
        return VfxPreset.DRAMATIC

    # Rule 2: High-energy / hype keywords.
    if tokens & _HIGH_ENERGY_KEYWORDS:
        matched = tokens & _HIGH_ENERGY_KEYWORDS
        logger.info("VFX preset -> VIBRANCE (matched: %s)", matched)
        return VfxPreset.VIBRANCE

    # Rule 3: Person-dominated clip.
    if scene.person_frame_ratio >= _PERSON_DOMINANT_THRESHOLD:
        logger.info(
            "VFX preset -> WARMTH (person_ratio=%.2f)", scene.person_frame_ratio
        )
        return VfxPreset.WARMTH

    # Rule 4: No person at all — likely action/nature/product.
    if scene.person_frame_ratio == 0.0 and scene.detected_classes:
        logger.info(
            "VFX preset -> CINEMATIC (no person, objects=%s)", scene.detected_classes
        )
        return VfxPreset.CINEMATIC

    logger.info("VFX preset -> SUBTLE (default fallback)")
    return VfxPreset.SUBTLE


# ── Application ────────────────────────────────────────────────────────────────


def apply_vfx(
    input_path: Path,
    output_path: Path,
    preset: VfxPreset,
) -> Path:
    """
    Apply a VFX / colour-grading preset to a video via a single FFmpeg pass.

    Video is re-encoded with libx264 (preset=fast, crf=23); audio is
    stream-copied so no audio quality is lost.  The filter chain is taken
    directly from *preset.value*.

    Args:
        input_path:  Source video (must exist).
        output_path: Destination path for the colour-graded video.
        preset:      The VfxPreset to apply.

    Returns:
        The written *output_path*.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        FFmpegError:       If FFmpeg fails.
    """
    # Import here to avoid any circular dependency at module level.
    try:
        from shorts_engine.services.video_engine import run_ffmpeg  # noqa: PLC0415
    except ModuleNotFoundError:
        from services.video_engine import run_ffmpeg  # noqa: PLC0415

    if not input_path.is_file():
        raise FileNotFoundError(f"VFX input not found: {input_path}")

    logger.info(
        "Applying VFX preset '%s' to '%s' -> '%s'.",
        preset.name, input_path.name, output_path.name,
    )

    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", preset.value,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",          # stream-copy audio — zero quality loss
        "-movflags", "+faststart",
        str(output_path),
    ])

    logger.info("VFX '%s' applied: '%s'.", preset.name, output_path.name)
    return output_path
