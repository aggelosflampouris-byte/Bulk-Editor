"""
services/content_decision_engine.py — Autopilot Production Strategy Classifier.

Analyzes candidate video content and speaker presence to decide dynamically
between:
  - Mode 3: "speaker" (On-camera speaker present: retain footage, mask old subtitles, add B-roll cutaways)
  - Mode 2: "ai_gen"  (No on-camera speaker / graphic slides: generate full AI short from scratch with Greek voiceover)
"""

from __future__ import annotations

import logging
from pathlib import Path

try:
    from services.face_tracker import track_active_speaker
    from services.video_engine import probe_resolution
except ImportError:
    from shorts_engine.services.face_tracker import track_active_speaker
    from shorts_engine.services.video_engine import probe_resolution

logger = logging.getLogger(__name__)

# Minimum fraction of sampled frames containing a clear speaker face
_SPEAKER_PRESENCE_THRESHOLD = 0.20


def classify_production_mode(
    video_path: Path,
    user_preference: str = "auto",
) -> str:
    """
    Classify whether to produce video via 'speaker' preservation or 'ai_gen' synthesis.

    Args:
        video_path:      Local video file to inspect.
        user_preference: 'auto' | 'speaker' | 'ai_gen'.

    Returns:
        'speaker' or 'ai_gen'.
    """
    pref = user_preference.lower().strip()
    if pref in ("speaker", "ai_gen", "hybrid"):
        logger.info("Using explicit user production strategy: '%s'", pref)
        return pref

    if not video_path.is_file():
        logger.warning("Video file not found for classification (%s) — defaulting to speaker.", video_path)
        return "speaker"

    try:
        w, h = probe_resolution(video_path)
        tracking_info = track_active_speaker(video_path, source_width=w, source_height=h)
        presence = tracking_info.speaker_presence_ratio
        logger.info("Content Classifier: speaker_presence_ratio=%.2f, has_speaker=%s", presence, tracking_info.has_speaker)

        if tracking_info.has_speaker and presence >= _SPEAKER_PRESENCE_THRESHOLD:
            logger.info("Detected authentic on-camera speaker (%.0f%% presence) -> Choosing Mode 3 (Speaker Preservation).", presence * 100)
            return "speaker"
        else:
            logger.info("No dominant on-camera speaker detected (%.0f%% presence) -> Choosing Mode 2 (Full AI Generation).", presence * 100)
            return "ai_gen"
    except (RuntimeError, OSError, ValueError) as exc:
        logger.warning("Speaker classification analysis failed (%s) — defaulting to speaker mode.", exc)
        return "speaker"
