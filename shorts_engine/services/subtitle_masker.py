"""
services/subtitle_masker.py — Clean lower-third subtitle obliteration.

Applies a frosted glass blur plate and subtle dark scrim over the lower-third
area of 9:16 videos (y=1200..1640) where old burned-in teletext/captions exist,
preventing double-text collisions before new dynamic karaoke subtitles are burned.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Lower-third box coordinates tailored for 1080x1920 (9:16)
_DEFAULT_MASK_Y = 1200
_DEFAULT_MASK_H = 440
_DEFAULT_OPACITY = 0.68


def build_subtitle_mask_filter(
    mask_y: int = _DEFAULT_MASK_Y,
    mask_h: int = _DEFAULT_MASK_H,
    opacity: float = _DEFAULT_OPACITY,
) -> str:
    """
    Construct the FFmpeg complex filter string to blur the lower third and overlay a dark scrim.
    """
    return (
        f"[0:v]split[base][blur];"
        f"[blur]crop=iw:{mask_h}:0:{mask_y},boxblur=24:12[blurred];"
        f"[base][blurred]overlay=0:{mask_y}[vmasked];"
        f"[vmasked]drawbox=x=0:y={mask_y}:w=iw:h={mask_h}:color=black@{opacity:.2f}:t=fill[vout]"
    )


def mask_burned_in_subtitles(
    video_path: Path,
    output_path: Path,
    mask_y: int = _DEFAULT_MASK_Y,
    mask_h: int = _DEFAULT_MASK_H,
    opacity: float = _DEFAULT_OPACITY,
) -> Path:
    """
    Obscure burned-in subtitles in the lower third using a frosted glass blur plate
    combined with a semi-transparent dark vignette scrim.

    Args:
        video_path:  Source video (e.g. 1080x1920).
        output_path: Destination video path.
        mask_y:      Starting Y coordinate (default: 1200).
        mask_h:      Height of masked region (default: 440).
        opacity:     Dark scrim opacity (default: 0.68).

    Returns:
        output_path: Path to the masked video.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Build filter: split base, crop masked region, apply heavy blur, overlay back, apply dark scrim
    fc = build_subtitle_mask_filter(mask_y=mask_y, mask_h=mask_h, opacity=opacity)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-filter_complex", fc,
        "-map", "[vout]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "copy",
        str(output_path),
    ]

    logger.info("Masking lower-third burned-in subtitles on '%s'...", video_path.name)
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        logger.warning(
            "Subtitle masking failed (%s) — falling back to unmasked source.",
            res.stderr[:200] if res.stderr else "Unknown error",
        )
        import shutil
        shutil.copy2(str(video_path), str(output_path))
        return output_path

    return output_path
