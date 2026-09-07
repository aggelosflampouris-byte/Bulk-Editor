"""
tests/test_face_tracker.py — Unit tests for active speaker face tracking and dynamic cropping.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from shorts_engine.services.face_tracker import calculate_active_speaker_crop_x
from shorts_engine.services.video_engine import crop_to_9_16, probe_resolution


def _generate_synthetic_video(path: Path, width: int, height: int, duration: float = 2.0) -> Path:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=red:s={width}x{height}:d={duration}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"FFmpeg failed: {res.stderr}"
    return path


def test_face_tracker_landscape_fallback_center_crop(tmp_path):
    # Plain solid red landscape video (1920x1080) — no human faces present
    landscape_video = _generate_synthetic_video(tmp_path / "landscape.mp4", 1920, 1080, duration=1.0)
    crop_x = calculate_active_speaker_crop_x(
        video_path=landscape_video,
        source_width=1920,
        source_height=1080,
        target_width=1080,
        target_height=1920,
    )

    assert isinstance(crop_x, int)
    # Scaled width = 1920 * (1920 / 1080) = 3413 px
    # Max crop X = 3413 - 1080 = 2333 px
    # Center crop offset = 2333 // 2 = 1166 px
    scale = 1920 / 1080
    scaled_w = int(1920 * scale)
    max_crop_x = scaled_w - 1080
    center_x = max_crop_x // 2

    assert 0 <= crop_x <= max_crop_x
    assert abs(crop_x - center_x) <= 2


def test_crop_to_9_16_with_dynamic_crop_offset(tmp_path):
    landscape_video = _generate_synthetic_video(tmp_path / "landscape_input.mp4", 1920, 1080, duration=1.0)
    output_video = tmp_path / "cropped_short.mp4"

    # Crop with explicit speaker offset
    result = crop_to_9_16(landscape_video, output_video, crop_x_offset=300)
    assert result.is_file()
    w, h = probe_resolution(result)
    assert (w, h) == (1080, 1920)
