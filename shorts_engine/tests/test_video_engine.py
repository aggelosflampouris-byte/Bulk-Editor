"""
tests/test_video_engine.py — Unit tests for video_engine primitives.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from shorts_engine.services.video_engine import (
    is_already_9_16,
    overlay_broll,
    probe_duration,
    probe_resolution,
)


def _generate_synthetic_video(path: Path, width: int, height: int, duration: float = 2.0) -> Path:
    """Create a lightweight synthetic MP4 video file using lavfi test patterns."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=blue:s={width}x{height}:d={duration}",
        "-f", "lavfi",
        "-i", f"sine=f=440:d={duration}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-b:a", "64k",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"FFmpeg failed to generate synthetic video: {res.stderr}"
    return path


@pytest.fixture(scope="session")
def test_videos(tmp_path_factory) -> dict[str, Path]:
    tmp_dir = tmp_path_factory.mktemp("test_media")
    main_720 = _generate_synthetic_video(tmp_dir / "main_720x1280.mp4", 720, 1280, duration=3.0)
    main_1080 = _generate_synthetic_video(tmp_dir / "main_1080x1920.mp4", 1080, 1920, duration=3.0)
    broll_landscape = _generate_synthetic_video(tmp_dir / "broll_1920x1080.mp4", 1920, 1080, duration=2.0)
    broll_portrait = _generate_synthetic_video(tmp_dir / "broll_1080x1920.mp4", 1080, 1920, duration=2.0)
    landscape_main = _generate_synthetic_video(tmp_dir / "landscape_1920x1080.mp4", 1920, 1080, duration=2.0)
    return {
        "main_720": main_720,
        "main_1080": main_1080,
        "broll_landscape": broll_landscape,
        "broll_portrait": broll_portrait,
        "landscape_main": landscape_main,
    }


def test_probe_resolution(test_videos):
    assert probe_resolution(test_videos["main_720"]) == (720, 1280)
    assert probe_resolution(test_videos["broll_landscape"]) == (1920, 1080)


def test_probe_duration(test_videos):
    dur = probe_duration(test_videos["main_720"])
    assert 2.9 <= dur <= 3.1


def test_is_already_9_16(test_videos):
    assert is_already_9_16(test_videos["main_720"]) is True
    assert is_already_9_16(test_videos["main_1080"]) is True
    assert is_already_9_16(test_videos["landscape_main"]) is False


def test_overlay_broll_720x1280_with_landscape_broll(test_videos, tmp_path):
    output_path = tmp_path / "out_720.mp4"
    res_path = overlay_broll(
        main_path=test_videos["main_720"],
        broll_path=test_videos["broll_landscape"],
        start_time=0.5,
        overlay_duration=1.5,
        output_path=output_path,
    )
    assert res_path.is_file()
    w, h = probe_resolution(res_path)
    assert (w, h) == (720, 1280)
    dur = probe_duration(res_path)
    assert 2.9 <= dur <= 3.1


def test_overlay_broll_1080x1920_with_portrait_broll(test_videos, tmp_path):
    output_path = tmp_path / "out_1080.mp4"
    res_path = overlay_broll(
        main_path=test_videos["main_1080"],
        broll_path=test_videos["broll_portrait"],
        start_time=0.0,
        overlay_duration=1.0,
        output_path=output_path,
    )
    assert res_path.is_file()
    w, h = probe_resolution(res_path)
    assert (w, h) == (1080, 1920)


def test_overlay_broll_short_broll_eof_repeat(test_videos, tmp_path):
    # Test case where overlay_duration exceeds the broll clip length
    output_path = tmp_path / "out_repeat.mp4"
    res_path = overlay_broll(
        main_path=test_videos["main_720"],
        broll_path=test_videos["broll_landscape"],  # 2.0s long
        start_time=0.0,
        overlay_duration=2.5,  # longer than 2.0s clip -> tests eof_action=repeat
        output_path=output_path,
    )
    assert res_path.is_file()
    w, h = probe_resolution(res_path)
    assert (w, h) == (720, 1280)
    dur = probe_duration(res_path)
    assert 2.9 <= dur <= 3.1

