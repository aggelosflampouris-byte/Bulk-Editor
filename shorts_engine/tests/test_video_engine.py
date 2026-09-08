"""
tests/test_video_engine.py — Unit tests for video_engine primitives.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from shorts_engine.config import Settings
from shorts_engine.services.video_engine import (
    concatenate_with_outro,
    crop_to_9_16,
    is_already_9_16,
    mix_background_music,
    overlay_broll,
    probe_duration,
    probe_has_audio,
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


def test_probe_has_audio_and_concatenate_silent_outro(test_videos, tmp_path):
    # 1. Create a silent synthetic video without audio track
    silent_outro = tmp_path / "silent_outro.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=red:s=1080x1920:d=1.5",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        str(silent_outro),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0

    assert probe_has_audio(test_videos["main_1080"]) is True
    assert probe_has_audio(silent_outro) is False

    # 2. Concat main (with audio) + silent outro (no audio)
    out_path = tmp_path / "final_with_silent_outro.mp4"
    concat_result = concatenate_with_outro(
        main_path=test_videos["main_1080"],
        outro_path=silent_outro,
        output_path=out_path,
        target_width=1080,
        target_height=1920,
    )
    assert concat_result.is_file()
    assert probe_has_audio(concat_result) is True
    dur = probe_duration(concat_result)
    # 3.0s main + 1.5s outro = ~4.5s
    assert 4.3 <= dur <= 4.7


def test_overlay_broll_fade_transition(test_videos, tmp_path):
    output_path = tmp_path / "out_broll_fade.mp4"
    res_path = overlay_broll(
        main_path=test_videos["main_720"],
        broll_path=test_videos["broll_landscape"],
        start_time=0.5,
        overlay_duration=1.5,
        output_path=output_path,
        transition="fade",
        transition_duration=0.35,
    )
    assert res_path.is_file()
    w, h = probe_resolution(res_path)
    assert (w, h) == (720, 1280)
    dur = probe_duration(res_path)
    assert 2.9 <= dur <= 3.1


def test_overlay_broll_flash_transition(test_videos, tmp_path):
    output_path = tmp_path / "out_broll_flash.mp4"
    res_path = overlay_broll(
        main_path=test_videos["main_1080"],
        broll_path=test_videos["broll_portrait"],
        start_time=0.5,
        overlay_duration=1.5,
        output_path=output_path,
        transition="flash",
        transition_duration=0.3,
    )
    assert res_path.is_file()
    w, h = probe_resolution(res_path)
    assert (w, h) == (1080, 1920)
    dur = probe_duration(res_path)
    assert 2.9 <= dur <= 3.1


def test_concatenate_with_outro_fade_transition(test_videos, tmp_path):
    silent_outro = tmp_path / "silent_outro_fade.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=red:s=1080x1920:d=1.5",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        str(silent_outro),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    out_path = tmp_path / "final_xfade.mp4"
    concat_result = concatenate_with_outro(
        main_path=test_videos["main_1080"],
        outro_path=silent_outro,
        output_path=out_path,
        target_width=1080,
        target_height=1920,
        transition="fade",
        transition_duration=0.5,
    )
    assert concat_result.is_file()
    assert probe_has_audio(concat_result) is True
    dur = probe_duration(concat_result)
    # 3.0s main + 1.5s outro - 0.5s xfade = ~4.0s
    assert 3.8 <= dur <= 4.2


def test_concatenate_with_outro_flash_transition(test_videos, tmp_path):
    silent_outro = tmp_path / "silent_outro_flash.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=red:s=1080x1920:d=1.5",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        str(silent_outro),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    out_path = tmp_path / "final_flash.mp4"
    concat_result = concatenate_with_outro(
        main_path=test_videos["main_1080"],
        outro_path=silent_outro,
        output_path=out_path,
        target_width=1080,
        target_height=1920,
        transition="flash",
        transition_duration=0.4,
    )
    assert concat_result.is_file()
    assert probe_has_audio(concat_result) is True
    dur = probe_duration(concat_result)
    # 3.0s main + 1.5s outro - 0.4s xfade = ~4.1s
    assert 3.9 <= dur <= 4.3


def test_mix_background_music_with_speech_and_ducking(test_videos, tmp_path):
    settings = Settings(enable_bg_music=True, bg_music_track="ambient_calm")
    music_path = settings.resolve_bg_music_path()
    assert music_path is not None and music_path.is_file()

    out_path = tmp_path / "out_mixed.mp4"
    res = mix_background_music(
        video_path=test_videos["main_720"],
        music_path=music_path,
        output_path=out_path,
        volume=0.10,
        ducking=True,
    )
    assert res.is_file()
    assert probe_has_audio(res) is True
    dur = probe_duration(res)
    assert 2.9 <= dur <= 3.1


def test_mix_background_music_silent_video(test_videos, tmp_path):
    # 1. Create a video without audio
    silent_vid = tmp_path / "silent.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=green:s=720x1280:d=2.0",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        str(silent_vid),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    assert probe_has_audio(silent_vid) is False

    settings = Settings(enable_bg_music=True, bg_music_track="dramatic_pulse")
    music_path = settings.resolve_bg_music_path()
    assert music_path is not None

    out_path = tmp_path / "out_silent_mixed.mp4"
    res = mix_background_music(
        video_path=silent_vid,
        music_path=music_path,
        output_path=out_path,
        volume=0.12,
        ducking=False,
    )
    assert res.is_file()
    assert probe_has_audio(res) is True
    dur = probe_duration(res)
    assert 1.9 <= dur <= 2.1


def test_settings_bg_music_resolution(tmp_path):
    # Default preset
    s1 = Settings(enable_bg_music=True, bg_music_track="ambient_calm")
    p1 = s1.resolve_bg_music_path()
    assert p1 is not None and p1.name == "ambient_calm.m4a"

    # Disabled
    s2 = Settings(enable_bg_music=False)
    assert s2.resolve_bg_music_path() is None

    # Custom track
    custom_track = tmp_path / "my_track.mp3"
    custom_track.write_bytes(b"dummy")
    s3 = Settings(enable_bg_music=True, bg_music_track="custom", bg_music_path=custom_track)
    assert s3.resolve_bg_music_path() == custom_track


def test_crop_to_9_16_aspect_ratio_and_pix_fmt(test_videos, tmp_path):
    out_cropped = tmp_path / "cropped_9_16.mp4"
    res = crop_to_9_16(
        input_path=test_videos["landscape_main"],
        output_path=out_cropped,
        target_width=1080,
        target_height=1920,
    )
    assert res.is_file()
    w, h = probe_resolution(res)
    assert (w, h) == (1080, 1920)
    assert abs((w / h) - (9 / 16)) < 1e-4

    # Verify pixel format is yuv420p for universal device compatibility
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=pix_fmt",
        "-of", "csv=p=0",
        str(res),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert p.stdout.strip() == "yuv420p"


def test_mix_background_music_mono_speech_preserves_stereo(tmp_path):
    mono_vid = tmp_path / "mono_speech.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=720x1280:d=2.5",
        "-f", "lavfi", "-i", "sine=f=350:d=2.7",
        "-ac", "1",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        str(mono_vid),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    settings = Settings(enable_bg_music=True, bg_music_track="ambient_calm")
    music_path = settings.resolve_bg_music_path()
    assert music_path is not None

    out_mixed = tmp_path / "mono_mixed.mp4"
    res = mix_background_music(
        video_path=mono_vid,
        music_path=music_path,
        output_path=out_mixed,
        volume=0.10,
        ducking=True,
    )
    assert res.is_file()
    # Verify channels == 2 (stereo) and sample_rate == 44100
    cmd_probe = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels,sample_rate",
        "-of", "csv=p=0",
        str(res),
    ]
    p_probe = subprocess.run(cmd_probe, capture_output=True, text=True, check=True)
    rate_str, channels_str = p_probe.stdout.strip().split(",")
    assert int(rate_str) == 44100
    assert int(channels_str) == 2

    # Verify duration is clamped to video duration (2.5s), not the trailing audio (2.7s)
    dur = probe_duration(res)
    assert 2.4 <= dur <= 2.6

