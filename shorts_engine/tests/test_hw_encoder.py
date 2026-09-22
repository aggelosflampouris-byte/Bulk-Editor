"""
tests/test_hw_encoder.py — Unit tests for hardware-accelerated encoder detection and tuning.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from shorts_engine.services.hw_encoder import (
    get_encoder_args,
    get_hwaccel_input_args,
    get_optimal_threads,
    get_pix_fmt_args,
    get_thread_args,
)
from shorts_engine.services.video_engine import run_ffmpeg


def test_get_optimal_threads_bounds():
    """Verify optimal thread calculation is bounded and safe for laptops."""
    threads = get_optimal_threads()
    assert isinstance(threads, int)
    assert 1 <= threads <= 8

    # Simulate 16 cores (laptop with 8P+8E or high hyperthreads)
    with patch("os.cpu_count", return_value=16):
        assert get_optimal_threads() == 8

    # Simulate 4 cores (quad-core laptop)
    with patch("os.cpu_count", return_value=4):
        assert get_optimal_threads() == 3

    # Simulate 2 cores
    with patch("os.cpu_count", return_value=2):
        assert get_optimal_threads() == 1


def test_get_thread_args_format():
    """Verify thread arguments formatting."""
    args = get_thread_args()
    assert len(args) == 2
    assert args[0] == "-threads"
    assert args[1].isdigit()
    assert int(args[1]) >= 1


def test_get_pix_fmt_args():
    """Verify pixel format args are valid lists with -pix_fmt."""
    pix_fmt = get_pix_fmt_args()
    assert len(pix_fmt) == 2
    assert pix_fmt[0] == "-pix_fmt"
    assert pix_fmt[1] in ("yuv420p", "nv12", "vaapi")


def test_get_encoder_args_fallback():
    """Verify encoder arguments list always contains valid codec declaration."""
    args = get_encoder_args(crf_equivalent=23)
    assert "-c:v" in args
    codec_idx = args.index("-c:v") + 1
    assert args[codec_idx] in ("libx264", "h264_amf", "h264_qsv", "h264_vaapi")


def test_qsv_encoder_args_mocked():
    """Verify Intel Quick Sync parameter generation."""
    with patch("shorts_engine.services.hw_encoder._detect_best_encoder", return_value=("h264_qsv", None)):
        args = get_encoder_args(crf_equivalent=23)
        assert args == ["-c:v", "h264_qsv", "-preset", "faster", "-global_quality", "23"]

        pix_args = get_pix_fmt_args()
        assert pix_args == ["-pix_fmt", "nv12"]

        hw_in = get_hwaccel_input_args()
        assert hw_in == []


def test_amf_encoder_args_mocked():
    """Verify AMD AMF parameter generation."""
    with patch("shorts_engine.services.hw_encoder._detect_best_encoder", return_value=("h264_amf", None)):
        args = get_encoder_args(crf_equivalent=23)
        assert "-c:v" in args
        assert "h264_amf" in args
        assert "-quality" in args
        assert "speed" in args

        pix_args = get_pix_fmt_args()
        assert pix_args == ["-pix_fmt", "yuv420p"]


def test_vaapi_encoder_args_mocked():
    """Verify VA-API parameter generation."""
    with patch("shorts_engine.services.hw_encoder._detect_best_encoder", return_value=("h264_vaapi", "/dev/dri/renderD128")):
        args = get_encoder_args(crf_equivalent=23)
        assert args == ["-c:v", "h264_vaapi", "-qp", "23"]

        pix_args = get_pix_fmt_args()
        assert pix_args == ["-pix_fmt", "vaapi"]

        hw_in = get_hwaccel_input_args()
        assert "-hwaccel" in hw_in
        assert "-hwaccel_device" in hw_in
        assert "/dev/dri/renderD128" in hw_in


def test_run_ffmpeg_thread_injection():
    """Verify run_ffmpeg injects -threads if omitted and preserves it if present."""
    captured_commands: list[list[str]] = []

    def fake_subprocess_run(cmd, **kwargs):
        captured_commands.append(cmd)
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = ""
        mock_res.stderr = ""
        return mock_res

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        # Call without -threads
        run_ffmpeg(["ffmpeg", "-y", "-i", "input.mp4", "output.mp4"])
        assert len(captured_commands) == 1
        assert captured_commands[0][0] == "ffmpeg"
        assert captured_commands[0][1] == "-threads"
        assert captured_commands[0][2].isdigit()

        # Call with explicit -threads
        run_ffmpeg(["ffmpeg", "-threads", "2", "-y", "-i", "input.mp4", "output.mp4"])
        assert len(captured_commands) == 2
        # -threads should not be duplicated
        assert captured_commands[1].count("-threads") == 1
        assert captured_commands[1][1] == "-threads"
        assert captured_commands[1][2] == "2"
