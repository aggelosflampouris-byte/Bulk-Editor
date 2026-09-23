"""
services/hw_encoder.py — Hardware-accelerated FFmpeg encoder selection.

Probes the current system for available GPU encoders and returns the best
available codec tuple for video encoding. Probe results are cached in-process
to avoid repeated FFmpeg subprocess calls on every encode.

Priority order:
  1. h264_amf  — AMD AMF (Windows/AMD GPU, e.g. RX 6700 / RDNA)
  2. h264_vaapi — VA-API (Linux/AMD or Intel iGPU)
  3. libx264   — Software fallback (always available)
"""
from __future__ import annotations

import logging
import subprocess
import sys
from functools import lru_cache
from typing import Literal

logger = logging.getLogger(__name__)

EncoderName = Literal["h264_amf", "h264_qsv", "h264_vaapi", "libx264"]

_AMF_QP_MAP: dict[int, int] = {18: 18, 20: 20, 23: 24, 25: 26, 28: 28, 32: 32}
_QSV_QP_MAP: dict[int, int] = {18: 18, 20: 20, 23: 23, 25: 25, 28: 28, 32: 32}
_VAAPI_QP_MAP: dict[int, int] = {18: 18, 20: 20, 23: 23, 25: 25, 28: 28, 32: 32}


def _probe_encoder(encoder: str) -> bool:
    """Return True if FFmpeg was compiled with *encoder* support."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return encoder in result.stdout
    except FileNotFoundError:
        return False
    except Exception as exc:
        logger.debug("Encoder probe failed for '%s': %s", encoder, exc)
        return False


def _probe_qsv_device() -> bool:
    """
    Check whether Intel Quick Sync Video (QSV) is functional by running a
    minimal dummy encode.
    """
    if not _probe_encoder("h264_qsv"):
        return False
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=black:s=16x16:d=0.1",
                "-c:v", "h264_qsv",
                "-preset", "faster",
                "-global_quality", "24",
                "-f", "null", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.debug("QSV functional probe failed: %s", exc)
        return False


def _probe_vaapi_device() -> bool:
    """
    Check that a VA-API render node actually exists and can encode.
    VA-API requires a physical /dev/dri/renderD* device; probing the encoder
    list alone is insufficient — FFmpeg may be built with VA-API support but
    no hardware available.
    """
    from pathlib import Path

    if sys.platform == "win32":
        return False

    dri_path = Path("/dev/dri")
    if not dri_path.exists():
        return False

    render_nodes = list(dri_path.glob("renderD*"))
    if not render_nodes:
        return False

    test_device = str(render_nodes[0])
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-hwaccel", "vaapi",
                "-hwaccel_device", test_device,
                "-f", "lavfi", "-i", "color=black:s=16x16:d=0.1",
                "-vf", "format=nv12,hwupload",
                "-c:v", "h264_vaapi",
                "-qp", "24",
                "-f", "null", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.debug("VA-API functional probe failed: %s", exc)
        return False


@lru_cache(maxsize=1)
def _detect_best_encoder() -> tuple[EncoderName, str | None]:
    """
    Detect the best available encoder once and cache the result.

    Returns:
        (encoder_name, vaapi_device_path | None)
    """
    if sys.platform == "win32":
        if _probe_encoder("h264_amf"):
            logger.info("Hardware encoder selected: h264_amf (AMD AMF, Windows)")
            return "h264_amf", None
        if _probe_qsv_device():
            logger.info("Hardware encoder selected: h264_qsv (Intel Quick Sync, Windows)")
            return "h264_qsv", None
    else:
        if _probe_encoder("h264_vaapi"):
            from pathlib import Path

            dri_path = Path("/dev/dri")
            render_nodes = list(dri_path.glob("renderD*")) if dri_path.exists() else []
            device = str(render_nodes[0]) if render_nodes else "/dev/dri/renderD128"
            if _probe_vaapi_device():
                logger.info(
                    "Hardware encoder selected: h264_vaapi (VA-API, Linux) device=%s", device
                )
                return "h264_vaapi", device
        if _probe_qsv_device():
            logger.info("Hardware encoder selected: h264_qsv (Intel Quick Sync, Linux)")
            return "h264_qsv", None

    logger.info("Hardware encoder selected: libx264 (software fallback)")
    return "libx264", None


def get_encoder_name() -> EncoderName:
    """Return just the codec name string."""
    name, _ = _detect_best_encoder()
    return name


def get_encoder_args(crf_equivalent: int = 23) -> list[str]:
    """
    Return the FFmpeg video-encoder argument block for the best available encoder.

    The returned list is a drop-in replacement for:
        ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]

    Args:
        crf_equivalent: Target quality level (analogous to libx264 CRF).

    Returns:
        List of FFmpeg arguments covering codec, quality, and preset/speed options.
    """
    encoder, _vaapi_device = _detect_best_encoder()

    if encoder == "h264_amf":
        qp = _AMF_QP_MAP.get(crf_equivalent, crf_equivalent + 1)
        return [
            "-c:v", "h264_amf",
            "-quality", "speed",
            "-rc", "cqp",
            "-qp_i", str(qp),
            "-qp_p", str(qp),
            "-qp_b", str(qp + 2),
        ]

    if encoder == "h264_qsv":
        qp = _QSV_QP_MAP.get(crf_equivalent, crf_equivalent)
        return [
            "-c:v", "h264_qsv",
            "-preset", "faster",
            "-global_quality", str(qp),
        ]

    if encoder == "h264_vaapi":
        qp = _VAAPI_QP_MAP.get(crf_equivalent, crf_equivalent)
        return [
            "-c:v", "h264_vaapi",
            "-qp", str(qp),
        ]

    return [
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", str(crf_equivalent),
    ]


def get_pix_fmt_args() -> list[str]:
    """
    Return the appropriate pixel format arguments for the active encoder.
    - libx264: -pix_fmt yuv420p (maximum player compatibility)
    - h264_amf: -pix_fmt yuv420p
    - h264_qsv: -pix_fmt nv12 (required by Intel QSV)
    - h264_vaapi: -pix_fmt vaapi (hardware frames)
    """
    encoder, _ = _detect_best_encoder()
    if encoder == "h264_qsv":
        return ["-pix_fmt", "nv12"]
    if encoder == "h264_vaapi":
        return ["-pix_fmt", "vaapi"]
    return ["-pix_fmt", "yuv420p"]


def get_optimal_threads() -> int:
    """
    Calculate the optimal thread count for FFmpeg/video processing.

    Prevents thermal throttling and fan screaming on laptops (e.g. ThinkPad)
    by preserving thermal headroom and keeping cores available for the OS and UI.
    """
    import os

    cores = os.cpu_count() or 4
    # For laptops, use 75% of logical cores, capped at 8 to prevent thermal throttling
    return max(1, min(8, int(cores * 0.75)))


def get_thread_args() -> list[str]:
    """Return FFmpeg thread-limiting arguments."""
    return ["-threads", str(get_optimal_threads())]


def get_hwaccel_input_args() -> list[str]:
    """
    Return FFmpeg input-side hardware-acceleration arguments (before -i).

    For VA-API these must appear before the input file to enable hardware
    upload via the hwaccel pipeline. For AMF, QSV, and software they are empty.
    """
    encoder, vaapi_device = _detect_best_encoder()
    if encoder == "h264_vaapi" and vaapi_device:
        return [
            "-hwaccel", "vaapi",
            "-hwaccel_device", vaapi_device,
            "-hwaccel_output_format", "vaapi",
        ]
    return []


def get_vf_hwupload_prefix() -> str:
    """
    Return an FFmpeg vf/filter_complex prefix string required to upload frames
    to VA-API hardware. For other encoders returns an empty string.
    """
    encoder, _ = _detect_best_encoder()
    if encoder == "h264_vaapi":
        return "format=nv12,hwupload,"
    return ""
