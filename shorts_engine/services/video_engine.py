"""
services/video_engine.py — FFmpeg composition primitives.

Responsibilities:
  1. Provide a fail-fast subprocess wrapper for all FFmpeg calls.
  2. Crop any input video to 9:16 (1080×1920) via center-crop.
  3. Overlay a B-roll clip on the upper half of the main clip for a timed window.
  4. Burn ASS subtitles with Greek-safe font directory injection.
  5. Concatenate the processed Short with an optional outro bumper.

This module has zero transcription, HTTP, or AI dependencies.
All paths passed to FFmpeg filters are safely escaped to prevent
filter_complex syntax breakage on all platforms.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Custom Exception ───────────────────────────────────────────────────────────


class FFmpegError(RuntimeError):
    """
    Raised when an FFmpeg subprocess exits with a non-zero return code.

    Carries the full stderr log so callers can surface it in the UI.
    """

    def __init__(self, command: list[str], returncode: int, stderr: str) -> None:
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        cmd_str = " ".join(command[:6]) + (" ..." if len(command) > 6 else "")
        super().__init__(
            f"FFmpeg exited with code {returncode}.\n"
            f"Command: {cmd_str}\n"
            f"Stderr:\n{stderr}"
        )


# ── Subprocess Wrapper ─────────────────────────────────────────────────────────


def run_ffmpeg(args: list[str]) -> None:
    """
    Execute an FFmpeg command, capturing stderr for error reporting.

    Args:
        args: Full argument list, starting with 'ffmpeg'.

    Raises:
        FFmpegError: On non-zero exit code, with formatted stderr.
    """
    logger.debug("FFmpeg command: %s", " ".join(args))
    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise FFmpegError(args, result.returncode, result.stderr)
    logger.debug("FFmpeg completed successfully.")


# ── Path Escaping ──────────────────────────────────────────────────────────────


def _escape_filter_path(path: Path) -> str:
    """
    Escape a filesystem path for safe embedding inside an FFmpeg filter_complex
    or filter option string.

    FFmpeg filter options use ':' as a key-value separator and '\\' as an
    escape character. Colons and backslashes in paths must be escaped.
    On Windows, drive letter colons (C:\\) would otherwise break the parser.

    Returns a string with backslashes and colons escaped for FFmpeg.
    """
    path_str = str(path.resolve())
    # On Windows, convert backslashes to forward slashes first (FFmpeg accepts both)
    if sys.platform == "win32":
        path_str = path_str.replace("\\", "/")
    # Escape colons (drive letter colon on Windows, and any colon in Unix paths)
    path_str = path_str.replace(":", "\\:")
    # Escape single-quotes which delimit filter values in libavfilter
    path_str = path_str.replace("'", "\\'")
    return path_str


# ── Video Processing Primitives ────────────────────────────────────────────────


def probe_duration(video_path: Path) -> float:
    """
    Query the duration of a video file using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds as a float.

    Raises:
        FFmpegError: If ffprobe fails.
        ValueError:  If the duration cannot be parsed.
    """
    args = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise FFmpegError(args, result.returncode, result.stderr)

    raw = result.stdout.strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse duration from ffprobe output: '{raw}'"
        ) from exc


def crop_to_9_16(
    input_path: Path,
    output_path: Path,
    target_width: int = 1080,
    target_height: int = 1920,
) -> Path:
    """
    Center-crop and scale *input_path* to exactly *target_width* × *target_height* (9:16).

    The filter chain:
      1. `scale` — scale so the shortest dimension fits, preserving aspect ratio.
      2. `crop`  — center-crop to the exact target dimensions.
      3. `setsar=1` — fix the Sample Aspect Ratio to 1:1 (square pixels).

    Args:
        input_path:    Source video path.
        output_path:   Destination path for the cropped video.
        target_width:  Output width in pixels (default 1080).
        target_height: Output height in pixels (default 1920).

    Returns:
        The written *output_path*.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        FFmpegError:       If FFmpeg fails.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # Scale so the video covers the target rectangle, then center-crop.
    # The scale formula: scale to w=target_w if width is the limiting axis,
    # else scale to h=target_h, then crop. The -2 ensures even dimensions.
    vf = (
        f"scale=w={target_width}:h={target_height}:force_original_aspect_ratio=increase,"
        f"crop={target_width}:{target_height}:(iw-{target_width})/2:(ih-{target_height})/2,"
        f"setsar=1"
    )

    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ])

    logger.info("Cropped '%s' → '%s'.", input_path.name, output_path.name)
    return output_path


def overlay_broll(
    main_path: Path,
    broll_path: Path,
    start_time: float,
    overlay_duration: float,
    output_path: Path,
    target_width: int = 1080,
    target_height: int = 1920,
) -> Path:
    """
    Overlay a B-roll clip on the upper half of the main video for a timed window.

    The B-roll is scaled to fill the top half (target_width × target_height/2),
    and composited over the main video using an `enable='between(t,...)'` gate
    so it only appears during [start_time, start_time + overlay_duration].

    Args:
        main_path:        The 9:16 main video (already cropped).
        broll_path:       The downloaded B-roll clip.
        start_time:       Seconds from the beginning of the main video at
                          which the B-roll overlay begins.
        overlay_duration: Duration (seconds) the B-roll is visible.
        output_path:      Destination path for the composited video.
        target_width:     Width of the main video (default 1080).
        target_height:    Height of the main video (default 1920).

    Returns:
        The written *output_path*.

    Raises:
        FileNotFoundError: If either input path does not exist.
        FFmpegError:       If FFmpeg fails.
    """
    for path in (main_path, broll_path):
        if not path.is_file():
            raise FileNotFoundError(f"Input video not found: {path}")

    overlay_height = target_height // 2  # Top 50% of the frame
    end_time = start_time + overlay_duration

    # filter_complex breakdown:
    #   [1:v] scale — resize B-roll to fill the top half of the frame
    #   [0:v][broll_scaled] overlay — composite at (0, 0) with time gate
    filter_complex = (
        f"[1:v]scale={target_width}:{overlay_height},"
        f"setsar=1[broll_scaled];"
        f"[0:v][broll_scaled]overlay=0:0:"
        f"enable='between(t,{start_time:.3f},{end_time:.3f})'[v_out]"
    )

    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(main_path),
        "-i", str(broll_path),
        "-filter_complex", filter_complex,
        "-map", "[v_out]",
        "-map", "0:a?",  # Preserve original audio; '?' = optional (no audio = skip)
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ])

    logger.info(
        "B-roll overlay applied: %.1fs–%.1fs on '%s' → '%s'.",
        start_time, end_time, main_path.name, output_path.name,
    )
    return output_path


def burn_subtitles(
    input_path: Path,
    ass_path: Path,
    output_path: Path,
) -> Path:
    """
    Burn ASS subtitles into *input_path* using the FFmpeg subtitles filter.

    The `fontsdir` parameter points FFmpeg's libass renderer to the
    system font directory. On Linux/macOS, `/usr/share/fonts` is used;
    on Windows, `C:/Windows/Fonts` is used. This ensures Arial (Greek-capable)
    is found regardless of platform.

    Paths are escaped via `_escape_filter_path` to prevent filter_complex
    syntax breakage on all platforms (especially Windows drive letters).

    Args:
        input_path:  Video path (with or without B-roll overlay).
        ass_path:    Path to the generated .ass subtitle file.
        output_path: Destination path for the subtitle-burned video.

    Returns:
        The written *output_path*.

    Raises:
        FileNotFoundError: If either input path does not exist.
        FFmpegError:       If FFmpeg fails.
    """
    for path in (input_path, ass_path):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    # Determine platform font directory for Greek glyph coverage
    if sys.platform == "win32":
        fonts_dir = Path("C:/Windows/Fonts")
    elif sys.platform == "darwin":
        fonts_dir = Path("/Library/Fonts")
    else:
        fonts_dir = Path("/usr/share/fonts")

    escaped_ass = _escape_filter_path(ass_path)
    escaped_fonts = _escape_filter_path(fonts_dir)

    # Build the subtitles filter with fontsdir
    vf = f"subtitles='{escaped_ass}':fontsdir='{escaped_fonts}'"

    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",  # Audio is already encoded; avoid re-encoding
        "-movflags", "+faststart",
        str(output_path),
    ])

    logger.info("Subtitles burned into '%s' → '%s'.", input_path.name, output_path.name)
    return output_path


def concatenate_with_outro(
    main_path: Path,
    outro_path: Path,
    output_path: Path,
    target_width: int = 1080,
    target_height: int = 1920,
) -> Path:
    """
    Concatenate *main_path* and *outro_path* into a single output video.

    Both clips are normalised to *target_width* × *target_height* with
    consistent pixel format (yuv420p), frame rate (30fps), sample rate
    (44100 Hz), and stereo channels before concatenation, so that the
    `concat` demuxer never encounters mismatched stream parameters.

    Args:
        main_path:     The subtitle-burned Short video.
        outro_path:    The outro bumper clip (any resolution/codec accepted).
        output_path:   Destination path for the final concatenated video.
        target_width:  Normalisation width (default 1080).
        target_height: Normalisation height (default 1920).

    Returns:
        The written *output_path*.

    Raises:
        FileNotFoundError: If either input path does not exist.
        FFmpegError:       If FFmpeg fails.
    """
    for path in (main_path, outro_path):
        if not path.is_file():
            raise FileNotFoundError(f"Input video not found: {path}")

    # Normalise both streams to identical parameters, then concat.
    # The scale2ref approach is not needed here because we have a fixed
    # target resolution. We use a filter_complex with two normalisation
    # chains followed by a [v][a]concat.
    norm_vf = (
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1,fps=30,format=yuv420p"
    )
    norm_af = "aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo"

    filter_complex = (
        f"[0:v]{norm_vf}[v0];[0:a]{norm_af}[a0];"
        f"[1:v]{norm_vf}[v1];[1:a]{norm_af}[a1];"
        f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[v_out][a_out]"
    )

    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(main_path),
        "-i", str(outro_path),
        "-filter_complex", filter_complex,
        "-map", "[v_out]",
        "-map", "[a_out]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ])

    logger.info(
        "Concatenated '%s' + outro → '%s'.",
        main_path.name, output_path.name,
    )
    return output_path
