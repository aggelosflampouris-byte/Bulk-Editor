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


def probe_resolution(video_path: Path) -> tuple[int, int]:
    """
    Query the width and height of the first video stream using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        (width, height) in pixels.

    Raises:
        FFmpegError: If ffprobe fails.
        ValueError:  If dimensions cannot be parsed.
    """
    args = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
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
    # ffprobe outputs "WxH" with -of csv=s=x:p=0
    try:
        w_str, h_str = raw.split("x", 1)
        return int(w_str), int(h_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Could not parse resolution from ffprobe output: '{raw}'"
        ) from exc


def probe_has_audio(video_path: Path) -> bool:
    """
    Check if a video file contains at least one audio stream using ffprobe.

    Args:
        video_path: Path to the media file.

    Returns:
        True if an audio stream is detected, False otherwise.
    """
    args = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
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
        return False
    return bool(result.stdout.strip())


def is_already_9_16(
    video_path: Path,
    target_width: int = 1080,
    target_height: int = 1920,
    tolerance: float = 0.02,
) -> bool:
    """
    Return True if *video_path* is already at the target 9:16 resolution
    (within *tolerance* of the aspect ratio and within ±2 px of dimensions).

    Used to skip the crop stage when the input clip is already correctly sized,
    avoiding a re-encode that wastes time and slightly reduces quality.

    Args:
        video_path:    Video to inspect.
        target_width:  Expected width (default 1080).
        target_height: Expected height (default 1920).
        tolerance:     Allowed fractional deviation from the ideal aspect ratio.

    Returns:
        True if the crop can safely be skipped.
    """
    try:
        w, h = probe_resolution(video_path)
    except (FFmpegError, ValueError):
        # If probing fails we play it safe and let crop_to_9_16 run
        return False

    # Check exact match first (most common case for already-processed clips)
    if abs(w - target_width) <= 2 and abs(h - target_height) <= 2:
        return True

    # Check aspect ratio match (handles e.g. 720×1280 → already 9:16)
    if h == 0:
        return False
    actual_ratio = w / h
    target_ratio = target_width / target_height
    return abs(actual_ratio - target_ratio) / target_ratio <= tolerance


def crop_to_9_16(
    input_path: Path,
    output_path: Path,
    target_width: int = 1080,
    target_height: int = 1920,
    crop_x_offset: Optional[int] = None,
) -> Path:
    """
    Scale and crop *input_path* to exactly *target_width* × *target_height* (9:16).

    If *crop_x_offset* is provided, the horizontal crop is positioned at that
    pixel offset (used for active speaker tracking). Otherwise, center-crop is applied.

    The filter chain:
      1. `scale` — scale so the shortest dimension fits, preserving aspect ratio.
      2. `crop`  — crop to the exact target dimensions.
      3. `setsar=1` — fix the Sample Aspect Ratio to 1:1 (square pixels).

    Args:
        input_path:    Source video path.
        output_path:   Destination path for the cropped video.
        target_width:  Output width in pixels (default 1080).
        target_height: Output height in pixels (default 1920).
        crop_x_offset: Optional horizontal pixel offset for dynamic speaker framing.

    Returns:
        The written *output_path*.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        FFmpegError:       If FFmpeg fails.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    x_expr = str(crop_x_offset) if crop_x_offset is not None else f"(iw-{target_width})/2"

    vf = (
        f"scale=w={target_width}:h={target_height}:force_original_aspect_ratio=increase,"
        f"crop={target_width}:{target_height}:{x_expr}:(ih-{target_height})/2,"
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

    The B-roll is scaled and centre-cropped (CSS cover behaviour) to fill
    target_width × (target_height / 2) with no black bars and no distortion,
    then composited over the main video using an ``enable='between(t,...)'`` gate
    so it only appears during [start_time, start_time + overlay_duration].

    Cover-fill filter chain:
      1. scale=w=target_width:h=-2  — scale so width = target_width, height auto
         (aspect ratio preserved, even number guaranteed by -2)
      2. vflip/crop — if the scaled height is still less than overlay_height,
         scale with h=overlay_height:w=-2 instead; then crop to exact size.
      The ``scale2ref`` / iw/ih expressions handle both landscape and portrait
      source clips cleanly.

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

    # Determine canvas dimensions from main_path so the B-roll overlay matches
    # the exact resolution of the main video (e.g. 720×1280 or 1080×1920)
    try:
        main_w, main_h = probe_resolution(main_path)
    except (FFmpegError, ValueError):
        main_w, main_h = target_width, target_height

    tw = main_w if main_w % 2 == 0 else main_w - 1
    oh = main_h if main_h % 2 == 0 else main_h - 1

    end_time = start_time + overlay_duration

    # Cover-fill: scale so the clip fills tw × oh (full 9:16 frame) with no black bars
    # (CSS object-fit: cover equivalent).
    # Using max(tw/iw, oh/ih) scales both axes by the larger factor,
    # guaranteeing that width >= tw and height >= oh without distortion.
    # crop={tw}:{oh} trims the excess from the center.
    scale_expr = (
        f"scale="
        fr"w=iw*max({tw}/iw\,{oh}/ih):"
        fr"h=ih*max({tw}/iw\,{oh}/ih),"
        f"crop={tw}:{oh},"
        f"setsar=1"
    )

    filter_complex = (
        f"[1:v]{scale_expr}[broll_filled];"
        f"[0:v][broll_filled]overlay=0:0:eof_action=repeat:"
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

    main_has_audio = probe_has_audio(main_path)
    outro_has_audio = probe_has_audio(outro_path)

    # Normalise video streams to identical parameters, then concat.
    norm_vf = (
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1,fps=30,format=yuv420p"
    )
    norm_af = "aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo"

    filter_chains = [
        f"[0:v]{norm_vf}[v0]",
        f"[1:v]{norm_vf}[v1]",
    ]

    # Handle audio for main video
    if main_has_audio:
        filter_chains.append(f"[0:a]{norm_af}[a0]")
    else:
        main_dur = probe_duration(main_path)
        filter_chains.append(
            f"anullsrc=channel_layout=stereo:sample_rate=44100,atrim=duration={main_dur}[a0]"
        )

    # Handle audio for outro bumper (many outros have video only)
    if outro_has_audio:
        filter_chains.append(f"[1:a]{norm_af}[a1]")
    else:
        outro_dur = probe_duration(outro_path)
        filter_chains.append(
            f"anullsrc=channel_layout=stereo:sample_rate=44100,atrim=duration={outro_dur}[a1]"
        )

    filter_chains.append("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v_out][a_out]")
    filter_complex = ";".join(filter_chains)

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


def slice_video(
    source_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
) -> Path:
    """
    Extract a sub-clip from *source_path* between [start_time, end_time].

    Strategy:
      1. Attempt a fast stream-copy cut (-c copy). This is near-instant but
         may produce slightly inaccurate in/out points due to keyframe alignment.
      2. Verify the resulting duration. If it deviates from the expected duration
         by more than 0.5 seconds, fall back to a full re-encode for a precise cut.

    Args:
        source_path: Path to the source video (any codec).
        start_time:  Cut start time in seconds (relative to source origin).
        end_time:    Cut end time in seconds (relative to source origin).
        output_path: Destination path for the extracted clip.

    Returns:
        The written *output_path*.

    Raises:
        FileNotFoundError: If *source_path* does not exist.
        FFmpegError:       If both the stream-copy and re-encode attempts fail.
        ValueError:        If start_time >= end_time.
    """
    if not source_path.is_file():
        raise FileNotFoundError(f"Source video not found: {source_path}")

    if start_time >= end_time:
        raise ValueError(
            f"start_time ({start_time:.3f}) must be less than end_time ({end_time:.3f})"
        )

    expected_duration = end_time - start_time

    # ── Attempt 1: Fast stream copy ───────────────────────────────────────────
    logger.info(
        "Slicing '%s' [%.2f → %.2f] (%.1fs) via stream copy...",
        source_path.name, start_time, end_time, expected_duration,
    )
    try:
        run_ffmpeg([
            "ffmpeg", "-y",
            "-ss", f"{start_time:.3f}",
            "-to", f"{end_time:.3f}",
            "-i", str(source_path),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            str(output_path),
        ])

        # Verify the resulting duration
        actual_duration = probe_duration(output_path)
        if abs(actual_duration - expected_duration) <= 0.5:
            logger.info(
                "Stream-copy slice successful: actual=%.2fs, expected=%.2fs.",
                actual_duration, expected_duration,
            )
            return output_path

        logger.warning(
            "Stream-copy duration mismatch: actual=%.2fs vs expected=%.2fs — "
            "falling back to re-encode for precise cut.",
            actual_duration, expected_duration,
        )

    except FFmpegError as exc:
        logger.warning(
            "Stream-copy slice failed: %s — falling back to re-encode.", exc
        )

    # ── Attempt 2: Precise re-encode ──────────────────────────────────────────
    logger.info(
        "Re-encoding slice [%.2f → %.2f] with libx264 for precise cut...",
        start_time, end_time,
    )
    run_ffmpeg([
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-to", f"{end_time:.3f}",
        "-i", str(source_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        str(output_path),
    ])

    logger.info(
        "Re-encode slice complete: '%s' → '%s'.",
        source_path.name, output_path.name,
    )
    return output_path

