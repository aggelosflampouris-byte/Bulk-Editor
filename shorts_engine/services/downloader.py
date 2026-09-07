"""
services/downloader.py — URL-based video ingestion via yt-dlp.

Responsibilities:
  1. Probe a URL for metadata (title, channel, duration) without downloading.
  2. Download the best available video quality ≤ 1080p into a caller-supplied
     scratch directory.
  3. Enforce a configurable maximum source duration to prevent runaway jobs.

This module has zero transcription, AI, or FFmpeg dependencies.
All subprocess calls target the `yt-dlp` CLI which must be on PATH or
installed as a Python package (which provides the `yt-dlp` entry-point).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Maximum allowed source video duration in seconds (2 hours).
# Passed in from Settings but also enforced here as a hard ceiling.
_HARD_MAX_DURATION_SECONDS: int = 7200

# yt-dlp format selector: best video+audio ≤ 1080p, prefer mp4 container.
# Falls back to best available if no match under 1080p.
_FORMAT_SELECTOR: str = (
    "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
    "bestvideo[height<=1080]+bestaudio/"
    "best[height<=1080]/"
    "best"
)


# ── Public Types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UrlMetadata:
    """Metadata extracted from a video URL before downloading."""

    title: str
    channel: str
    duration_seconds: float
    url: str

    @property
    def duration_display(self) -> str:
        """Human-readable duration string (HH:MM:SS or MM:SS)."""
        total = int(self.duration_seconds)
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


# ── Internal Helpers ───────────────────────────────────────────────────────────


def _resolve_ytdlp_bin() -> list[str]:
    """
    Return the yt-dlp command as a list of strings suitable for subprocess.

    Resolution order:
      1. 'yt-dlp' CLI on PATH (installed by pip as an entry-point).
      2. 'yt_dlp' alias (some platforms).
      3. [sys.executable, '-m', 'yt_dlp'] (works when the venv bin isn't on
         PATH, e.g. when a parent process didn't activate the venv).

    Returns a list so that paths with spaces are handled correctly.

    Raises:
        RuntimeError: If yt-dlp cannot be found by any method.
    """
    import sys as _sys

    # Prefer a direct binary on PATH
    for candidate in ("yt-dlp", "yt_dlp"):
        found = shutil.which(candidate)
        if found:
            return [found]

    # Fall back to invoking the module via the running Python interpreter.
    # Using sys.executable as a list element avoids any space-splitting issues.
    try:
        import importlib.util as _imp
        if _imp.find_spec("yt_dlp") is not None:
            return [_sys.executable, "-m", "yt_dlp"]
    except Exception:
        pass

    raise RuntimeError(
        "yt-dlp not found. Install it with: pip install yt-dlp"
    )


def _run_ytdlp(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """
    Execute a yt-dlp command and return the CompletedProcess.

    Args:
        args:    Complete argument list including the yt-dlp binary/module prefix.
                 Build this by concatenating _resolve_ytdlp_bin() + option flags.
        timeout: Maximum seconds to wait before raising TimeoutExpired.

    Raises:
        RuntimeError: On non-zero exit code, with formatted stderr.
        TimeoutError: If the process exceeds *timeout* seconds.
    """
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"yt-dlp timed out after {timeout}s for command: {' '.join(str(a) for a in args[:4])}"
        ) from exc

    if result.returncode != 0:
        # Surface the last 500 chars of stderr for concise diagnostics
        stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
        raise RuntimeError(
            f"yt-dlp exited with code {result.returncode}.\n"
            f"Command: {' '.join(str(a) for a in args[:6])} ...\n"
            f"Stderr: {stderr_tail}"
        )
    return result


# ── Public API ─────────────────────────────────────────────────────────────────


def probe_url_metadata(
    url: str,
    max_duration_seconds: int = _HARD_MAX_DURATION_SECONDS,
) -> UrlMetadata:
    """
    Extract video metadata from a URL without downloading the media.

    Uses `yt-dlp --dump-json --no-download` to query the platform API
    and return structured metadata.

    Args:
        url:                  The video URL (YouTube, Vimeo, direct MP4, etc.).
        max_duration_seconds: Reject videos longer than this duration.

    Returns:
        A populated UrlMetadata instance.

    Raises:
        ValueError:   If the URL is empty or the video exceeds max_duration_seconds.
        RuntimeError: If yt-dlp fails or returns malformed JSON.
    """
    url = url.strip()
    if not url:
        raise ValueError("URL must not be empty.")

    ytdlp = _resolve_ytdlp_bin()

    cmd = ytdlp + [
        "--dump-json",
        "--no-download",
        "--no-playlist",
        "--quiet",
        url,
    ]

    logger.info("Probing URL metadata: %s", url)
    result = _run_ytdlp(cmd, timeout=60)

    raw_json = result.stdout.strip()
    if not raw_json:
        raise RuntimeError(
            "yt-dlp returned empty output when probing URL. "
            "The URL may be private, geo-restricted, or unsupported."
        )

    # yt-dlp may output multiple JSON objects for playlists; take the first
    first_line = raw_json.splitlines()[0]
    try:
        info = json.loads(first_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse yt-dlp JSON output: {exc}\nRaw: {raw_json[:200]}"
        ) from exc

    duration: float = float(info.get("duration") or 0.0)
    if duration > max_duration_seconds:
        raise ValueError(
            f"Video duration {duration/3600:.1f}h exceeds the maximum allowed "
            f"{max_duration_seconds/3600:.1f}h. Please use a shorter source video."
        )

    title: str = str(info.get("title") or info.get("id") or "Untitled")
    channel: str = str(
        info.get("uploader")
        or info.get("channel")
        or info.get("uploader_id")
        or "Unknown"
    )

    metadata = UrlMetadata(
        title=title,
        channel=channel,
        duration_seconds=duration,
        url=url,
    )
    logger.info(
        "URL probed: title='%s', channel='%s', duration=%s",
        metadata.title, metadata.channel, metadata.duration_display,
    )
    return metadata


def download_video(
    url: str,
    dest_dir: Path,
    max_duration_seconds: int = _HARD_MAX_DURATION_SECONDS,
) -> Path:
    """
    Download a video from *url* into *dest_dir* using yt-dlp.

    The best quality ≤ 1080p is selected. The output filename is determined
    by yt-dlp using its default template (`%(title)s [%(id)s].%(ext)s`) but
    sanitised via --restrict-filenames to avoid filesystem-unsafe characters.

    Args:
        url:                  Source video URL.
        dest_dir:             Directory in which to write the downloaded file.
        max_duration_seconds: Hard limit; yt-dlp will abort if exceeded.

    Returns:
        Path to the downloaded video file.

    Raises:
        RuntimeError: If the download fails or the output file cannot be found.
        ValueError:   If the URL is empty.
    """
    url = url.strip()
    if not url:
        raise ValueError("URL must not be empty.")

    dest_dir.mkdir(parents=True, exist_ok=True)

    ytdlp = _resolve_ytdlp_bin()
    output_template = str(dest_dir / "%(title).50s [%(id)s].%(ext)s")

    base_args = [
        "--format", _FORMAT_SELECTOR,
        "--no-playlist",
        "--restrict-filenames",
        "--merge-output-format", "mp4",
        "--output", output_template,
        # Abort (without error) if video is longer than the hard ceiling.
        # This is a second line of defence after probe_url_metadata.
        "--match-filter", f"duration<={max_duration_seconds}",
        "--quiet",
        "--no-warnings",
        url,
    ]

    cmd = ytdlp + base_args

    logger.info("Downloading video from: %s → %s", url, dest_dir)
    _run_ytdlp(cmd, timeout=1800)  # 30 min ceiling for very large files

    # Locate the downloaded file — yt-dlp writes exactly one file per URL
    video_extensions = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
    candidates = [
        f for f in dest_dir.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]

    if not candidates:
        raise RuntimeError(
            f"yt-dlp reported success but no video file was found in '{dest_dir}'."
        )

    # If multiple files exist (shouldn't happen with --no-playlist), take the newest
    downloaded = max(candidates, key=lambda p: p.stat().st_mtime)
    logger.info("Download complete: '%s' (%.1f MB)", downloaded.name, downloaded.stat().st_size / 1_048_576)
    return downloaded
