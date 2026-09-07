"""
services/broll_fetcher.py — Pexels Video API search and clip download.

Responsibilities:
  1. Accept a text query derived from the video's full transcript.
  2. Search the Pexels Videos API for a relevant HD landscape clip.
  3. Download the best-matching clip to a caller-supplied destination path.
  4. Degrade gracefully (return None) if the API key is absent or quota is hit.

This module has zero FFmpeg or transcription dependencies.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"
_REQUEST_TIMEOUT_SECONDS = 30
_DOWNLOAD_CHUNK_SIZE = 65_536  # 64 KB streaming chunks
_MIN_CLIP_DURATION_SECONDS = 5
# Prefer HD landscape clips — overlay_broll already scales them to the frame.
# Portrait stock on Pexels is dominated by lifestyle/feet/beach content.
_PREFERRED_WIDTH = 1920


# ── Public Types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BRollClip:
    """Immutable descriptor for a single Pexels video clip."""

    video_id: int
    url: str
    duration: int        # seconds
    width: int
    height: int
    file_size: Optional[int]  # bytes, may be absent in API response


# ── Internal Helpers ───────────────────────────────────────────────────────────

def _build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": api_key,
        "User-Agent": "GreekShortsEngine/1.0",
    }


def _select_best_video_file(video_files: list[dict]) -> Optional[dict]:
    """
    From the Pexels video_files list, select the best HD landscape variant.

    Selection priority:
      1. Must have a non-empty download link.
      2. Landscape orientation preferred (width > height) — better topical
         match availability and scales cleanly into the B-roll overlay area.
      3. Width closest to 1920 px (HD landscape).
      4. If no landscape files exist, accept any valid file as fallback.

    Returns the chosen video_file dict, or None if no suitable file exists.
    """
    valid_files = [vf for vf in video_files if vf.get("link")]
    if not valid_files:
        return None

    # Prefer landscape files (best topical variety on Pexels)
    landscape_files = [
        vf for vf in valid_files
        if vf.get("width", 0) > vf.get("height", 0)
    ]
    candidates = landscape_files if landscape_files else valid_files

    # Sort by absolute deviation from preferred HD width, ascending
    candidates.sort(
        key=lambda vf: abs(vf.get("width", 0) - _PREFERRED_WIDTH)
    )
    return candidates[0]


# ── Public API ─────────────────────────────────────────────────────────────────

def search_broll(
    query: str,
    api_key: str,
    per_page: int = 25,
) -> Optional[BRollClip]:
    """
    Search the Pexels Videos API for a relevant B-roll clip.

    No orientation filter is applied — Pexels portrait stock is dominated by
    lifestyle/feet/beach content that is almost never topically relevant.
    Searching all orientations and selecting the best HD landscape clip gives
    far better topical matches across any subject matter.

    Args:
        query:    English search query (ideally from generate_broll_query).
        api_key:  Pexels API key (v1).
        per_page: Number of results to request (max 80). Higher values
                  increase the chance of finding a quality topical clip.

    Returns:
        A BRollClip describing the best candidate, or None.
    """
    if not api_key or not api_key.strip():
        logger.warning("Pexels API key is empty — B-roll step will be skipped.")
        return None

    # No orientation filter: searching all orientations yields the best topical
    # match. The overlay_broll function scales the clip to fit regardless.
    params = urlencode({
        "query": query,
        "per_page": min(per_page, 80),
        "size": "large",  # HD quality
    })
    url = f"{_PEXELS_VIDEO_SEARCH_URL}?{params}"

    logger.info("Searching Pexels for B-roll: query='%s'", query)
    try:
        response = requests.get(
            url,
            headers=_build_headers(api_key),
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            logger.warning("Pexels API rate limit hit — B-roll step skipped.")
        else:
            logger.warning("Pexels API HTTP error: %s", exc)
        return None
    except requests.exceptions.RequestException as exc:
        logger.warning("Pexels API request failed: %s", exc)
        return None

    data = response.json()
    videos: list[dict] = data.get("videos", [])

    if not videos:
        logger.info("Pexels returned no results for query: '%s'", query)
        return None

    # Filter by minimum duration and select the best video file
    for video in videos:
        duration: int = video.get("duration", 0)
        if duration < _MIN_CLIP_DURATION_SECONDS:
            continue

        video_files: list[dict] = video.get("video_files", [])
        best_file = _select_best_video_file(video_files)
        if best_file is None:
            continue

        clip = BRollClip(
            video_id=video.get("id", 0),
            url=best_file["link"],
            duration=duration,
            width=best_file.get("width", 0),
            height=best_file.get("height", 0),
            file_size=best_file.get("file_type"),  # May be None
        )
        logger.info(
            "Selected B-roll clip: id=%d, %dx%d, %ds",
            clip.video_id, clip.width, clip.height, clip.duration,
        )
        return clip

    logger.info("No B-roll clip passed quality filters for query: '%s'", query)
    return None


def download_clip(clip: BRollClip, dest: Path) -> Path:
    """
    Stream-download a Pexels video clip to *dest*.

    Uses chunked streaming to avoid loading large files into memory.

    Args:
        clip: BRollClip descriptor with the download URL.
        dest: Destination file path (parent directory must exist).

    Returns:
        The written *dest* path.

    Raises:
        RuntimeError: If the download fails or the file is empty after writing.
    """
    logger.info("Downloading B-roll clip %d → '%s'", clip.video_id, dest)

    try:
        with requests.get(
            clip.url,
            stream=True,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        ) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Failed to download B-roll clip {clip.video_id}: {exc}"
        ) from exc

    # Validate that we actually wrote something
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(
            f"B-roll download produced an empty file at '{dest}'."
        )

    logger.info(
        "B-roll download complete: %.1f MB",
        dest.stat().st_size / (1024 * 1024),
    )
    return dest


def extract_broll_query(transcript_text: str, max_words: int = 6) -> str:
    """
    Derive a short, search-friendly query from a full transcript string.

    Strategy: strip common Greek stop-words, take the first *max_words*
    meaningful words, and return them joined. This is intentionally simple
    to avoid adding an NLP dependency; the Pexels API is forgiving with
    Greek + English mixed queries.

    Args:
        transcript_text: Raw concatenated transcript text (Greek).
        max_words:       Maximum number of words to include in the query.

    Returns:
        A search query string (may be empty if transcript is empty).
    """
    # Pexels performs better with short, English-style keywords.
    # We send the first few words of the Greek transcript — Pexels
    # understands multilingual content to a reasonable degree.
    words = transcript_text.split()

    # Minimal Greek stopword filter (common filler words)
    _GREEK_STOP_WORDS: frozenset[str] = frozenset({
        "και", "το", "τα", "τη", "τον", "την", "της", "τους", "τις",
        "ο", "η", "οι", "τα", "σε", "με", "από", "για", "που", "δεν",
        "είναι", "να", "αλλά", "ότι", "αυτό", "αυτά", "μου", "μας",
        "σας", "σου", "του", "τους", "κι", "ως", "πως", "κάθε",
    })

    meaningful = [w for w in words if w.lower() not in _GREEK_STOP_WORDS]
    query_words = meaningful[:max_words] if meaningful else words[:max_words]
    return " ".join(query_words)
