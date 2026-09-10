"""
services/youtube_transcript_fetcher.py — Fetch transcripts from YouTube video URLs via youtube-transcript-api.

Responsibilities:
  1. Parse and validate YouTube URLs / IDs.
  2. Fetch Greek (or fallback English / auto-generated) captions without downloading video media.
  3. Transform snippets into typed TranscriptionSegment objects.
  4. Format timestamped transcript blocks for LLM prompt ingestion.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import (
    InvalidVideoId,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
    YouTubeTranscriptApiException,
)

try:
    from services.transcriber import TranscriptionSegment
except ImportError:
    from shorts_engine.services.transcriber import TranscriptionSegment

logger = logging.getLogger(__name__)

# Standard YouTube ID pattern: 11 characters (alphanumeric, -, _)
_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class YouTubeTranscriptError(Exception):
    """Base exception for YouTube transcript retrieval errors."""


class InvalidYouTubeURLError(YouTubeTranscriptError):
    """Raised when the provided string is not a valid YouTube URL or ID."""


class YouTubeTranscriptUnavailableError(YouTubeTranscriptError):
    """Raised when subtitles/transcripts are disabled or unavailable for the video."""


def extract_youtube_id(url_or_id: str) -> str | None:
    """
    Extract an 11-character YouTube video ID from a URL or raw ID string.

    Supports:
      - Raw ID: dQw4w9WgXcQ
      - Standard: https://www.youtube.com/watch?v=dQw4w9WgXcQ
      - Short URL: https://youtu.be/dQw4w9WgXcQ
      - Shorts: https://www.youtube.com/shorts/dQw4w9WgXcQ
      - Embed: https://www.youtube.com/embed/dQw4w9WgXcQ
      - Extra query params: https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s

    Args:
        url_or_id: Raw string containing URL or ID.

    Returns:
        11-character video ID string, or None if not a valid YouTube reference.
    """
    cleaned = url_or_id.strip()
    if not cleaned:
        return None

    if _YOUTUBE_ID_RE.match(cleaned):
        return cleaned

    parsed = urlparse(cleaned)
    hostname = (parsed.hostname or "").lower().replace("www.", "")

    if hostname == "youtu.be":
        # Path is /<video_id>
        vid_id = parsed.path.lstrip("/").split("/")[0]
        return vid_id if _YOUTUBE_ID_RE.match(vid_id) else None

    if hostname in ("youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            v_list = qs.get("v")
            if v_list and _YOUTUBE_ID_RE.match(v_list[0]):
                return v_list[0]
        elif parsed.path.startswith(("/shorts/", "/embed/", "/v/")):
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and _YOUTUBE_ID_RE.match(parts[1]):
                return parts[1]

    return None


def fetch_youtube_transcript(
    url_or_id: str,
    languages: Sequence[str] = ("el", "en"),
) -> list[TranscriptionSegment]:
    """
    Fetch the transcript for a YouTube video URL using youtube-transcript-api.

    Prioritizes Greek ('el') and falls back to secondary languages ('en').
    Returns structured TranscriptionSegment items compatible with the shorts engine.

    Args:
        url_or_id: YouTube video URL or 11-character video ID.
        languages: Tuple/list of language codes to attempt in order of priority.

    Returns:
        List of TranscriptionSegment objects with accurate start, end, and text.

    Raises:
        InvalidYouTubeURLError: If url_or_id cannot be resolved to a video ID.
        YouTubeTranscriptUnavailableError: If captions are disabled or unavailable.
        YouTubeTranscriptError: On network or unparsable API errors.
    """
    video_id = extract_youtube_id(url_or_id)
    if not video_id:
        raise InvalidYouTubeURLError(f"Invalid YouTube URL or video ID: '{url_or_id}'")

    logger.info("Fetching YouTube transcript for video ID '%s' (languages: %s)...", video_id, languages)

    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id=video_id, languages=list(languages))
    except (TranscriptsDisabled, NoTranscriptFound) as exc:
        logger.warning("YouTube transcript unavailable for '%s': %s", video_id, exc)
        raise YouTubeTranscriptUnavailableError(
            f"Captions are disabled or not found for video '{video_id}'."
        ) from exc
    except (VideoUnavailable, InvalidVideoId) as exc:
        logger.error("YouTube video '%s' unavailable: %s", video_id, exc)
        raise YouTubeTranscriptUnavailableError(
            f"YouTube video '{video_id}' is unavailable: {exc}"
        ) from exc
    except YouTubeTranscriptApiException as exc:
        logger.error("YouTube transcript API error for '%s': %s", video_id, exc)
        raise YouTubeTranscriptError(
            f"Failed to fetch YouTube transcript: {exc}"
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error retrieving YouTube transcript: %s", exc)
        raise YouTubeTranscriptError(f"Unexpected transcript error: {exc}") from exc

    segments: list[TranscriptionSegment] = []
    for snippet in fetched:
        text = snippet.text.strip()
        if not text:
            continue
        start = float(snippet.start)
        duration = float(snippet.duration)
        end = start + duration

        # Estimate word-level chunks across the snippet duration for downstream alignment
        words_raw = text.split()
        words_timed: list[tuple[float, float, str]] | None = None
        if words_raw:
            w_step = duration / len(words_raw)
            words_timed = [
                (round(start + i * w_step, 3), round(start + (i + 1) * w_step, 3), w)
                for i, w in enumerate(words_raw)
            ]

        segments.append(
            TranscriptionSegment(
                start=round(start, 3),
                end=round(end, 3),
                text=text,
                words=words_timed,
            )
        )

    logger.info("Retrieved %d transcript segments for YouTube video '%s'.", len(segments), video_id)
    return segments


def format_transcript_for_llm(segments: Sequence[TranscriptionSegment]) -> str:
    """
    Format transcript segments into timestamped text blocks for LLM prompt ingestion.

    Output format:
      [MM:SS - MM:SS] Segment text here...
    """
    lines: list[str] = []
    for seg in segments:
        s_m, s_s = divmod(int(seg.start), 60)
        e_m, e_s = divmod(int(seg.end), 60)
        lines.append(f"[{s_m:02d}:{s_s:02d} - {e_m:02d}:{e_s:02d}] {seg.text}")
    return "\n".join(lines)
