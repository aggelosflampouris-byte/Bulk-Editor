"""
services/channel_analyzer.py — YouTube channel analysis via yt-dlp + Gemini.

Responsibilities:
  1. Scrape a YouTube channel's video metadata using yt-dlp (no API key required).
  2. Parse the raw JSON into typed VideoMeta objects.
  3. Call Gemini to generate a ChannelInsights report (topic clusters, content gaps,
     best upload windows, and Short extraction candidates).

This module has zero FFmpeg or transcription dependencies.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_MAX_VIDEOS: int = 30
_YTDLP_TIMEOUT_SECONDS: int = 60


# ── Public Types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VideoMeta:
    """
    Typed container for a single YouTube video's metadata as returned by yt-dlp.
    All numeric fields default to 0 / empty so parsing never raises KeyError.
    """
    video_id: str
    title: str
    url: str
    view_count: int
    duration_seconds: int
    upload_date: str       # YYYYMMDD string from yt-dlp
    like_count: int
    comment_count: int
    description: str

    @property
    def duration_display(self) -> str:
        """Human-readable duration e.g. '12:34'."""
        m, s = divmod(self.duration_seconds, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    @property
    def upload_date_display(self) -> str:
        """Human-readable upload date e.g. '2024-03-15'."""
        try:
            return datetime.strptime(self.upload_date, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            return self.upload_date

    @property
    def is_short_candidate(self) -> bool:
        """
        A video is a good Short extraction candidate if:
          - It is at least 5 minutes long (enough content to extract from)
          - It is not already a Short itself (< 65 seconds)
          - It has a meaningful title
        """
        return (
            self.duration_seconds >= 300
            and self.duration_seconds < 7200
            and bool(self.title.strip())
        )


@dataclass
class ChannelInsights:
    """
    Structured analysis of a YouTube channel's content performance.
    """
    channel_url: str
    total_videos_analysed: int

    # Top performing by view count
    top_videos: list[VideoMeta] = field(default_factory=list)

    # Gemini-generated narrative analysis
    topic_clusters: str = ""
    content_gaps: str = ""
    best_upload_window: str = ""
    virality_patterns: str = ""
    short_recommendations: str = ""

    # Videos recommended for Short extraction
    short_candidates: list[VideoMeta] = field(default_factory=list)

    # Raw error message if analysis partially failed
    analysis_error: str = ""


# ── Internal Helpers ───────────────────────────────────────────────────────────


def _run_ytdlp_metadata(channel_url: str, max_videos: int) -> list[dict[str, Any]]:
    """
    Run yt-dlp in flat-playlist mode to extract video metadata without downloading.
    No media is ever downloaded — this is metadata-only.
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
        "--playlist-end", str(max_videos),
        "--extractor-args", "youtubetab:approximate_date",
        channel_url,
    ]

    logger.info("Running yt-dlp metadata fetch for: %s (max %d videos)", channel_url, max_videos)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_YTDLP_TIMEOUT_SECONDS,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "yt-dlp is not installed or not on PATH. "
            "Install it with: pip install yt-dlp"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"yt-dlp timed out after {_YTDLP_TIMEOUT_SECONDS}s for URL: {channel_url}"
        ) from exc

    if result.returncode not in (0, 1):
        logger.warning("yt-dlp exited with code %d: %s", result.returncode, result.stderr[:200])

    raw_entries: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            raw_entries.append(entry)
        except json.JSONDecodeError:
            continue

    logger.info("yt-dlp returned %d metadata entries.", len(raw_entries))
    return raw_entries


def _parse_video_meta(entry: dict[str, Any]) -> VideoMeta | None:
    """
    Parse a single yt-dlp flat-playlist JSON entry into a VideoMeta object.
    Returns None if the entry lacks a valid video ID or title.
    """
    video_id = entry.get("id") or ""
    if not video_id or video_id.startswith("http"):
        video_id = entry.get("id", "")

    title = str(entry.get("title") or entry.get("fulltitle") or "").strip()
    if not video_id or not title:
        return None

    url = entry.get("url") or entry.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"
    if not url.startswith("http"):
        url = f"https://www.youtube.com/watch?v={video_id}"

    return VideoMeta(
        video_id=video_id,
        title=title,
        url=url,
        view_count=int(entry.get("view_count") or 0),
        duration_seconds=int(entry.get("duration") or 0),
        upload_date=str(entry.get("upload_date") or ""),
        like_count=int(entry.get("like_count") or 0),
        comment_count=int(entry.get("comment_count") or 0),
        description=str(entry.get("description") or "")[:500],
    )


def _build_analysis_prompt(videos: list[VideoMeta], channel_url: str) -> str:
    """Build a structured Gemini prompt from the video metadata list."""
    video_lines: list[str] = []
    for i, v in enumerate(videos, 1):
        line = (
            f"{i}. [{v.upload_date_display}] \"{v.title}\" | "
            f"Views: {v.view_count:,} | Duration: {v.duration_display} | "
            f"Likes: {v.like_count:,}"
        )
        video_lines.append(line)

    videos_block = "\n".join(video_lines)

    return f"""\
You are a YouTube content strategy expert specialising in Greek-language channels.

Below is a list of the most recent videos from a YouTube channel.
Channel URL: {channel_url}

VIDEOS (most recent first):
{videos_block}

Based on this data, provide a strategic analysis in JSON format with exactly these keys:

{{
  "topic_clusters": "<2-3 sentences identifying the main recurring topic categories>",
  "content_gaps": "<2-3 sentences describing underserved topics relative to their potential>",
  "best_upload_window": "<1-2 sentences on which days/frequency correlate with higher views>",
  "virality_patterns": "<2-3 sentences on what makes top-performing videos different>",
  "short_recommendations": "<3-5 sentences recommending video topics/styles for YouTube Shorts>",
  "top_short_candidate_ids": ["<video_id_1>", "<video_id_2>", "<video_id_3>"]
}}

Respond ONLY with the JSON object. No markdown, no code fences.
"""


# ── Public API ─────────────────────────────────────────────────────────────────


def fetch_channel_videos(
    channel_url: str,
    max_videos: int = _DEFAULT_MAX_VIDEOS,
) -> list[VideoMeta]:
    """
    Fetch and parse video metadata from a YouTube channel using yt-dlp.

    Args:
        channel_url: Full channel URL, @handle, or playlist URL.
        max_videos:  Maximum number of recent videos to fetch (default: 30).

    Returns:
        Ordered list of VideoMeta objects (most recent first).

    Raises:
        RuntimeError: If yt-dlp is unavailable or the URL fails completely.
        ValueError:   If no valid videos could be parsed from the response.
    """
    raw_entries = _run_ytdlp_metadata(channel_url, max_videos)

    videos: list[VideoMeta] = []
    for entry in raw_entries:
        meta = _parse_video_meta(entry)
        if meta is not None:
            videos.append(meta)

    if not videos:
        raise ValueError(
            f"No valid videos found for channel: {channel_url}. "
            "Check that the URL is correct and the channel is public."
        )

    return videos


def analyze_channel(
    videos: list[VideoMeta],
    channel_url: str,
    gemini_api_key: str,
) -> ChannelInsights:
    """
    Run Gemini analysis on the fetched video metadata to produce ChannelInsights.

    The short_candidates field is always populated from heuristics so the caller
    gets useful output even if the Gemini call fails.

    Args:
        videos:          List of VideoMeta objects from fetch_channel_videos().
        channel_url:     Original channel URL (for attribution in the report).
        gemini_api_key:  Google Gemini API key.

    Returns:
        A populated ChannelInsights object.
    """
    # Heuristic candidates: long-form, high view count
    short_candidates = sorted(
        [v for v in videos if v.is_short_candidate],
        key=lambda v: v.view_count,
        reverse=True,
    )[:5]

    top_videos = sorted(videos, key=lambda v: v.view_count, reverse=True)[:10]

    insights = ChannelInsights(
        channel_url=channel_url,
        total_videos_analysed=len(videos),
        top_videos=top_videos,
        short_candidates=short_candidates,
    )

    # Gemini narrative analysis
    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=gemini_api_key)
        prompt = _build_analysis_prompt(videos, channel_url)

        config = genai_types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )

        _MODELS = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.7-flash")
        raw_text = ""
        for model_name in _MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                raw_text = response.text or ""
                if raw_text.strip():
                    break
            except Exception as model_exc:
                logger.warning("Channel analyzer: model '%s' failed: %s", model_name, model_exc)
                continue

        if raw_text:
            parsed = json.loads(raw_text)
            insights.topic_clusters = parsed.get("topic_clusters", "")
            insights.content_gaps = parsed.get("content_gaps", "")
            insights.best_upload_window = parsed.get("best_upload_window", "")
            insights.virality_patterns = parsed.get("virality_patterns", "")
            insights.short_recommendations = parsed.get("short_recommendations", "")

            # AI-recommended IDs override heuristic candidates when available
            recommended_ids: list[str] = parsed.get("top_short_candidate_ids", [])
            if recommended_ids:
                id_to_video = {v.video_id: v for v in videos}
                ai_candidates = [
                    id_to_video[vid_id]
                    for vid_id in recommended_ids
                    if vid_id in id_to_video
                ]
                if ai_candidates:
                    insights.short_candidates = ai_candidates

    except Exception as exc:
        error_msg = f"Gemini analysis failed: {exc}"
        logger.error(error_msg)
        insights.analysis_error = error_msg

    return insights
