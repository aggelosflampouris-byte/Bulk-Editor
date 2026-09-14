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
import math
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
          - It is at least 65 seconds long (so it's not already a Short)
          - It has a meaningful title
        """
        return (
            self.duration_seconds >= 65
            and self.duration_seconds < 14400
            and bool(self.title.strip())
        )


@dataclass
class NicheInsights:
    """
    Structured analysis of a YouTube niche or channel's content performance.
    """
    query: str
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

    # Viral recent videos (uploaded within the last 3 weeks, high virality score)
    viral_recent: list["ViralRecentVideo"] = field(default_factory=list)

    # Competitor viral recent videos (discovered via automated search)
    competitor_viral_recent: list["ViralRecentVideo"] = field(default_factory=list)

    # The AI-deduced search query used to find competitors
    suggested_search_query: str = ""

    # Raw error message if analysis partially failed
    analysis_error: str = ""


@dataclass(frozen=True)
class ViralRecentVideo:
    """
    A recently uploaded video (≤ 3 weeks old) with a computed virality score,
    ranked as a high-priority candidate for Short extraction.

    Virality Score formula:
        base     = log10(max(views, 1))
        recency  = max(0, 1 - days_old / 21)   # decays linearly: 1.0 → 0.0
        engmt    = likes / max(views, 1)         # engagement ratio 0.0 → 1.0
        score    = base * (1 + recency) * (1 + engmt * 10)
    A higher score = more viral AND more recent.
    """
    video: VideoMeta
    days_old: int
    virality_score: float
    virality_label: str  # 'Hot 🔥', 'Rising 📈', or 'Trending ⚡'

    @property
    def score_display(self) -> str:
        return f"{self.virality_score:.1f}"


# ── Internal Helpers ───────────────────────────────────────────────────────────


def _run_ytdlp_metadata(query: str, max_videos: int) -> list[dict[str, Any]]:
    """
    Run yt-dlp to extract video metadata without downloading.
    If query is a URL or @handle, it fetches a flat-playlist (channel).
    Otherwise, it performs a YouTube search.
    No media is ever downloaded — this is metadata-only.
    """
    target = query
    if not query.startswith("http") and not query.startswith("@"):
        import urllib.parse
        encoded = urllib.parse.quote_plus(query)
        # sp=EgIIBA%253D%253D is YouTube's "Upload Date: This Month" filter
        target = f"https://www.youtube.com/results?search_query={encoded}&sp=EgIIBA%253D%253D"

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
        "--playlist-end", str(max_videos),
        "--extractor-args", "youtubetab:approximate_date",
        target,
    ]

    logger.info("Running yt-dlp metadata fetch for: %s (max %d videos)", target, max_videos)
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
            f"yt-dlp timed out after {_YTDLP_TIMEOUT_SECONDS}s for query: {target}"
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


def _build_analysis_prompt(videos: list[VideoMeta], query: str, analytics_data: dict[str, Any] | None = None, target_niche: str | None = None) -> str:
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
    
    analytics_context = ""
    if analytics_data:
        analytics_context = (
            "Here is the authenticated 30-day channel analytics data for this exact channel:\n"
            f"- Total Views: {analytics_data.get('views', 0):,}\n"
            f"- Watch Time (Minutes): {analytics_data.get('estimatedMinutesWatched', 0):,}\n"
            f"- Average View Duration (Seconds): {analytics_data.get('averageViewDuration', 0):,}\n"
            f"- Subscribers Gained: {analytics_data.get('subscribersGained', 0):,}\n"
            f"- Total Likes: {analytics_data.get('likes', 0):,}\n"
            f"- Total Comments: {analytics_data.get('comments', 0):,}\n\n"
            "Use this deep analytics data to provide more precise recommendations on how to improve retention and engagement.\n"
        )

    target_niche_context = ""
    if target_niche:
        target_niche_context = (
            f"CRITICAL INSTRUCTION: The user has specified their target niche/topic as: '{target_niche}'. "
            "You MUST tailor all your topic clusters, content gaps, virality patterns, and short recommendations "
            "specifically to bridging their current channel audience/content towards this target niche, or generating "
            "highly viral ideas strictly within this target niche.\n\n"
        )

    return f"""\
You are a YouTube content strategy expert specialising in Greek-language content.

Below is a list of the top recent videos for the search query/channel: "{query}"

{analytics_context}
{target_niche_context}
VIDEOS (most recent or top ranked):
{videos_block}

Based on this data, provide a strategic analysis of this niche in JSON format with exactly these keys:

{{
  "suggested_search_query": "<2-4 word YouTube search query that perfectly captures the overarching niche of these videos>",
  "topic_clusters": "<2-3 sentences identifying the main recurring topic categories in this niche>",
  "content_gaps": "<2-3 sentences describing underserved topics relative to their potential in this niche>",
  "best_upload_window": "<1-2 sentences on what frequency or timing correlates with higher views>",
  "virality_patterns": "<2-3 sentences on what makes top-performing videos in this niche stand out>",
  "short_recommendations": "<3-5 sentences recommending specific video topics/styles for YouTube Shorts to dominate this niche>",
  "top_short_candidate_ids": ["<video_id_1>", "<video_id_2>", "<video_id_3>"]
}}

Respond ONLY with the JSON object. No markdown, no code fences.
"""


# ── Public API ─────────────────────────────────────────────────────────────────

_RECENCY_WINDOW_DAYS: int = 21  # 3 weeks


def find_viral_recent_videos(
    videos: list[VideoMeta],
    channel_avg_views: float | None = None,
    max_results: int = 10,
) -> list[ViralRecentVideo]:
    """
    Filter the channel's videos to those uploaded within the last 3 weeks and
    rank them by a composite virality score.

    Virality Score formula (all components are non-negative):
        base    = log10(max(views, 1))
        recency = max(0, 1 - days_old / 21)   # linear decay: 1.0 (today) → 0.0 (21 days)
        engmt   = likes / max(views, 1)        # engagement ratio
        score   = base * (1.0 + recency) * (1.0 + engmt * 10.0)

    The recency multiplier ensures a video uploaded yesterday with 10k views scores
    higher than one uploaded 20 days ago with 15k views, reflecting the YouTube
    algorithm's preference for fresh content with fast early engagement.

    Args:
        videos:            Full list of VideoMeta from fetch_channel_videos().
        channel_avg_views: Optional channel average views (unused in current formula but
                           available for future relative scoring).
        max_results:       Maximum number of candidates to return (default: 10).

    Returns:
        List of ViralRecentVideo ordered by virality_score descending.
        Empty list if no videos were uploaded in the last 3 weeks.
    """
    now = datetime.now(tz=timezone.utc)
    candidates: list[ViralRecentVideo] = []

    for v in videos:
        if not v.upload_date or len(v.upload_date) < 8:
            continue

        try:
            upload_dt = datetime.strptime(v.upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        days_old = (now - upload_dt).days
        if days_old > _RECENCY_WINDOW_DAYS or days_old < 0:
            continue

        # Must be long enough to extract Shorts from
        if not v.is_short_candidate:
            continue

        # Compute virality score
        base = math.log10(max(v.view_count, 1))
        recency = max(0.0, 1.0 - days_old / _RECENCY_WINDOW_DAYS)
        engmt = v.like_count / max(v.view_count, 1)
        score = base * (1.0 + recency) * (1.0 + engmt * 10.0)

        # Classify label based on score percentile (computed post-sort)
        candidates.append(
            ViralRecentVideo(
                video=v,
                days_old=days_old,
                virality_score=round(score, 2),
                virality_label="",  # filled after sorting
            )
        )

    # Sort by score descending
    candidates.sort(key=lambda c: c.virality_score, reverse=True)

    # Fallback: if no videos found in the last 3 weeks, score the most recent videos 
    # regardless of age, purely based on views and engagement (recency = 0)
    if not candidates:
        for v in videos:
            if not v.is_short_candidate:
                continue
            
            base = math.log10(max(v.view_count, 1))
            engmt = v.like_count / max(v.view_count, 1)
            score = base * 1.0 * (1.0 + engmt * 10.0)
            
            candidates.append(
                ViralRecentVideo(
                    video=v,
                    days_old=(now - datetime.strptime(v.upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)).days if v.upload_date else 999,
                    virality_score=round(score, 2),
                    virality_label="",
                )
            )
        candidates.sort(key=lambda c: c.virality_score, reverse=True)

    # Assign labels based on rank
    labelled: list[ViralRecentVideo] = []
    for rank, c in enumerate(candidates[:max_results]):
        if rank == 0:
            label = "Hot 🔥"
        elif rank <= 2:
            label = "Rising 📈"
        else:
            label = "Trending ⚡"
        labelled.append(
            ViralRecentVideo(
                video=c.video,
                days_old=c.days_old,
                virality_score=c.virality_score,
                virality_label=label,
            )
        )

    logger.info(
        "find_viral_recent_videos: %d / %d videos qualify as recent viral candidates.",
        len(labelled), len(videos),
    )
    return labelled


def fetch_youtube_videos(
    query: str,
    max_videos: int = _DEFAULT_MAX_VIDEOS,
) -> list[VideoMeta]:
    """
    Fetch and parse video metadata from YouTube using yt-dlp.
    Supports channel URLs, @handles, or plain search queries.

    Args:
        query:       Search term (e.g. "Greek politics"), channel URL, or @handle.
        max_videos:  Maximum number of recent videos to fetch (default: 30).

    Returns:
        Ordered list of VideoMeta objects.

    Raises:
        RuntimeError: If yt-dlp is unavailable or the query fails completely.
        ValueError:   If no valid videos could be parsed from the response.
    """
    raw_entries = _run_ytdlp_metadata(query, max_videos)

    videos: list[VideoMeta] = []
    for entry in raw_entries:
        meta = _parse_video_meta(entry)
        if meta is not None:
            videos.append(meta)

    if not videos:
        raise ValueError(
            f"No valid videos found for query: {query}. "
            "Check that the query is correct and not blocked."
        )

    return videos


def analyze_niche(
    videos: list[VideoMeta],
    query: str,
    gemini_api_key: str,
    analytics_data: dict[str, Any] | None = None,
    target_niche: str | None = None,
) -> NicheInsights:
    """
    Run Gemini analysis on the fetched video metadata to produce NicheInsights.

    The short_candidates field is always populated from heuristics so the caller
    gets useful output even if the Gemini call fails.

    Args:
        videos:          List of VideoMeta objects from fetch_youtube_videos().
        query:           Original search query or channel URL.
        gemini_api_key:  Google Gemini API key.
        analytics_data:  Optional 30-day analytics data from YouTube Analytics API.

    Returns:
        A populated NicheInsights object.
    """
    # Heuristic candidates: long-form, high view count
    short_candidates = sorted(
        [v for v in videos if v.is_short_candidate],
        key=lambda v: v.view_count,
        reverse=True,
    )[:5]

    top_videos = sorted(videos, key=lambda v: v.view_count, reverse=True)[:10]

    # Always compute viral recent picks — independent of Gemini outcome
    channel_avg_views = (
        sum(v.view_count for v in videos) / len(videos) if videos else 0.0
    )
    viral_recent = find_viral_recent_videos(
        videos,
        channel_avg_views=channel_avg_views,
    )

    insights = NicheInsights(
        query=query,
        total_videos_analysed=len(videos),
        top_videos=top_videos,
        short_candidates=short_candidates,
        viral_recent=viral_recent,
    )

    # Gemini narrative analysis
    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=gemini_api_key)
        prompt = _build_analysis_prompt(videos, query, analytics_data, target_niche)

        config = genai_types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )

        _MODELS = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.7-flash")
        parsed = None
        for model_name in _MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                raw_text = response.text or ""
                if not raw_text.strip():
                    continue
                    
                import re
                
                # Strip markdown code fences if present
                fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
                if fenced:
                    raw_text = fenced.group(1)
                    
                # Extract the first JSON object block to ignore conversational padding
                obj_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if obj_match:
                    raw_text = obj_match.group(0)
                    
                # strict=False allows unescaped control chars (like \n) inside strings
                parsed = json.loads(raw_text, strict=False)
                break
            except Exception as model_exc:
                logger.warning("Channel analyzer: model '%s' failed or returned invalid JSON: %s", model_name, model_exc)
                continue

        if parsed:
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

            insights.suggested_search_query = parsed.get("suggested_search_query", "").strip()

            # Competitor Discovery
            is_channel_scan = query.startswith("http") or query.startswith("@")
            if is_channel_scan and insights.suggested_search_query:
                try:
                    logger.info("Executing competitor discovery for niche: %s", insights.suggested_search_query)
                    competitor_videos = fetch_youtube_videos(insights.suggested_search_query, max_videos=30)
                    if competitor_videos:
                        insights.competitor_viral_recent = find_viral_recent_videos(
                            competitor_videos,
                            channel_avg_views=channel_avg_views,
                        )
                except Exception as comp_exc:
                    logger.warning("Competitor discovery failed: %s", comp_exc)
        else:
            insights.analysis_error = "AI models failed to generate valid JSON insights after multiple attempts."

    except Exception as exc:
        error_msg = f"Gemini analysis failed: {exc}"
        logger.error(error_msg)
        insights.analysis_error = error_msg

    return insights
