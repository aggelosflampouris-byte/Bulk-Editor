"""
services/niche_sourcing.py — Sourcing fresh viral competitor & niche videos from YouTube.

Responsibilities:
  1. Search YouTube for high-velocity videos in our content niche (Greek news, politics, economics, debates).
  2. Strictly filter to videos uploaded within the last 3 weeks (≤ 21 days old).
  3. Strictly exclude @DianismaNews to prevent re-using our own channel's older content.
  4. Deduplicate and score candidates by view velocity and virality potential.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from services.cache_manager import (
        get_processed_video_ids,
        is_video_already_processed,
    )
    from services.channel_analyzer import (
        DIANISMA_CHANNEL_ID,
        VideoMeta,
        fetch_youtube_videos,
        find_viral_recent_videos,
    )
except ImportError:
    from shorts_engine.services.cache_manager import (
        get_processed_video_ids,
        is_video_already_processed,
    )
    from shorts_engine.services.channel_analyzer import (
        DIANISMA_CHANNEL_ID,
        VideoMeta,
        fetch_youtube_videos,
        find_viral_recent_videos,
    )

logger = logging.getLogger(__name__)

# Maximum video age: 3 weeks (21 days)
MAX_RECENCY_DAYS: int = 21

# Channel identifiers to exclude from sourcing (ensures we do not re-use our own videos)
EXCLUDED_CHANNEL_IDENTIFIERS: set[str] = {
    DIANISMA_CHANNEL_ID.lower(),
    "uczmnsmxzae4m_hzh6g1jkg",
    "@dianismanews",
    "dianismanews",
    "dianisma",
}

# Curated high-signal Greek news and politics niche search queries
DEFAULT_NICHE_QUERIES: tuple[str, ...] = (
    "ελληνική πολιτική ειδήσεις συνεντεύξεις",
    "εξελίξεις ελλάδα οικονομία επικαιρότητα",
    "δηλώσεις πολιτική βουλή νέα αντιπαράθεση",
    "ελληνικά νέα επικαιρότητα οικονομικά μέτρα ακρίβεια",
    "γεωπολιτική ελλάδα τουρκία διεθνείς σχέσεις",
    "συνεντεύξεις αρχηγών κομμάτων πολιτικές αναλύσεις",
    "ελληνική οικονομία τράπεζες συντάξεις μισθοί",
    "κοινωνικά θέματα ελλάδα συζητήσεις τηλεόραση εκπομπές",
    "ελληνικό κοινοβούλιο ένταση τοποθετήσεις debate",
    "πρωτοσέλιδα ειδήσεων αποκαλύψεις πολιτικό ρεπορτάζ",
    "ελληνική επικαιρότητα έρευνες οικονομία φορολογία",
    "πολιτική επικαιρότητα εκλογές δημοσκοπήσεις",
)


def _is_channel_excluded(channel_id: str, channel_title: str) -> bool:
    """Check if a video belongs to our own channel or blacklisted identifiers."""
    cid = (channel_id or "").strip().lower()
    ctitle = (channel_title or "").strip().lower()
    for ident in EXCLUDED_CHANNEL_IDENTIFIERS:
        if ident and (ident in cid or ident in ctitle):
            return True
    return False


def _parse_iso_duration(dur: str) -> int:
    """Parse ISO 8601 duration (e.g., PT14M32S) to integer seconds."""
    match = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", dur)
    if not match:
        return 0
    h, m, s = match.groups()
    return int(h or 0) * 3600 + int(m or 0) * 60 + int(s or 0)


def fetch_niche_videos_via_api(
    youtube_client: Any,
    query: str,
    max_results: int = 20,
    max_age_days: int = MAX_RECENCY_DAYS,
    output_dir: Path | None = None,
) -> list[VideoMeta]:
    """
    Search YouTube Data API v3 for recent niche videos with strict 3-week cutoff.
    """
    now = datetime.now(timezone.utc)
    cutoff_dt = now - timedelta(days=max_age_days)
    published_after = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(
        "Searching YouTube Data API for niche videos (Query: '%s', Published After: %s)...",
        query,
        published_after,
    )

    try:
        search_res = youtube_client.search().list(
            q=query,
            part="snippet",
            type="video",
            order="viewCount",
            publishedAfter=published_after,
            maxResults=min(50, max_results * 2),
        ).execute()
    except (RuntimeError, OSError, ValueError, KeyError, AttributeError) as exc:
        logger.warning("YouTube Data API niche search failed: %s", exc)
        return []

    items = search_res.get("items", [])
    candidate_video_ids: list[str] = []
    for item in items:
        id_info = item.get("id", {})
        vid_id = id_info.get("videoId")
        if not vid_id:
            continue
        snip = item.get("snippet", {})
        ch_id = snip.get("channelId", "")
        ch_title = snip.get("channelTitle", "")
        if _is_channel_excluded(ch_id, ch_title):
            logger.debug("Skipping our own channel video from search: %s", vid_id)
            continue
        candidate_video_ids.append(vid_id)

    if not candidate_video_ids:
        return []

    try:
        details_res = youtube_client.videos().list(
            id=",".join(candidate_video_ids[:50]),
            part="snippet,statistics,contentDetails",
        ).execute()
    except (RuntimeError, OSError, ValueError, KeyError, AttributeError) as exc:
        logger.warning("YouTube Data API video details fetch failed: %s", exc)
        return []

    results: list[VideoMeta] = []
    for item in details_res.get("items", []):
        vid_id = item.get("id", "")
        snip = item.get("snippet", {})
        stats = item.get("statistics", {})
        content = item.get("contentDetails", {})

        ch_id = snip.get("channelId", "")
        ch_title = snip.get("channelTitle", "")
        if _is_channel_excluded(ch_id, ch_title):
            continue

        pub_str = snip.get("publishedAt", "")
        try:
            pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        days_old = (now - pub_dt).days
        if days_old > max_age_days or days_old < 0:
            continue

        raw_dur = content.get("duration", "")
        duration_sec = _parse_iso_duration(raw_dur)
        # Shorts candidates: 20 seconds to 60 minutes
        if duration_sec < 20 or duration_sec > 3600:
            continue

        video_url = f"https://www.youtube.com/watch?v={vid_id}"
        if is_video_already_processed(vid_id, output_dir) or is_video_already_processed(video_url, output_dir):
            logger.debug("Skipping already processed video from API results: %s", vid_id)
            continue

        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0))
        upload_date_compact = pub_dt.strftime("%Y%m%d")

        results.append(
            VideoMeta(
                video_id=vid_id,
                title=snip.get("title", ""),
                url=video_url,
                view_count=views,
                duration_seconds=duration_sec,
                upload_date=upload_date_compact,
                like_count=likes,
                comment_count=comments,
                description=snip.get("description", ""),
            )
        )

    return results


def fetch_niche_videos_via_ytdlp(
    query: str,
    max_results: int = 20,
    max_age_days: int = MAX_RECENCY_DAYS,
    output_dir: Path | None = None,
) -> list[VideoMeta]:
    """
    Search YouTube via yt-dlp and filter strictly to videos uploaded within max_age_days.
    """
    now = datetime.now(timezone.utc)
    try:
        raw_videos = fetch_youtube_videos(query, max_videos=max_results * 2)
    except (RuntimeError, ValueError) as exc:
        logger.warning("yt-dlp niche query failed for '%s': %s", query, exc)
        return []

    filtered: list[VideoMeta] = []
    for v in raw_videos:
        if (
            _is_channel_excluded("", v.title)
            or "dianisma" in v.title.lower()
            or "dianisma" in (v.description or "").lower()
        ):
            continue
        if is_video_already_processed(v.video_id, output_dir) or is_video_already_processed(v.url, output_dir):
            logger.debug("Skipping already processed video from yt-dlp results: %s", v.video_id)
            continue

        if not v.upload_date:
            continue
        try:
            upload_dt = datetime.strptime(v.upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        days_old = (now - upload_dt).days
        if days_old > max_age_days or days_old < 0:
            continue

        if not v.is_short_candidate:
            continue

        filtered.append(v)

    return filtered


def find_niche_trend_videos(
    query: str | None = None,
    max_videos: int = 5,
    max_age_days: int = MAX_RECENCY_DAYS,
    youtube_client: Any | None = None,
    output_dir: Path | None = None,
    exclude_video_ids: set[str] | None = None,
    shuffle_queries: bool = True,
) -> list[VideoMeta]:
    """
    Find top trending, viral videos in our niche uploaded within the last 3 weeks.
    Strictly excludes @DianismaNews to avoid re-using our own content.
    Excludes previously processed videos to prevent re-sourcing the same content.

    Returns:
        List of fresh VideoMeta sorted by viral breakout potential.
    """
    import random

    is_custom_query = bool(query and query.strip())
    if is_custom_query:
        search_queries = [query.strip()]  # type: ignore[union-attr]
    else:
        all_q = list(DEFAULT_NICHE_QUERIES)
        if shuffle_queries:
            random.shuffle(all_q)
        search_queries = all_q

    processed_ids = get_processed_video_ids(output_dir)
    if exclude_video_ids:
        processed_ids |= set(exclude_video_ids)

    all_candidates: dict[str, VideoMeta] = {}

    for q in search_queries:
        videos: list[VideoMeta] = []
        if youtube_client is not None:
            videos = fetch_niche_videos_via_api(
                youtube_client=youtube_client,
                query=q,
                max_results=max_videos * 3,
                max_age_days=max_age_days,
                output_dir=output_dir,
            )

        # Fallback to yt-dlp if API returned 0 or wasn't provided
        if not videos:
            videos = fetch_niche_videos_via_ytdlp(
                query=q,
                max_results=max_videos * 3,
                max_age_days=max_age_days,
                output_dir=output_dir,
            )

        for v in videos:
            if (
                v.video_id not in all_candidates
                and v.video_id not in processed_ids
                and not is_video_already_processed(v.video_id, output_dir)
                and not is_video_already_processed(v.url, output_dir)
            ):
                all_candidates[v.video_id] = v

        # If we have collected plenty of fresh candidates, stop querying
        if len(all_candidates) >= max(10, max_videos * 3):
            break

    candidate_list = list(all_candidates.values())
    if not candidate_list:
        logger.warning("No fresh unprocessed niche videos found within the %d-day window.", max_age_days)
        return []

    # Rank by virality score and view velocity
    viral_recent = find_viral_recent_videos(candidate_list, max_results=len(candidate_list))
    if viral_recent:
        ranked = [vr.video for vr in viral_recent]
    else:
        now = datetime.now(timezone.utc)

        def _velocity_key(v: VideoMeta) -> float:
            try:
                upload_dt = datetime.strptime(v.upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
                days = max(1, (now - upload_dt).days)
            except ValueError:
                days = 1
            return v.view_count / days

        candidate_list.sort(key=_velocity_key, reverse=True)
        ranked = candidate_list

    # When query is generic/auto and multiple candidates exist, diversify top-tier candidates
    # to avoid picking the exact same video on consecutive runs.
    if not is_custom_query and shuffle_queries and len(ranked) > max_videos:
        tier_size = min(len(ranked), max_videos * 2)
        top_tier = list(ranked[:tier_size])
        random.shuffle(top_tier)
        return top_tier[:max_videos]

    return ranked[:max_videos]
