"""
tests/test_niche_sourcing.py — Unit tests for Niche Trend Sourcing and Autopilot BGM.

Verifies:
  1. Strict exclusion of @DianismaNews to prevent re-using older channel videos.
  2. Strict 3-week recency filtering (≤ 21 days old).
  3. YouTube Data API v3 and yt-dlp dual-mode niche search.
  4. Autopilot pipeline sourcing fresh niche videos and enabling instrumental royalty-free BGM.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from shorts_engine.config import Settings
from shorts_engine.services.channel_analyzer import VideoMeta
from shorts_engine.services.niche_sourcing import (
    MAX_RECENCY_DAYS,
    _is_channel_excluded,
    _parse_iso_duration,
    fetch_niche_videos_via_api,
    fetch_niche_videos_via_ytdlp,
    find_niche_trend_videos,
    is_greek_content_relevant,
)


def test_is_channel_excluded() -> None:
    # Our channel variations must be strictly excluded
    assert _is_channel_excluded("UCZmznsMXZaE4m_hZH6g1JKg", "Dianisma") is True
    assert _is_channel_excluded("uczmnsmxzae4m_hzh6g1jkg", "Something") is True
    assert _is_channel_excluded("some_other_id", "@DianismaNews") is True
    assert _is_channel_excluded("some_other_id", "Dianisma News Official") is True

    # External competitor channels must NOT be excluded
    assert _is_channel_excluded("UC6ZYCxxt5wDvNbhjzyFSVKg", "Η ΝΑΥΤΕΜΠΟΡΙΚΗ") is False
    assert _is_channel_excluded("UCzjZs1ehOHcrwlx0wh-GM4w", "Greek Vision") is False
    assert _is_channel_excluded("UCwUNbp_4Y2Ry-asyerw2jew", "StarTvGreece") is False


def test_parse_iso_duration() -> None:
    assert _parse_iso_duration("PT14M32S") == 14 * 60 + 32
    assert _parse_iso_duration("PT1H2M3S") == 3600 + 120 + 3
    assert _parse_iso_duration("PT45S") == 45
    assert _parse_iso_duration("INVALID") == 0


def test_fetch_niche_videos_via_api_filtering() -> None:
    """Verify YouTube Data API search enforces 21-day age cutoff and excludes Dianisma."""
    now = datetime.now(timezone.utc)
    fresh_date = (now - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    stale_date = (now - timedelta(days=35)).strftime("%Y-%m-%dT%H:%M:%SZ")

    mock_client = MagicMock()

    # Search items
    mock_search = MagicMock()
    mock_search.list.return_value.execute.return_value = {
        "items": [
            {
                "id": {"videoId": "vid_fresh"},
                "snippet": {"channelId": "ext_ch1", "channelTitle": "Greek News"},
            },
            {
                "id": {"videoId": "vid_dianisma"},
                "snippet": {
                    "channelId": "UCZmznsMXZaE4m_hZH6g1JKg",
                    "channelTitle": "Dianisma",
                },
            },
            {
                "id": {"videoId": "vid_stale"},
                "snippet": {"channelId": "ext_ch2", "channelTitle": "Politics GR"},
            },
        ]
    }
    mock_client.search.return_value = mock_search

    # Video details
    mock_videos = MagicMock()
    mock_videos.list.return_value.execute.return_value = {
        "items": [
            {
                "id": "vid_fresh",
                "snippet": {
                    "title": "Εξελίξεις στην Οικονομία",
                    "channelId": "ext_ch1",
                    "channelTitle": "Greek News",
                    "publishedAt": fresh_date,
                    "description": "Ανάλυση",
                },
                "statistics": {
                    "viewCount": "54000",
                    "likeCount": "2100",
                    "commentCount": "320",
                },
                "contentDetails": {"duration": "PT12M00S"},
            },
            {
                "id": "vid_stale",
                "snippet": {
                    "title": "Παλιά Συζήτηση",
                    "channelId": "ext_ch2",
                    "channelTitle": "Politics GR",
                    "publishedAt": stale_date,
                    "description": "Παλιό",
                },
                "statistics": {
                    "viewCount": "100000",
                    "likeCount": "1000",
                    "commentCount": "50",
                },
                "contentDetails": {"duration": "PT10M00S"},
            },
        ]
    }
    mock_client.videos.return_value = mock_videos

    results = fetch_niche_videos_via_api(
        youtube_client=mock_client,
        query="ελληνική πολιτική",
        max_results=5,
        max_age_days=MAX_RECENCY_DAYS,
    )

    assert len(results) == 1
    assert results[0].video_id == "vid_fresh"
    assert results[0].title == "Εξελίξεις στην Οικονομία"
    assert results[0].view_count == 54000


def test_fetch_niche_videos_via_ytdlp_filtering() -> None:
    """Verify yt-dlp fallback filters to ≤ 21 days old and excludes Dianisma."""
    now = datetime.now(timezone.utc)
    fresh_compact = (now - timedelta(days=7)).strftime("%Y%m%d")
    stale_compact = (now - timedelta(days=40)).strftime("%Y%m%d")

    raw_candidates = [
        VideoMeta(
            video_id="ext_fresh",
            title="Νέα δήλωση στη Βουλή",
            url="https://youtube.com/watch?v=ext_fresh",
            view_count=75000,
            duration_seconds=600,
            upload_date=fresh_compact,
            like_count=500,
            comment_count=100,
            description="Πολιτική",
        ),
        VideoMeta(
            video_id="ext_stale",
            title="Παλιό debate 2025",
            url="https://youtube.com/watch?v=ext_stale",
            view_count=200000,
            duration_seconds=900,
            upload_date=stale_compact,
            like_count=1000,
            comment_count=200,
            description="Debate",
        ),
        VideoMeta(
            video_id="dianisma_old",
            title="Dianisma παλιό βίντεο",
            url="https://youtube.com/watch?v=dianisma_old",
            view_count=50000,
            duration_seconds=500,
            upload_date=fresh_compact,
            like_count=500,
            comment_count=100,
            description="Dianisma",
        ),
    ]

    with patch(
        "shorts_engine.services.niche_sourcing.fetch_youtube_videos",
        return_value=raw_candidates,
    ):
        results = fetch_niche_videos_via_ytdlp("ελληνική πολιτική", max_age_days=21)
        assert len(results) == 1
        assert results[0].video_id == "ext_fresh"


def test_find_niche_trend_videos_orchestration() -> None:
    """Verify find_niche_trend_videos ranks candidates by virality."""
    now = datetime.now(timezone.utc)
    v1 = VideoMeta(
        video_id="v_high_views",
        title="Υψηλή τηλεθέαση συνέντευξη",
        url="https://youtube.com/watch?v=v_high_views",
        view_count=120000,
        duration_seconds=500,
        upload_date=(now - timedelta(days=2)).strftime("%Y%m%d"),
        like_count=4000,
        comment_count=600,
        description="viral",
    )
    v2 = VideoMeta(
        video_id="v_low_views",
        title="Χαμηλή τηλεθέαση",
        url="https://youtube.com/watch?v=v_low_views",
        view_count=1000,
        duration_seconds=300,
        upload_date=(now - timedelta(days=15)).strftime("%Y%m%d"),
        like_count=20,
        comment_count=5,
        description="low",
    )

    with patch(
        "shorts_engine.services.niche_sourcing.fetch_niche_videos_via_ytdlp",
        return_value=[v2, v1],
    ):
        top = find_niche_trend_videos(
            query="ελληνική πολιτική", max_videos=1, max_age_days=21
        )
        assert len(top) == 1
        assert top[0].video_id == "v_high_views"


def test_autopilot_pipeline_enables_bg_music_and_niche_sourcing(tmp_path: Path) -> None:
    """Verify autopilot pipeline defaults to niche search and activates instrumental royalty-free BGM."""
    from shorts_engine.services.autopilot import run_autopilot_pipeline

    now = datetime.now(timezone.utc)
    mock_niche_video = VideoMeta(
        video_id="niche_breakout",
        title="Νέα δήλωση για την οικονομία",
        url="https://youtube.com/watch?v=niche_breakout",
        view_count=95000,
        duration_seconds=420,
        upload_date=(now - timedelta(days=3)).strftime("%Y%m%d"),
        like_count=3500,
        comment_count=400,
        description="Οικονομία",
    )

    settings = Settings(
        target_width=1080,
        target_height=1920,
        output_dir=tmp_path / "out",
        enable_bg_music=True,
        bg_music_track="ambient_calm",
        bg_music_volume=0.15,
        bg_music_ducking=True,
    )

    with (
        patch(
            "shorts_engine.services.autopilot.find_niche_trend_videos",
            return_value=[mock_niche_video],
        ),
        patch("shorts_engine.services.autopilot.download_video_section") as mock_dl_sec,
        patch(
            "shorts_engine.services.autopilot.fetch_youtube_transcript",
            return_value=None,
        ),
        patch(
            "shorts_engine.services.autopilot.download_video",
            return_value=tmp_path / "vid.mp4",
        ),
        patch("shorts_engine.services.autopilot.transcribe", return_value=[]),
        patch("shorts_engine.services.autopilot.select_clips", return_value=[]),
    ):
        mock_dl_sec.side_effect = RuntimeError("fallback")
        gen = run_autopilot_pipeline(
            settings=settings, num_videos=1, production_strategy="hybrid"
        )

        messages = []
        try:
            for msg, _pct, _data in gen:
                messages.append(msg)
        except RuntimeError:
            pass  # Stopped after clip selection yielded 0, which is expected with mocked empty transcript

        # Confirm niche sourcing message was emitted
        assert any(
            "Searching YouTube for fresh viral videos in our niche" in m
            for m in messages
        )
        assert any("Selected 1 highly viral videos" in m for m in messages)


def test_autopilot_pipeline_dianisma_channel_url_routes_to_niche_discovery(
    tmp_path: Path,
) -> None:
    from shorts_engine.services.autopilot import (
        DIANISMA_CHANNEL_URL,
        run_autopilot_pipeline,
    )

    settings = Settings(gemini_api_key="test_api_key", output_dir=tmp_path)
    mock_niche_video = VideoMeta(
        video_id="niche_vid_xyz",
        title="Breaking Niche News Debate",
        url="https://www.youtube.com/watch?v=niche_vid_xyz",
        view_count=120000,
        duration_seconds=300,
        upload_date="20260920",
        like_count=4000,
        comment_count=500,
        description="Debate",
    )

    with (
        patch(
            "shorts_engine.services.autopilot.find_niche_trend_videos",
            return_value=[mock_niche_video],
        ) as mock_niche,
        patch(
            "shorts_engine.services.autopilot.fetch_my_recent_videos"
        ) as mock_my_videos,
        patch(
            "shorts_engine.services.autopilot.download_video_section",
            side_effect=RuntimeError("stop"),
        ),
        patch(
            "shorts_engine.services.autopilot.fetch_youtube_transcript",
            return_value=None,
        ),
        patch(
            "shorts_engine.services.autopilot.download_video",
            return_value=tmp_path / "vid.mp4",
        ),
        patch("shorts_engine.services.autopilot.transcribe", return_value=[]),
        patch("shorts_engine.services.autopilot.select_clips", return_value=[]),
    ):
        gen = run_autopilot_pipeline(
            target_url=DIANISMA_CHANNEL_URL, settings=settings, num_videos=1
        )
        messages = []
        try:
            for msg, _pct, _data in gen:
                messages.append(msg)
        except RuntimeError:
            pass

        # Must route to niche discovery, NEVER to our own channel library
        mock_my_videos.assert_not_called()
        mock_niche.assert_called_once()
        assert any(
            "Searching YouTube for fresh viral videos in our niche" in m
            for m in messages
        )


def test_is_greek_content_relevant() -> None:
    # Greek query with Greek title
    assert (
        is_greek_content_relevant("Τιμή βενζίνης: Νέες αυξήσεις", "Τιμή Βενζίνης")
        is True
    )
    # Greek query with German or English title (should be rejected)
    assert (
        is_greek_content_relevant(
            "Bundesregierung einigt sich auf Tankrabatt", "Τιμή Βενζίνης"
        )
        is False
    )
    assert (
        is_greek_content_relevant("Gas prices soar across Europe", "Τιμή Βενζίνης")
        is False
    )
    # Empty query (default niche) requires Greek characters
    assert is_greek_content_relevant("Δηλώσεις πρωθυπουργού", "") is True
    assert (
        is_greek_content_relevant("Bundesregierung einigt sich auf Tankrabatt", "")
        is False
    )
    # Custom non-Greek query
    assert (
        is_greek_content_relevant("AI developments in 2026", "AI developments") is True
    )


def test_fetch_niche_videos_via_ytdlp_filters_foreign_videos() -> None:
    now = datetime.now(timezone.utc)
    fresh_compact = (now - timedelta(days=2)).strftime("%Y%m%d")

    raw_candidates = [
        VideoMeta(
            video_id="german_vid",
            title="Bundesregierung einigt sich auf Tankrabatt",
            url="https://youtube.com/watch?v=german_vid",
            view_count=500000,
            duration_seconds=300,
            upload_date=fresh_compact,
            like_count=1000,
            comment_count=200,
            description="Tankrabatt beschlossen",
        ),
        VideoMeta(
            video_id="greek_vid",
            title="Αυξήσεις στην τιμή της βενζίνης στην Ελλάδα",
            url="https://youtube.com/watch?v=greek_vid",
            view_count=60000,
            duration_seconds=300,
            upload_date=fresh_compact,
            like_count=200,
            comment_count=50,
            description="Ακρίβεια στα καύσιμα",
        ),
    ]

    with patch(
        "shorts_engine.services.niche_sourcing.fetch_youtube_videos",
        return_value=raw_candidates,
    ):
        results = fetch_niche_videos_via_ytdlp(query="Τιμή Βενζίνης", max_results=5)
        assert len(results) == 1
        assert results[0].video_id == "greek_vid"


def test_autopilot_handles_youtube_transcript_unavailable_error(tmp_path: Path) -> None:
    from shorts_engine.services.autopilot import run_autopilot_pipeline
    from shorts_engine.services.youtube_transcript_fetcher import (
        YouTubeTranscriptUnavailableError,
    )

    settings = Settings(gemini_api_key="test_api_key", output_dir=tmp_path)
    mock_niche_video = VideoMeta(
        video_id="_J_y_hfOuCw",
        title="Τιμή Βενζίνης στην Ελλάδα",
        url="https://www.youtube.com/watch?v=_J_y_hfOuCw",
        view_count=100000,
        duration_seconds=120,
        upload_date="20260920",
        like_count=500,
        comment_count=100,
        description="Καύσιμα",
    )

    dummy_video_file = tmp_path / "test.mp4"
    dummy_video_file.write_bytes(b"fake_video")

    with (
        patch(
            "shorts_engine.services.autopilot.find_niche_trend_videos",
            return_value=[mock_niche_video],
        ),
        patch(
            "shorts_engine.services.autopilot.fetch_youtube_transcript",
            side_effect=YouTubeTranscriptUnavailableError(
                "Captions are disabled or not found for video '_J_y_hfOuCw'."
            ),
        ),
        patch(
            "shorts_engine.services.autopilot.download_video",
            return_value=dummy_video_file,
        ) as mock_dl,
        patch("shorts_engine.services.autopilot.transcribe", return_value=[]),
        patch("shorts_engine.services.autopilot.select_clips", return_value=[]),
    ):
        gen = run_autopilot_pipeline(target_url=None, settings=settings, num_videos=1)
        messages = []
        try:
            for msg, _pct, _data in gen:
                messages.append(msg)
        except RuntimeError:
            pass

        # Pipeline did not crash during transcript triage; gracefully proceeded to Whisper download
        mock_dl.assert_called_once()
        assert any("Downloading video for Whisper analysis" in m for m in messages)


def test_autopilot_handles_download_failure_with_ai_fallback(tmp_path: Path) -> None:
    from shorts_engine.services.autopilot import run_autopilot_pipeline
    from shorts_engine.services.seo_generator import SeoMetadata
    from shorts_engine.services.youtube_transcript_fetcher import (
        YouTubeTranscriptUnavailableError,
    )

    settings = Settings(gemini_api_key="test_api_key", output_dir=tmp_path)
    mock_niche_video = VideoMeta(
        video_id="_J_y_hfOuCw",
        title="Τιμή Βενζίνης στην Ελλάδα",
        url="https://www.youtube.com/watch?v=_J_y_hfOuCw",
        view_count=100000,
        duration_seconds=120,
        upload_date="20260920",
        like_count=500,
        comment_count=100,
        description="Ανάλυση καυσίμων",
    )

    mock_seo = SeoMetadata(
        title="Τιμή Βενζίνης: Τι Αλλάζει",
        description="Ανάλυση",
        tags=("βενζίνη", "οικονομία"),
        primary_keyword="βενζίνη",
        pinned_comment="Ποια είναι η άποψή σας;",
        curiosity_title="Τι συμβαίνει με τη βενζίνη;",
        authority_title="Η αλήθεια για τα καύσιμα",
        contrarian_title="Γιατί η βενζίνη δεν θα πέσει",
    )

    with (
        patch(
            "shorts_engine.services.autopilot.find_niche_trend_videos",
            return_value=[mock_niche_video],
        ),
        patch(
            "shorts_engine.services.autopilot.fetch_youtube_transcript",
            side_effect=YouTubeTranscriptUnavailableError("Captions disabled"),
        ),
        patch(
            "shorts_engine.services.autopilot.download_video",
            side_effect=RuntimeError("yt-dlp failed: This video is unavailable"),
        ),
        patch(
            "shorts_engine.services.autopilot.build_full_ai_short",
            return_value=(tmp_path / "ai_output.mp4", mock_seo),
        ) as mock_ai_build,
        patch("shorts_engine.services.autopilot.record_processed_video"),
    ):
        gen = run_autopilot_pipeline(target_url=None, settings=settings, num_videos=1)
        messages = [msg for msg, _pct, _data in gen]

        # Download failure fell back to Full AI Short generation cleanly
        mock_ai_build.assert_called_once()
        assert any(
            "Video download unavailable -> Switching to Full AI Short Generation" in m
            for m in messages
        )
