"""
tests/test_channel_sourcing.py — Unit tests for enhanced channel video sourcing.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from shorts_engine.services.cache_manager import (
    get_processed_video_ids,
    is_video_already_processed,
)
from shorts_engine.services.channel_analyzer import (
    VideoMeta,
    _compute_virality_score,
    fetch_channel_rss_videos,
    find_viral_recent_videos,
)
from shorts_engine.services.downloader import download_video_section


def _make_sample_video(
    video_id: str,
    view_count: int,
    days_old: int = 5,
    likes: int = 100,
    comments: int = 20,
) -> VideoMeta:
    from datetime import datetime, timedelta, timezone

    dt = datetime.now(timezone.utc) - timedelta(days=days_old)
    upload_date = dt.strftime("%Y%m%d")
    return VideoMeta(
        video_id=video_id,
        title=f"Sample Video {video_id}",
        url=f"https://www.youtube.com/watch?v={video_id}",
        view_count=view_count,
        duration_seconds=600,
        upload_date=upload_date,
        like_count=likes,
        comment_count=comments,
        description="Sample description",
    )


def test_compute_virality_score_with_rvr() -> None:
    # A video with 50,000 views on a channel with 10,000 median views (RVR = 5.0)
    video = _make_sample_video("v1", view_count=50000, days_old=3)
    score_outlier = _compute_virality_score(video, days_old=3, rvr=5.0)

    # Same video on a channel with 100,000 median views (RVR = 0.5)
    score_underperformer = _compute_virality_score(video, days_old=3, rvr=0.5)

    assert score_outlier > score_underperformer


def test_find_viral_recent_videos_outlier_tier() -> None:
    # Channel median is around 5,000 views
    videos = [
        _make_sample_video("v1", view_count=5000, days_old=4),
        _make_sample_video("v2", view_count=4500, days_old=6),
        _make_sample_video("v3", view_count=25000, days_old=2),  # 5x median -> Breakout
        _make_sample_video("v4", view_count=10000, days_old=5),  # 2x median -> Outlier
    ]
    results = find_viral_recent_videos(videos)
    assert len(results) == 4
    # The 5x breakout video should be top-ranked
    assert results[0].video.video_id == "v3"
    assert "Breakout" in results[0].virality_label or "Hot" in results[0].virality_label


def test_processed_video_history(tmp_path: Path) -> None:
    history_file = tmp_path / "project_history.json"
    dummy_history = [
        {
            "id": "abc123",
            "input_file": "Source_dQw4w9WgXcQ.mp4",
            "output_file": "clip_01.mp4",
            "source_video_id": "dQw4w9WgXcQ",
        },
        {
            "id": "def456",
            "input_file": "regular_video.mp4",
            "output_file": "clip_02.mp4",
        },
    ]
    history_file.write_text(json.dumps(dummy_history), encoding="utf-8")

    processed = get_processed_video_ids(tmp_path)
    assert "dQw4w9WgXcQ" in processed
    assert is_video_already_processed("dQw4w9WgXcQ", tmp_path)
    assert not is_video_already_processed("unknown_id", tmp_path)


def test_fetch_channel_rss_videos() -> None:
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:yt="http://www.youtube.com/xml/schemas/2015">
      <title>Sample Channel</title>
      <entry>
        <id>yt:video:abc12345678</id>
        <yt:videoId>abc12345678</yt:videoId>
        <title>Greek Politics Breakdown</title>
        <link rel="alternate" href="https://www.youtube.com/watch?v=abc12345678"/>
        <published>2026-03-20T12:00:00+00:00</published>
        <media:group xmlns:media="http://search.yahoo.com/mrss/">
          <media:description>Analysis of latest events.</media:description>
          <media:community>
            <media:statistics views="12450"/>
          </media:community>
        </media:group>
      </entry>
    </feed>
    """
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = sample_xml.encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        videos = fetch_channel_rss_videos("UC1234567890")
        assert len(videos) == 1
        assert videos[0].video_id == "abc12345678"
        assert videos[0].title == "Greek Politics Breakdown"
        assert videos[0].view_count == 12450
        assert videos[0].upload_date == "20260320"


@patch("shorts_engine.services.downloader._run_ytdlp")
def test_download_video_section_args(mock_run: MagicMock, tmp_path: Path) -> None:
    # Create fake downloaded file in tmp_path
    fake_file = tmp_path / "test_section [abc12345678].mp4"
    fake_file.touch()

    res = download_video_section(
        url="https://www.youtube.com/watch?v=abc12345678",
        dest_dir=tmp_path,
        start_time=10.0,
        end_time=55.0,
        padding=2.0,
    )
    assert res == fake_file
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "--download-sections" in cmd
    # start_time - 2.0 = 8.0, end_time + 2.0 = 57.0
    section_arg = cmd[cmd.index("--download-sections") + 1]
    assert "*8.00-57.00" in section_arg or "*8-57" in section_arg or "8.0" in section_arg


def test_autopilot_anticannibalization(tmp_path: Path) -> None:
    from shorts_engine.config import Settings
    from shorts_engine.services.autopilot import run_autopilot_pipeline
    from shorts_engine.services.channel_analyzer import ViralRecentVideo

    # Create dummy history where video_1 is already clipped
    history_file = tmp_path / "project_history.json"
    history_file.write_text(
        json.dumps([{"source_video_id": "done1234567", "input_file": "done1234567.mp4", "output_file": "out.mp4"}]),
        encoding="utf-8",
    )

    settings = Settings(gemini_api_key="test_key", output_dir=tmp_path)

    v1 = _make_sample_video("done1234567", view_count=90000, days_old=1)
    v2 = _make_sample_video("fresh123456", view_count=50000, days_old=2)

    with (
        patch("shorts_engine.services.autopilot.fetch_youtube_videos", return_value=[v1, v2]),
        patch(
            "shorts_engine.services.autopilot.find_viral_recent_videos",
            return_value=[
                ViralRecentVideo(video=v1, days_old=1, virality_score=95.0, virality_label="Breakout 🔥", rvr=3.0),
                ViralRecentVideo(video=v2, days_old=2, virality_score=85.0, virality_label="Outlier 🚀", rvr=2.0),
            ],
        ),
        patch("shorts_engine.services.autopilot.fetch_youtube_transcript", return_value=[]),
        patch("shorts_engine.services.autopilot.download_video") as mock_down,
        patch("shorts_engine.services.autopilot.transcribe", return_value=[]),
        patch("shorts_engine.services.autopilot.select_clips", return_value=[]),
    ):
        mock_down.return_value = tmp_path / "dummy.mp4"
        gen = run_autopilot_pipeline("https://youtube.com/@channel", settings, num_videos=1)
        # Advance through generator
        messages = []
        try:
            for msg, _, _ in gen:
                messages.append(msg)
        except RuntimeError:
            pass  # Expected since no clips are found

        # Verify that fresh123456 was selected for processing instead of done1234567
        assert any("fresh123456" in m for m in messages)
        assert not any("done1234567" in m for m in messages)


def test_autopilot_transcript_triage_uses_section_download(tmp_path: Path) -> None:
    from shorts_engine.config import Settings
    from shorts_engine.services.autopilot import run_autopilot_pipeline
    from shorts_engine.services.channel_analyzer import ViralRecentVideo
    from shorts_engine.services.clip_selector import ClipCandidate
    from shorts_engine.services.seo_generator import SeoMetadata
    from shorts_engine.services.transcriber import TranscriptionSegment

    settings = Settings(gemini_api_key="test_key", output_dir=tmp_path)
    v1 = _make_sample_video("v_fast12345", view_count=20000, days_old=2)

    dummy_segments = [
        TranscriptionSegment(start=10.0, end=40.0, text="Viral insight about Greek history", words=[]),
    ]
    dummy_candidate = ClipCandidate(
        index=1,
        start_time=12.0,
        end_time=38.0,
        hook_summary="Hook",
        seo=SeoMetadata(
            title="Greek Mystery",
            description="desc",
            tags=("greek", "shorts"),
            primary_keyword="greek",
            pinned_comment="Check this out",
            curiosity_title="curiosity",
            authority_title="authority",
            contrarian_title="contrarian",
        ),
        broll_query="athens",
    )

    fake_section_file = tmp_path / "section.mp4"
    fake_section_file.touch()

    with (
        patch("shorts_engine.services.autopilot.fetch_youtube_videos", return_value=[v1]),
        patch(
            "shorts_engine.services.autopilot.find_viral_recent_videos",
            return_value=[ViralRecentVideo(video=v1, days_old=2, virality_score=80.0, virality_label="Hot", rvr=2.0)],
        ),
        patch("shorts_engine.services.autopilot.fetch_youtube_transcript", return_value=dummy_segments),
        patch("shorts_engine.services.autopilot.select_clips", return_value=[dummy_candidate]),
        patch("shorts_engine.services.autopilot.download_video_section", return_value=fake_section_file) as mock_section_down,
        patch("shorts_engine.services.autopilot.download_video") as mock_full_down,
        patch("shorts_engine.services.autopilot.process_url_clip") as mock_process,
    ):
        mock_result = MagicMock()
        mock_result.error = None
        mock_result.output_file = tmp_path / "out.mp4"
        mock_process.return_value = mock_result

        gen = run_autopilot_pipeline("https://youtube.com/@channel", settings, num_videos=1)
        for _ in gen:
            pass

        # Verify that download_video_section was called with start=12.0, end=38.0
        mock_section_down.assert_called_once()
        call_kwargs = mock_section_down.call_args[1]
        assert call_kwargs["start_time"] == 12.0
        assert call_kwargs["end_time"] == 38.0

        # Full video download should NOT have been called!
        mock_full_down.assert_not_called()

        # process_url_clip should have received source_is_section=True
        proc_kwargs = mock_process.call_args[1]
        assert proc_kwargs["source_is_section"] is True
        assert proc_kwargs["source_offset"] == 2.0  # min(12.0, padding=2.0)


def test_normalize_youtube_channel_url() -> None:
    from shorts_engine.services.channel_analyzer import normalize_youtube_channel_url

    # Handle variations
    assert normalize_youtube_channel_url("@DianismaNews") == "https://www.youtube.com/@DianismaNews/videos"
    assert normalize_youtube_channel_url("www.youtube.com/@DianismaNews") == "https://www.youtube.com/@DianismaNews/videos"
    assert normalize_youtube_channel_url("https://www.youtube.com/@DianismaNews") == "https://www.youtube.com/@DianismaNews/videos"
    assert normalize_youtube_channel_url("https://www.youtube.com/@DianismaNews/") == "https://www.youtube.com/@DianismaNews/videos"
    assert normalize_youtube_channel_url("https://www.youtube.com/@DianismaNews/videos") == "https://www.youtube.com/@DianismaNews/videos"

    # Channel IDs
    assert normalize_youtube_channel_url("https://www.youtube.com/channel/UC12345") == "https://www.youtube.com/channel/UC12345/videos"
    assert normalize_youtube_channel_url("www.youtube.com/channel/UC12345") == "https://www.youtube.com/channel/UC12345/videos"

    # Plain text search queries shouldn't append /videos
    assert normalize_youtube_channel_url("Greek politics news") == "Greek politics news"
    assert normalize_youtube_channel_url("") == ""
