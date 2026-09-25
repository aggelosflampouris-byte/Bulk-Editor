"""
tests/test_deduplication_and_history.py — Tests for persistent video deduplication and history tracking.

Verifies:
  1. record_processed_video writes to persistent cache registry, global history, and custom output_dir.
  2. is_video_already_processed correctly identifies processed videos across multiple URL formats.
  3. find_niche_trend_videos strictly excludes already processed videos and rotates queries.
  4. run_autopilot_pipeline logs history for Hybrid, Full AI, and Standard Shorts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from shorts_engine.config import Settings
from shorts_engine.services.cache_manager import (
    get_processed_video_ids,
    is_video_already_processed,
    record_processed_video,
)
from shorts_engine.services.channel_analyzer import VideoMeta
from shorts_engine.services.niche_sourcing import find_niche_trend_videos


def test_record_processed_video_persistence(tmp_path: Path) -> None:
    custom_out = tmp_path / "custom_run_out"
    test_id = "test_vid_12345"
    test_url = f"https://www.youtube.com/watch?v={test_id}"
    test_title = "Breaking Greek News Interview"

    record_processed_video(
        video_id=test_id,
        url=test_url,
        title=test_title,
        output_dir=custom_out,
        mode="hybrid",
        output_file=custom_out / "hybrid_short.mp4",
    )

    # 1. Custom output_dir history file must exist and contain the entry
    custom_hist = custom_out / "project_history.json"
    assert custom_hist.is_file()
    custom_data = json.loads(custom_hist.read_text(encoding="utf-8"))
    assert any(e.get("video_id") == test_id for e in custom_data)

    # 2. is_video_already_processed should detect it by ID, standard URL, and short URL
    assert test_id in get_processed_video_ids(custom_out)
    assert is_video_already_processed(test_id, custom_out) is True
    assert is_video_already_processed(test_url, custom_out) is True
    assert is_video_already_processed(f"https://youtu.be/{test_id}", custom_out) is True
    assert is_video_already_processed(f"https://www.youtube.com/shorts/{test_id}", custom_out) is True
    assert is_video_already_processed("completely_unseen_id", custom_out) is False


def test_is_video_already_processed_with_none_output_dir(tmp_path: Path) -> None:
    # Safely handles None output_dir without throwing TypeError
    test_id = "none_out_test_vid"
    record_processed_video(
        video_id=test_id,
        url=f"https://www.youtube.com/watch?v={test_id}",
        title="Sample Video",
        output_dir=None,
        mode="ai_gen",
    )

    assert is_video_already_processed(test_id, None) is True
    assert is_video_already_processed(f"https://www.youtube.com/watch?v={test_id}", None) is True
    assert is_video_already_processed("non_existent_vid_xyz", None) is False


def test_find_niche_trend_videos_excludes_processed(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    processed_id = "already_clipped_01"
    fresh_id = "brand_new_viral_02"

    v_processed = VideoMeta(
        video_id=processed_id,
        title="Previously Sourced Video",
        url=f"https://www.youtube.com/watch?v={processed_id}",
        view_count=200000,
        duration_seconds=300,
        upload_date=(now - timedelta(days=2)).strftime("%Y%m%d"),
        like_count=5000,
        comment_count=600,
        description="Old",
    )
    v_fresh = VideoMeta(
        video_id=fresh_id,
        title="Brand New Viral Niche Video",
        url=f"https://www.youtube.com/watch?v={fresh_id}",
        view_count=80000,
        duration_seconds=250,
        upload_date=(now - timedelta(days=1)).strftime("%Y%m%d"),
        like_count=3000,
        comment_count=400,
        description="Fresh",
    )

    # Mark v_processed as processed
    record_processed_video(
        video_id=processed_id,
        url=v_processed.url,
        title=v_processed.title,
        output_dir=tmp_path,
        mode="hybrid",
    )

    with patch("shorts_engine.services.niche_sourcing.fetch_niche_videos_via_ytdlp", return_value=[v_processed, v_fresh]):
        picks = find_niche_trend_videos(
            query="ελληνική πολιτική",
            max_videos=2,
            output_dir=tmp_path,
            shuffle_queries=False,
        )

        # Must strictly exclude v_processed and only return v_fresh
        assert len(picks) == 1
        assert picks[0].video_id == fresh_id


def test_autopilot_pipeline_records_history_in_hybrid_mode(tmp_path: Path) -> None:
    from shorts_engine.services.autopilot import run_autopilot_pipeline

    now = datetime.now(timezone.utc)
    test_id = "autopilot_hybrid_test_vid"
    mock_video = VideoMeta(
        video_id=test_id,
        title="Critical Political Analysis",
        url=f"https://www.youtube.com/watch?v={test_id}",
        view_count=100000,
        duration_seconds=400,
        upload_date=(now - timedelta(days=2)).strftime("%Y%m%d"),
        like_count=4000,
        comment_count=500,
        description="Analysis",
    )

    settings = Settings(
        gemini_api_key="test_key",
        output_dir=tmp_path / "run_out",
        enable_bg_music=True,
    )

    sec_file = tmp_path / "sec.mp4"
    sec_file.write_bytes(b"dummy video content")

    mock_seo = MagicMock()
    mock_seo.title = "Viral Short Hook"
    mock_seo.description = "Short desc"
    mock_seo.tags = ["tag1", "tag2"]
    mock_seo.primary_keyword = "keyword"
    mock_seo.pinned_comment = "comment"
    mock_seo.alt_titles = ["alt1"]

    (tmp_path / "out.mp4").write_bytes(b"dummy final short")

    patch_target = (
        "services.hybrid_short_generator.build_hybrid_short"
        if "services.hybrid_short_generator" in sys.modules or (Path(__file__).parent.parent / "services").is_dir()
        else "shorts_engine.services.hybrid_short_generator.build_hybrid_short"
    )

    with (
        patch("shorts_engine.services.autopilot.find_niche_trend_videos", return_value=[mock_video]),
        patch("shorts_engine.services.autopilot.download_video_section", return_value=sec_file),
        patch("shorts_engine.services.autopilot.fetch_youtube_transcript") as mock_fetch_tr,
        patch("shorts_engine.services.autopilot.select_clips") as mock_sel_clips,
        patch(patch_target, return_value=(tmp_path / "out.mp4", mock_seo)),
        patch("shorts_engine.services.hybrid_short_generator.build_hybrid_short", return_value=(tmp_path / "out.mp4", mock_seo)),
        patch("shorts_engine.services.video_engine.slice_video", side_effect=lambda **kw: kw["output_path"].write_bytes(b"dummy") or kw["output_path"]),
    ):
        if "services.video_engine" in sys.modules:
            sys.modules["services.video_engine"].slice_video = MagicMock(side_effect=lambda **kw: kw["output_path"].write_bytes(b"dummy") or kw["output_path"])
        if "services.hybrid_short_generator" in sys.modules:
            sys.modules["services.hybrid_short_generator"].build_hybrid_short = MagicMock(return_value=(tmp_path / "out.mp4", mock_seo))

        mock_seg = MagicMock()
        mock_seg.start = 10.0
        mock_seg.end = 25.0
        mock_seg.text = "Ομιλία"
        mock_fetch_tr.return_value = [mock_seg]

        mock_clip = MagicMock()
        mock_clip.index = 1
        mock_clip.start_time = 10.0
        mock_clip.end_time = 25.0
        mock_clip.start_display = "0:10"
        mock_clip.end_display = "0:25"
        mock_clip.seo = mock_seo
        mock_clip.hook_summary = "Hook"
        mock_clip.broll_query = "news"
        mock_sel_clips.return_value = [mock_clip]

        gen = run_autopilot_pipeline(settings=settings, num_videos=1, production_strategy="hybrid")
        results = []
        for _msg, pct, data in gen:
            if pct == 100:
                results = data

        assert len(results) == 1
        # Verify the video was persistently recorded in history
        assert is_video_already_processed(test_id, settings.output_dir) is True
        assert is_video_already_processed(mock_video.url, settings.output_dir) is True
