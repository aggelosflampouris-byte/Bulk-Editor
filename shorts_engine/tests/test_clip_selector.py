"""
tests/test_clip_selector.py — Unit tests for AI clip selection, timestamp normalization, and min_clips enforcement.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shorts_engine.services.clip_selector import (
    ClipCandidate,
    _normalize_clip_times,
    _parse_clips_json,
    _parse_single_clip,
    _parse_time_value,
    _supplement_clips,
    select_clips,
)
from shorts_engine.services.seo_generator import SeoMetadata
from shorts_engine.services.transcriber import TranscriptionSegment


def test_parse_time_value_various_formats():
    assert _parse_time_value(45.5) == 45.5
    assert _parse_time_value(120) == 120.0
    assert _parse_time_value("04:45") == 285.0
    assert _parse_time_value("12:34") == 754.0
    assert _parse_time_value("01:02:03") == 3723.0
    assert _parse_time_value("  10:14.500  ") == 614.5


def test_normalize_clip_times_decimal_minutes():
    # Model returns 4.45 (meaning 4m 45s = 285s) and 5.33 (meaning 5m 33s = 333s)
    # Raw diff is 0.88s (< min_dur 35s), normalized diff is 48s (matches 35-50s)
    start, end = _normalize_clip_times(4.45, 5.33, min_dur=35.0, max_dur=50.0)
    assert start == 285.0
    assert end == 333.0
    assert 35.0 <= (end - start) <= 50.0


def test_normalize_clip_times_mmss_strings():
    start, end = _normalize_clip_times("10:14", "10:52", min_dur=35.0, max_dur=50.0)
    assert start == 614.0
    assert end == 652.0
    assert end - start == 38.0


def test_parse_single_clip_clamps_duration():
    raw = {
        "start_time": "01:00",
        "end_time": "02:10",  # 70s duration > max_dur (50s)
        "hook_summary": "Great hook",
        "seo": {
            "title": "Τίτλος",
            "description": "Περιγραφή",
            "tags": ["tag1", "tag2"],
        },
        "broll_query": "greek flag",
    }
    clip = _parse_single_clip(raw, index=1, min_dur=35.0, max_dur=50.0)
    assert clip is not None
    assert clip.start_time == 60.0
    assert clip.end_time == 110.0  # Clamped to 60 + 50
    assert clip.duration == 50.0


def test_supplement_clips_guarantees_minimum_clips():
    # Source video duration 300 seconds (5 minutes)
    segments = [
        TranscriptionSegment(start=i * 10.0, end=(i + 1) * 10.0, text=f"Κείμενο {i}")
        for i in range(30)
    ]
    # Initially only 1 clip found at 100-140s
    existing = [
        ClipCandidate(
            index=1,
            start_time=100.0,
            end_time=140.0,
            hook_summary="Only clip found",
            seo=SeoMetadata.fallback("Only clip"),
            broll_query="crowd talking",
        )
    ]

    supplemented = _supplement_clips(
        existing_clips=existing,
        segments=segments,
        target_count=3,
        min_dur=35.0,
        max_dur=50.0,
    )

    assert len(supplemented) >= 3
    # Verify no overlaps among supplemented clips
    for i in range(len(supplemented)):
        for j in range(i + 1, len(supplemented)):
            c1, c2 = supplemented[i], supplemented[j]
            assert not (c1.start_time < c2.end_time and c1.end_time > c2.start_time)


def test_select_clips_empty_segments():
    assert select_clips([], "dummy_key") == []


def test_select_clips_fallback_generates_minimum_clips():
    # Long transcript (300s) without API key triggers fallback with >= 3 clips
    segments = [
        TranscriptionSegment(start=i * 10.0, end=(i + 1) * 10.0, text=f"Κείμενο {i}")
        for i in range(30)
    ]
    clips = select_clips(
        segments=segments,
        gemini_api_key="",
        max_clips=10,
        min_clips=3,
        min_dur=35.0,
        max_dur=50.0,
    )
    assert len(clips) >= 3
    for c in clips:
        assert 35.0 <= c.duration <= 50.0


def test_select_clips_gemini_single_clip_supplements_to_minimum_three():
    # Model returns only 1 clip, select_clips must supplement to at least 3 clips
    segments = [
        TranscriptionSegment(start=i * 5.0, end=(i + 1) * 5.0, text=f"Κείμενο {i}")
        for i in range(40)  # 200s total
    ]
    single_clip_response = (
        '{"clips": [{"start_time": 10.0, "end_time": 50.0, "hook_summary": "Single hook", '
        '"seo": {"title": "Τίτλος 1", "description": "Περιγραφή 1", "tags": ["tag1"]}, '
        '"broll_query": "greek business"}]}'
    )
    with patch("shorts_engine.services.clip_selector.genai.Client"), \
         patch("shorts_engine.services.clip_selector._call_gemini_with_fallback", return_value=single_clip_response):
        clips = select_clips(
            segments=segments,
            gemini_api_key="valid_fake_key",
            min_clips=3,
            max_clips=5,
            min_dur=35.0,
            max_dur=50.0,
        )
        assert len(clips) >= 3
        # Check first is the Gemini-selected clip
        assert clips[0].start_time == 10.0
        assert clips[0].end_time == 50.0


def test_select_clips_compact_duration_guarantees_minimum_three():
    # Even on compact transcripts (e.g. 70s duration), at least 3 clips are generated
    segments = [
        TranscriptionSegment(start=i * 5.0, end=(i + 1) * 5.0, text=f"Λέξη {i}")
        for i in range(14)  # 70s total
    ]
    clips = select_clips(
        segments=segments,
        gemini_api_key="",
        min_clips=3,
        max_clips=5,
        min_dur=30.0,
        max_dur=45.0,
    )
    assert len(clips) >= 3
    for c in clips:
        assert c.duration >= 30.0

