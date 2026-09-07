"""
tests/test_highlight_scorer.py — Unit tests for Qwen 2.5 highlight scoring & fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shorts_engine.services.highlight_scorer import (
    HighlightCut,
    _call_qwen_api,
    _format_segments_for_scoring,
    extract_clip_segment,
    score_highlight,
)
from shorts_engine.services.transcriber import TranscriptionSegment


def _make_dummy_segments(count: int, start_time: float = 0.0, duration_per_seg: float = 5.0) -> list[TranscriptionSegment]:
    segments = []
    t = start_time
    for i in range(count):
        segments.append(
            TranscriptionSegment(
                start=round(t, 2),
                end=round(t + duration_per_seg, 2),
                text=f"Πρόταση {i + 1} για το βίντεο.",
                words=[
                    (round(t, 2), round(t + 2.0, 2), f"Πρόταση{i+1}"),
                    (round(t + 2.0, 2), round(t + duration_per_seg, 2), "βίντεο"),
                ],
            )
        )
        t += duration_per_seg
    return segments


def test_format_segments_for_scoring():
    segs = _make_dummy_segments(2, start_time=0.0, duration_per_seg=3.0)
    formatted = _format_segments_for_scoring(segs)
    assert "[0.00s - 3.00s]" in formatted
    assert "[3.00s - 6.00s]" in formatted


def test_score_highlight_short_video_returns_full():
    # If video is already <= max_duration (e.g. 40s <= 60s), keep full video
    segs = _make_dummy_segments(8, start_time=0.0, duration_per_seg=5.0)  # 40s
    cut = score_highlight(segs, total_duration=40.0, max_duration=60.0)
    assert cut.start_time == 0.0
    assert cut.end_time == 40.0
    assert cut.virality_score == 8.5


@patch("shorts_engine.services.highlight_scorer.requests.post")
def test_score_highlight_qwen_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "start_time": 15.0,
                        "end_time": 45.0,
                        "hook_text": "Αυτό είναι το μυστικό για virality!",
                        "virality_score": 9.4,
                        "reasoning": "Strong opening problem-statement hook with high viewer retention.",
                    })
                }
            }
        ]
    }
    mock_post.return_value = mock_resp

    segs = _make_dummy_segments(24, start_time=0.0, duration_per_seg=5.0)  # 120s video
    cut = score_highlight(
        segments=segs,
        total_duration=120.0,
        api_base="http://localhost:11434/v1",
        model="qwen2.5:32b",
        min_duration=30.0,
        max_duration=60.0,
    )

    assert cut.start_time == 15.0
    assert cut.end_time == 45.0
    assert cut.hook_text == "Αυτό είναι το μυστικό για virality!"
    assert cut.virality_score == 9.4
    assert "retention" in cut.reasoning.lower()


@patch("shorts_engine.services.highlight_scorer.requests.post", side_effect=Exception("Connection refused"))
def test_score_highlight_qwen_failure_heuristic_fallback(mock_post):
    # When Qwen fails and no gemini_api_key is given, fallback to heuristic opening 45s
    segs = _make_dummy_segments(20, start_time=0.0, duration_per_seg=5.0)  # 100s
    cut = score_highlight(
        segments=segs,
        total_duration=100.0,
        min_duration=30.0,
        max_duration=60.0,
        gemini_api_key="",
    )
    assert cut.start_time == 0.0
    assert cut.end_time == 45.0
    assert cut.virality_score == 6.0


def test_extract_clip_segment(tmp_path):
    import subprocess
    input_file = tmp_path / "synthetic_in.mp4"
    output_file = tmp_path / "synthetic_out.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:d=4.0",
        "-c:v", "libx264", "-preset", "ultrafast",
        str(input_file),
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    res = extract_clip_segment(
        input_path=input_file,
        output_path=output_file,
        start_time=1.0,
        end_time=3.0,
    )
    assert res.is_file()
    assert res.stat().st_size > 0
