"""
tests/test_timeline_utils.py — Unit tests for silence snapping, clip structure, and word boundary protection.
"""

from __future__ import annotations

import pytest

from shorts_engine.services.timeline_utils import (
    format_transcript_with_timestamps,
    slice_segments,
    snap_to_silence,
)
from shorts_engine.services.transcriber import TranscriptionSegment


def test_snap_to_silence_sentence_and_pause_boundaries():
    # Segments representing continuous speech with natural pauses and punctuation
    segments = [
        TranscriptionSegment(
            start=10.0,
            end=14.0,
            text="Αυτό είναι το πρώτο μέρος.",
            words=[
                (10.0, 10.8, "Αυτό"),
                (10.9, 11.5, "είναι"),
                (11.6, 12.2, "το"),
                (12.3, 13.0, "πρώτο"),
                (13.1, 14.0, "μέρος."),
            ],
        ),
        # 1.5s pause between 14.0 and 15.5
        TranscriptionSegment(
            start=15.5,
            end=48.0,
            text="Εδώ ξεκινάει το δυνατό hook που αλλάζει τα πάντα στο βίντεο.",
            words=[
                (15.5, 16.0, "Εδώ"),
                (16.1, 17.2, "ξεκινάει"),
                (17.3, 17.8, "το"),
                (17.9, 18.5, "δυνατό"),
                (18.6, 19.2, "hook"),
                (45.0, 46.0, "στα"),
                (46.1, 48.0, "πάντα."),
            ],
        ),
        # 2.0s pause between 48.0 and 50.0
        TranscriptionSegment(
            start=50.0,
            end=55.0,
            text="Επόμενη πρόταση.",
            words=[
                (50.0, 52.0, "Επόμενη"),
                (52.1, 55.0, "πρόταση."),
            ],
        ),
    ]

    # AI suggested cut: [15.0, 48.5] (inside the pauses)
    snapped_start, snapped_end = snap_to_silence(
        start_time=15.0,
        end_time=48.5,
        segments=segments,
        min_dur=20.0,
        max_dur=40.0,
    )

    # Start should snap to the start of the speech (around 15.5s with pre-roll buffer)
    assert 15.0 <= snapped_start <= 15.55
    # End should snap cleanly after the sentence ends at 48.0 (with post-roll buffer)
    assert 48.0 <= snapped_end <= 48.20
    # Duration must be within bounds
    duration = snapped_end - snapped_start
    assert 20.0 <= duration <= 40.0


def test_snap_to_silence_never_cuts_mid_word():
    # Single segment where AI suggests cutting right in the middle of word "σπουδαίο" (12.0s - 13.0s)
    segments = [
        TranscriptionSegment(
            start=10.0,
            end=25.0,
            text="Είναι ένα πολύ σπουδαίο μήνυμα.",
            words=[
                (10.0, 10.5, "Είναι"),
                (10.6, 11.0, "ένα"),
                (11.1, 11.8, "πολύ"),
                (12.0, 13.0, "σπουδαίο"),
                (13.1, 14.5, "μήνυμα."),
                (14.6, 25.0, "Συνέχεια."),
            ],
        )
    ]

    # Target cut starts at 12.4s (right inside "σπουδαίο" 12.0 - 13.0)
    snapped_start, snapped_end = snap_to_silence(
        start_time=12.4,
        end_time=25.0,
        segments=segments,
        min_dur=10.0,
        max_dur=20.0,
    )

    # Must NOT start inside [12.0, 13.0]
    assert snapped_start < 12.05 or snapped_start >= 13.0


def test_slice_segments_rebasing():
    segments = [
        TranscriptionSegment(start=10.0, end=15.0, text="Πρώτο"),
        TranscriptionSegment(start=16.0, end=25.0, text="Δεύτερο"),
        TranscriptionSegment(start=30.0, end=35.0, text="Τρίτο"),
    ]

    sliced = slice_segments(segments, start_time=10.0, end_time=25.0)
    assert len(sliced) == 2
    assert sliced[0].start == 0.0
    assert sliced[0].end == 5.0
    assert sliced[1].start == 6.0
    assert sliced[1].end == 15.0


def test_snap_to_silence_greek_question_mark_and_ellipsis():
    segments = [
        TranscriptionSegment(
            start=0.0,
            end=10.0,
            text="Πώς μπορείς να το πετύχεις αυτό; Είναι απλό…",
            words=[
                (0.0, 1.0, "Πώς"),
                (1.1, 2.0, "μπορείς"),
                (2.1, 2.8, "να"),
                (2.9, 3.5, "το"),
                (3.6, 4.8, "πετύχεις"),
                (4.9, 5.8, "αυτό;"),
                (6.5, 7.5, "Είναι"),
                (7.6, 9.5, "απλό…"),
            ],
        ),
    ]

    # Target end around 5.9 (close to Greek question mark 'αυτό;')
    snapped_start, snapped_end = snap_to_silence(
        start_time=0.0,
        end_time=5.9,
        segments=segments,
        min_dur=4.0,
        max_dur=10.0,
    )
    # Start should snap to 0.0 (or pre-roll capped at 0.0)
    assert snapped_start == 0.0
    # End should snap cleanly after 'αυτό;' (5.8 + post-roll <= 6.0)
    assert 5.8 <= snapped_end <= 6.0


def test_snap_to_silence_duration_clamping():
    segments = [
        TranscriptionSegment(
            start=0.0,
            end=100.0,
            text="Μεγάλη πρόταση.",
            words=[
                (float(i), float(i) + 0.8, f"λέξη_{i}") for i in range(100)
            ],
        )
    ]

    # Model requests 80s duration, exceeding max_dur=45.0
    snapped_start, snapped_end = snap_to_silence(
        start_time=10.0,
        end_time=90.0,
        segments=segments,
        min_dur=20.0,
        max_dur=45.0,
    )
    dur = snapped_end - snapped_start
    assert 20.0 <= dur <= 45.01


def test_compute_zoom_intervals_single_segment():
    from shorts_engine.services.compositor import compute_zoom_intervals

    # Single continuous segment spanning 42 seconds
    segments = [
        TranscriptionSegment(
            start=0.0,
            end=42.0,
            text="Ομιλία χωρίς διακοπή.",
            words=[],
        )
    ]

    intervals = compute_zoom_intervals(42.0, segments=segments)
    # Must produce multiple alternating zoom intervals
    assert len(intervals) >= 4
    for start, end in intervals:
        assert 0.0 <= start < end <= 42.0
        assert (end - start) >= 0.8


def test_compute_zoom_intervals_empty_segments_fallback():
    from shorts_engine.services.compositor import compute_zoom_intervals

    # Empty segments should still produce alternating zoom intervals
    intervals = compute_zoom_intervals(20.0, segments=[])
    assert len(intervals) >= 2
    for start, end in intervals:
        assert 0.0 <= start < end <= 20.0


def test_compute_zoom_intervals_with_rebased_offset():
    from shorts_engine.services.compositor import compute_zoom_intervals

    # Pre-rebased segments (start at 0.0) but clip_start_offset was 150.0
    segments = [
        TranscriptionSegment(
            start=0.5,
            end=30.0,
            text="Απόσπασμα ήδη χρονισμένο.",
            words=[],
        )
    ]
    intervals = compute_zoom_intervals(30.0, segments=segments, clip_start_offset=150.0)
    assert len(intervals) >= 3
    for start, end in intervals:
        assert 0.0 <= start < end <= 30.0


def test_apply_dynamic_zoom_ffmpeg_calls_correct_overlay_filter(tmp_path: Path):
    from unittest.mock import patch, MagicMock
    from shorts_engine.services.compositor import apply_dynamic_zoom_ffmpeg

    src_video = tmp_path / "in.mp4"
    src_video.write_bytes(b"dummy")
    out_video = tmp_path / "out.mp4"

    intervals = [(1.5, 4.5), (8.0, 11.5)]

    mock_run = MagicMock()
    mock_run.returncode = 0

    with patch("subprocess.run", return_value=mock_run) as p_run:
        result = apply_dynamic_zoom_ffmpeg(
            video_path=src_video,
            output_path=out_video,
            zoom_intervals=intervals,
            target_width=1080,
            target_height=1920,
            zoom_factor=1.15,
        )

        assert result == out_video
        assert p_run.called
        cmd = p_run.call_args[0][0]
        # Verify overlay and filtergraph
        cmd_str = " ".join(cmd)
        assert "overlay=0:0:enable=" in cmd_str
        assert "between(t,1.50,4.50)" in cmd_str
        assert "between(t,8.00,11.50)" in cmd_str
        assert "split=2" in cmd_str
        assert "scale=1080:1920" in cmd_str


