"""
tests/test_transcriber.py — Unit tests for transcriber service.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from shorts_engine.services.transcriber import (
    TranscriptionSegment,
    _build_whisper_prompt,
    _escape_ass_text,
    _seconds_to_ass_time,
    extract_speech_audio,
    segments_to_ass,
)


def test_seconds_to_ass_time() -> None:
    assert _seconds_to_ass_time(0.0) == "0:00:00.00"
    assert _seconds_to_ass_time(65.432) == "0:01:05.43"
    assert _seconds_to_ass_time(3661.05) == "1:01:01.05"


def test_escape_ass_text() -> None:
    assert _escape_ass_text("Hello {world}") == r"Hello \{world\}"
    assert _escape_ass_text("Normal Greek text: Καλημέρα") == "Normal Greek text: Καλημέρα"


def test_build_whisper_prompt_without_hint_or_title() -> None:
    prompt = _build_whisper_prompt(None, None)
    assert "Ελληνικά" in prompt
    assert "τόνους" in prompt
    # Ensure no conversational hallucination bait
    assert "Γεια σας. Σήμερα θα μιλήσουμε" not in prompt


def test_build_whisper_prompt_with_title_and_hint() -> None:
    prompt = _build_whisper_prompt("politics", "Οικονομική Κρίση")
    assert prompt.startswith("Θέμα: Οικονομική Κρίση.")
    assert "Πολιτική, κυβέρνηση" in prompt
    assert "Ελληνικά" in prompt


def test_segments_to_ass_fallback_without_words() -> None:
    segments = [
        TranscriptionSegment(start=1.0, end=3.5, text="Πρώτη πρόταση."),
        TranscriptionSegment(start=4.0, end=6.0, text="Δεύτερη πρόταση."),
    ]
    ass = segments_to_ass(segments)
    assert "[Events]" in ass
    assert "Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,Πρώτη πρόταση." in ass
    assert "Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,Δεύτερη πρόταση." in ass


def test_segments_to_ass_word_timing_and_minimum_duration() -> None:
    # A word with tiny duration (0.05s) should be clamped to at least 0.22s
    segments = [
        TranscriptionSegment(
            start=1.0,
            end=2.0,
            text="και αυτό",
            words=[
                (1.0, 1.05, "και"),
                (1.10, 1.80, "αυτό"),
            ],
        )
    ]
    ass = segments_to_ass(segments)
    # "και" starts at 1.00; because next word starts at 1.10 (< 0.35s gap),
    # gap bridging snaps display_end to next_start (1.10) or minimum display duration.
    assert "Dialogue:" in ass
    assert "και" in ass
    assert "αυτό" in ass


def test_segments_to_ass_keyword_highlight() -> None:
    segments = [
        TranscriptionSegment(
            start=0.5,
            end=2.0,
            text="Η Ελλάδα κερδίζει",
            words=[
                (0.5, 0.8, "Η"),
                (0.9, 1.4, "Ελλάδα"),
                (1.5, 2.0, "κερδίζει"),
            ],
        )
    ]
    ass = segments_to_ass(segments, primary_keyword="Ελλάδα")
    # Verify yellow color override for "Ελλάδα"
    assert r"{\c&H00FFFF&}Ελλάδα" in ass


def test_segments_to_ass_phrase_mode() -> None:
    segments = [
        TranscriptionSegment(
            start=0.0,
            end=3.0,
            text="Αυτό είναι ένα τεστ",
            words=[
                (0.0, 0.5, "Αυτό"),
                (0.6, 1.0, "είναι"),
                (1.1, 1.5, "ένα"),
                (1.6, 2.2, "τεστ"),
            ],
        )
    ]
    ass = segments_to_ass(segments, subtitle_mode="phrase")
    assert "[Events]" in ass
    assert "Dialogue:" in ass


@patch("subprocess.run")
def test_extract_speech_audio(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    fake_video = tmp_path / "input.mp4"
    fake_video.touch()
    out_wav = tmp_path / "speech.wav"

    extracted = extract_speech_audio(fake_video, out_wav)
    assert extracted == out_wav
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "-ar" in cmd
    assert "16000" in cmd
    assert "-ac" in cmd
    assert "1" in cmd
    assert "dynaudnorm" in cmd[cmd.index("-af") + 1]
