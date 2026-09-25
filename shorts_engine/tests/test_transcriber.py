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
    # Verify centisecond rounding does not produce 3-digit overflow (e.g. 59.100)
    assert _seconds_to_ass_time(59.996) == "0:01:00.00"
    assert _seconds_to_ass_time(1.999) == "0:00:02.00"


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
    ass = segments_to_ass(segments, primary_keyword="Ελλάδα", subtitle_mode="word")
    # Verify HighlightBox border box for spoken word
    assert r"{\rHighlightBox}Ελλάδα{\rDefault}" in ass


def test_segments_to_ass_dynamic_mode_highlight() -> None:
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
    ass = segments_to_ass(segments, subtitle_mode="dynamic")
    assert r"{\rHighlightBox}Ελλάδα{\rDefault}" in ass


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


def _ass_time_to_seconds(ts: str) -> float:
    h, m, s_cs = ts.split(":")
    s, cs = s_cs.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0


def test_segments_to_ass_zero_dialogue_overlap_word_mode() -> None:
    # Test realistic scenario with overlapping segments and short-interval words
    segments = [
        TranscriptionSegment(
            start=1.0,
            end=3.5,
            text="Πώς τα καρτέλ κλέβουν",
            words=[
                (1.0, 1.15, "Πώς"),
                (1.15, 1.25, "τα"),  # Short duration (0.10s) clamped to min_word_duration or next_start
                (1.22, 1.80, "καρτέλ"),  # Starts before prev word display_end would have finished
                (1.75, 2.20, "κλέβουν"),
            ],
        ),
        TranscriptionSegment(
            start=2.10,
            end=4.0,
            text="τα χρήματά σου",
            words=[
                (2.10, 2.25, "τα"),
                (2.30, 2.90, "χρήματά"),
                (2.90, 3.40, "σου"),
            ],
        ),
    ]

    ass = segments_to_ass(segments, subtitle_mode="word")
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) >= 6

    for i in range(1, len(dialogues)):
        prev_end = _ass_time_to_seconds(dialogues[i - 1].split(",")[2])
        curr_start = _ass_time_to_seconds(dialogues[i].split(",")[1])
        assert curr_start >= prev_end, f"Overlap detected between lines {i-1} and {i}: {prev_end} > {curr_start}"


def test_segments_to_ass_zero_dialogue_overlap_phrase_mode() -> None:
    segments = [
        TranscriptionSegment(
            start=0.0,
            end=4.0,
            text="η κατανομή των τιμών στην αγορά",
            words=[
                (0.0, 0.4, "η"),
                (0.35, 0.9, "κατανομή"),
                (0.85, 1.3, "των"),
                (1.25, 1.8, "τιμών"),
                (1.75, 2.1, "στην"),
                (2.05, 2.8, "αγορά"),
            ],
        )
    ]
    ass = segments_to_ass(segments, subtitle_mode="phrase")
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) >= 2

    for i in range(1, len(dialogues)):
        prev_end = _ass_time_to_seconds(dialogues[i - 1].split(",")[2])
        curr_start = _ass_time_to_seconds(dialogues[i].split(",")[1])
        assert curr_start >= prev_end, f"Overlap detected between phrase lines {i-1} and {i}: {prev_end} > {curr_start}"


def test_segments_to_ass_zero_dialogue_overlap_dynamic_mode() -> None:
    segments = [
        TranscriptionSegment(
            start=0.0,
            end=4.0,
            text="η κατανομή των τιμών στην αγορά",
            words=[
                (0.0, 0.4, "η"),
                (0.35, 0.9, "κατανομή"),
                (0.85, 1.3, "των"),
                (1.25, 1.8, "τιμών"),
                (1.75, 2.1, "στην"),
                (2.05, 2.8, "αγορά"),
            ],
        )
    ]
    ass = segments_to_ass(segments, subtitle_mode="dynamic")
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) >= 6
    assert any(r"{\rHighlightBox}" in line for line in dialogues)

    for i in range(1, len(dialogues)):
        prev_end = _ass_time_to_seconds(dialogues[i - 1].split(",")[2])
        curr_start = _ass_time_to_seconds(dialogues[i].split(",")[1])
        assert curr_start >= prev_end, f"Overlap detected between dynamic lines {i-1} and {i}: {prev_end} > {curr_start}"


def test_segments_to_ass_zero_dialogue_overlap_fallback_mode() -> None:
    # Segments that have overlapping timestamps from rolling teletext
    segments = [
        TranscriptionSegment(start=1.0, end=4.5, text="Πρώτη πρόταση."),
        TranscriptionSegment(start=3.0, end=6.5, text="Δεύτερη πρόταση."),
        TranscriptionSegment(start=5.5, end=8.0, text="Τρίτη πρόταση."),
    ]
    ass = segments_to_ass(segments)
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) == 3

    for i in range(1, len(dialogues)):
        prev_end = _ass_time_to_seconds(dialogues[i - 1].split(",")[2])
        curr_start = _ass_time_to_seconds(dialogues[i].split(",")[1])
        assert curr_start >= prev_end, f"Fallback overlap detected between lines {i-1} and {i}: {prev_end} > {curr_start}"


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


def test_is_hallucinated_text() -> None:
    from shorts_engine.services.transcriber import is_hallucinated_text

    # Standard Greek Whisper hallucinations
    assert is_hallucinated_text("Υπότιτλοι:") is True
    assert is_hallucinated_text("Ευχαριστούμε που παρακολουθήσατε") is True
    assert is_hallucinated_text("Κάντε like και subscribe") is True
    assert is_hallucinated_text("Subtitles by") is True

    # Repetition stutter loops
    assert is_hallucinated_text("και και και και") is True
    assert is_hallucinated_text("της της της της") is True

    # Silence glitch (long duration with 1 character)
    assert is_hallucinated_text("α", duration=3.5) is True

    # Authentic speech
    assert is_hallucinated_text("Οι λογαριασμοί ρεύματος αυξήθηκαν κατακόρυφα.") is False
    assert is_hallucinated_text("Σύμφωνα με τα επίσημα στοιχεία της ΕΛΣΤΑΤ.") is False


def test_dynamic_subtitle_chunking_avoids_dangling_particles() -> None:
    from shorts_engine.services.transcriber import TranscriptionSegment, segments_to_ass

    # Sentence where "των" would dangle if blindly cut at 5 words
    words = [
        (0.0, 0.4, "όμιλοι"),
        (0.4, 0.9, "θησαυρίζουν"),
        (0.9, 1.2, "στις"),
        (1.2, 1.6, "πλάτες"),
        (1.6, 1.9, "των"),
        (1.9, 2.5, "καταναλωτών"),
    ]
    seg = TranscriptionSegment(start=0.0, end=2.5, text="όμιλοι θησαυρίζουν στις πλάτες των καταναλωτών", words=words)
    ass_out = segments_to_ass([seg], subtitle_mode="dynamic")
    dialogues = [line for line in ass_out.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) > 0
    # First dialogue line should not have "των" as its last word
    first_line_text = dialogues[0].split(",")[-1]
    # Strip ASS tags
    plain_words = [w for w in first_line_text.replace(r"{\rHighlightBox}", "").replace(r"{\rDefault}", "").split() if w]
    assert plain_words[-1] != "των"

