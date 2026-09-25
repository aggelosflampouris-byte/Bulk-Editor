"""
tests/test_logic_guardrail.py — Unit tests for logic guardrails and subtitle synchronization.
"""

from unittest.mock import MagicMock, patch

from shorts_engine.services.logic_guardrail import (
    enforce_clip_coherence,
    evaluate_logical_coherence,
    sanitize_voiceover_script,
    validate_text_boundaries,
)
from shorts_engine.services.transcriber import (
    TranscriptionSegment,
    segments_to_ass,
)


def test_validate_text_boundaries_complete_sentence() -> None:
    valid, msg = validate_text_boundaries("Ο πρωθυπουργός ανακοίνωσε νέα μέτρα στήριξης για την οικονομία.")
    assert valid is True
    assert "Valid" in msg


def test_validate_text_boundaries_dangling_end() -> None:
    # Ends on 'και' (dangling conjunction)
    valid, msg = validate_text_boundaries("Ο πρωθυπουργός ανακοίνωσε μέτρα και")
    assert valid is False
    assert "dangling" in msg.lower()


def test_validate_text_boundaries_missing_punctuation() -> None:
    # No terminal punctuation mark
    valid, msg = validate_text_boundaries("Ο πρωθυπουργός ανακοίνωσε νέα μέτρα στήριξης")
    assert valid is False
    assert "punctuation" in msg.lower()


def test_enforce_clip_coherence_trims_dangling_word() -> None:
    segs = [
        TranscriptionSegment(
            start=0.0,
            end=4.0,
            text="Αυτή είναι η κατάσταση και",
            words=[
                (0.0, 0.8, "Αυτή"),
                (0.8, 1.5, "είναι"),
                (1.5, 2.0, "η"),
                (2.0, 3.2, "κατάσταση"),
                (3.2, 4.0, "και"),
            ],
        )
    ]
    _s, e, text = enforce_clip_coherence(segs, 0.0, 4.0)
    # The dangling 'και' should be trimmed
    assert e == 3.2
    assert "και" not in text.split()
    assert "κατάσταση" in text


def test_evaluate_logical_coherence_llm() -> None:
    mock_resp = MagicMock()
    mock_resp.text = '{"makes_sense": true, "score": 9, "reason": "Clear narrative flow", "repaired_script": null}'

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = mock_resp

        res = evaluate_logical_coherence(
            topic_title="Οικονομία",
            part1_text="Η Ελλάδα καταγράφει ανάπτυξη.",
            part2_text="Αυτά είναι τα στοιχεία για το νέο έτος.",
            gemini_api_key="fake_key",
        )

        assert res.makes_sense is True
        assert res.score == 9
        assert "narrative" in res.reason.lower()
        assert res.repaired_script is None


def test_evaluate_logical_coherence_repairs_bad_script() -> None:
    mock_resp = MagicMock()
    mock_resp.text = '{"makes_sense": false, "score": 4, "reason": "Non sequitur", "repaired_script": "Νέα στοιχεία δείχνουν ότι η κατάσταση σταθεροποιείται."}'

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = mock_resp

        res = evaluate_logical_coherence(
            topic_title="Οικονομία",
            part1_text="Η κατάσταση είναι δύσκολη.",
            part2_text="Και μετά πήγε για καφέ.",
            gemini_api_key="fake_key",
        )

        assert res.makes_sense is False
        assert res.score == 4
        assert res.repaired_script == "Νέα στοιχεία δείχνουν ότι η κατάσταση σταθεροποιείται."


def test_segments_to_ass_interpolates_missing_word_timestamps() -> None:
    # Segment 1 has words, Segment 2 has words=None
    # Must interpolate Segment 2 words so subtitles NEVER drop off or get stuck
    segs = [
        TranscriptionSegment(
            start=0.0,
            end=2.0,
            text="Πρώτη πρόταση εδώ.",
            words=[(0.0, 0.6, "Πρώτη"), (0.6, 1.3, "πρόταση"), (1.3, 2.0, "εδώ.")],
        ),
        TranscriptionSegment(
            start=2.5,
            end=4.5,
            text="Δεύτερη πρόταση ακολουθεί.",
            words=None,  # Missing word timestamps!
        ),
    ]

    ass_content = segments_to_ass(segs, subtitle_mode="dynamic")

    # Verify that words from Segment 2 appear in Dialogue lines
    assert "Δεύτερη" in ass_content
    assert "ακολουθεί" in ass_content
    assert "Πρώτη" in ass_content


def test_segments_to_ass_dynamic_karaoke_no_collision_or_negative_duration() -> None:
    # Verify Dialogue lines have strictly increasing start/end and never overlap
    segs = [
        TranscriptionSegment(
            start=0.0,
            end=3.0,
            text="Μία δύο τρεις τέσσερις πέντε λέξεις.",
            words=[
                (0.0, 0.4, "Μία"),
                (0.4, 0.9, "δύο"),
                (0.9, 1.4, "τρεις"),
                (1.4, 2.0, "τέσσερις"),
                (2.0, 2.5, "πέντε"),
                (2.5, 3.0, "λέξεις."),
            ],
        )
    ]

    ass_content = segments_to_ass(segs, subtitle_mode="dynamic")
    lines = [line for line in ass_content.splitlines() if line.startswith("Dialogue:")]
    assert len(lines) == 6

    # Verify valid ASS timestamp formatting
    for line in lines:
        parts = line.split(",", 9)
        t_start = parts[1]
        t_end = parts[2]
        assert t_start != t_end
        assert t_end > t_start


def test_sanitize_voiceover_script_strips_first_person_plural() -> None:
    raw_script = "Όπως ανέφερε ο καλεσμένος μας, τα πράγματα είναι δύσκολα. Γράψτε μας τη γνώμη σας στα σχόλια!"
    cleaned = sanitize_voiceover_script(raw_script)
    assert "ο καλεσμένος μας" not in cleaned
    assert "ο ομιλητής" in cleaned
    assert "γράψτε μας" not in cleaned
    assert "γράψε τη γνώμη σου στα σχόλια" in cleaned

    another = "Πάμε να δούμε τι μας είπε για την υπόθεση."
    cleaned2 = sanitize_voiceover_script(another)
    assert "πάμε να δούμε" not in cleaned2
    assert "δες" in cleaned2
    assert "μας είπε" not in cleaned2

