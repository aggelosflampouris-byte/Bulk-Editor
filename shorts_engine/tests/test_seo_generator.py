"""
tests/test_seo_generator.py — Unit tests for SEO generation and transcript correction helpers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shorts_engine.services.seo_generator import (
    _call_gemini_with_fallback,
    align_words_with_corrected_text,
)


def test_align_words_matching_length():
    orig_words = [
        (0.0, 0.5, "Καλημέρα"),
        (0.5, 1.0, "σας"),
    ]
    corrected = "Καλημέρα σας!"
    aligned = align_words_with_corrected_text(orig_words, corrected, 0.0, 1.0)
    assert len(aligned) == 2
    assert aligned[0] == (0.0, 0.5, "Καλημέρα")
    assert aligned[1] == (0.5, 1.0, "σας!")


def test_align_words_merged_words():
    # Whisper heard 'Χαίρο' and 'με' as 2 separate words; Gemini merged them into 'Χαίρομαι'
    orig_words = [
        (0.0, 0.4, "Χαίρο"),
        (0.4, 0.7, "με"),
        (0.7, 1.2, "πολύ"),
    ]
    corrected = "Χαίρομαι πολύ"
    aligned = align_words_with_corrected_text(orig_words, corrected, 0.0, 1.2)
    assert len(aligned) == 2
    assert aligned[0][2] == "Χαίρομαι"
    assert aligned[1][2] == "πολύ"
    assert aligned[0][0] == 0.0
    assert aligned[1][1] == 1.2
    # Ensure times are continuous
    assert aligned[0][1] == aligned[1][0]


def test_align_words_empty_or_whitespace():
    orig_words = [(0.0, 0.5, "test")]
    assert align_words_with_corrected_text(orig_words, "", 0.0, 0.5) == orig_words


def test_call_gemini_with_fallback():
    mock_client = MagicMock()
    # First model raises 404, second model succeeds
    first_call = True

    def side_effect(model, contents, config):
        nonlocal first_call
        if first_call:
            first_call = False
            raise RuntimeError("404 Not Found")
        mock_resp = MagicMock()
        mock_resp.text = "Success from fallback"
        return mock_resp

    mock_client.models.generate_content.side_effect = side_effect
    res = _call_gemini_with_fallback(mock_client, "test", MagicMock())
    assert res == "Success from fallback"


def test_call_gemini_with_fallback_retries_503_with_backoff():
    mock_client = MagicMock()
    first_call = True

    def side_effect(model, contents, config):
        nonlocal first_call
        if first_call:
            first_call = False
            raise RuntimeError("503 UNAVAILABLE: High demand spike")
        mock_resp = MagicMock()
        mock_resp.text = "Success after 503 retry"
        return mock_resp

    mock_client.models.generate_content.side_effect = side_effect
    with patch("time.sleep") as mock_sleep:
        res = _call_gemini_with_fallback(mock_client, "test", MagicMock())
        assert res == "Success after 503 retry"
        mock_sleep.assert_called_once_with(2.0)


def test_call_gemini_with_fallback_sets_thinking_budget_zero():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Success"
    mock_client.models.generate_content.return_value = mock_resp

    from google.genai import types as genai_types

    config = genai_types.GenerateContentConfig(max_output_tokens=1000)
    assert config.thinking_config is None

    _call_gemini_with_fallback(mock_client, "test", config)


def test_seo_metadata_youtube_tags_display():
    from shorts_engine.services.seo_generator import SeoMetadata

    seo = SeoMetadata(
        title="Τίτλος",
        description="Περιγραφή",
        tags=("Τσίπρας", "πολιτική", "Ελλάδα", "shorts"),
        primary_keyword="keyword",
        pinned_comment="pinned",
        curiosity_title="curiosity",
        authority_title="authority",
        contrarian_title="contrarian",
    )
    assert seo.youtube_tags_display == "Τσίπρας, πολιτική, Ελλάδα, shorts"


def test_validate_seo_dict_tag_normalization():
    from shorts_engine.services.seo_generator import _validate_seo_dict

    data = {
        "title": "Δοκιμαστικός τίτλος",
        "description": "Περιγραφή",
        "tags": ["#Τσίπρας", "πολιτική,", "Ελλάδα", "#shorts", "Τσίπρας"],
    }
    seo = _validate_seo_dict(data)
    assert "Τσίπρας" in seo.tags
    assert "#Τσίπρας" not in seo.tags
    assert "πολιτική" in seo.tags
    assert seo.tags.count("Τσίπρας") == 1
    # Check comma-separated output
    assert seo.youtube_tags_display.startswith("Τσίπρας, πολιτική, Ελλάδα, shorts")


def test_sanitize_filename_greek_and_separators():
    from shorts_engine.services.seo_generator import sanitize_filename

    title = "Αλέξης Τσίπρας: Η νέα εποχή για την Ελλάδα"
    assert sanitize_filename(title) == "Αλέξης Τσίπρας - Η νέα εποχή για την Ελλάδα"


def test_sanitize_filename_prohibited_characters():
    from shorts_engine.services.seo_generator import sanitize_filename

    title = 'Shorts/Reels: "Πώς λειτουργεί" ο αλγόριθμος; <2025> *viral* | νέα'
    expected = "Shorts - Reels - Πώς λειτουργεί ο αλγόριθμος; 2025 viral - νέα"
    assert sanitize_filename(title) == expected


def test_sanitize_filename_whitespace_and_dots():
    from shorts_engine.services.seo_generator import sanitize_filename

    title = "  ...Το μυστικό της επιτυχίας...   "
    assert sanitize_filename(title) == "Το μυστικό της επιτυχίας"


def test_sanitize_filename_empty_or_special():
    from shorts_engine.services.seo_generator import sanitize_filename

    assert sanitize_filename("") == ""
    assert sanitize_filename("???***///") == ""


def test_sanitize_filename_max_length():
    from shorts_engine.services.seo_generator import sanitize_filename

    long_title = "Αυτό είναι ένα πάρα πολύ μεγάλο κείμενο που ξεπερνάει κατά πολύ το όριο των χαρακτήρων"
    sanitized = sanitize_filename(long_title, max_length=30)
    assert len(sanitized) <= 30
    assert not sanitized.endswith(" ")
    assert not sanitized.endswith("-")


def test_seo_metadata_safe_filename():
    from shorts_engine.services.seo_generator import SeoMetadata

    seo = SeoMetadata(
        title="Shorts: Το μεγάλο κόλπο!",
        description="Περιγραφή",
        tags=("shorts",),
        primary_keyword="keyword",
        pinned_comment="pinned",
        curiosity_title="curiosity",
        authority_title="authority",
        contrarian_title="contrarian",
    )
    assert seo.safe_filename == "Shorts - Το μεγάλο κόλπο!"


def test_get_download_filename():
    from shorts_engine.services.seo_generator import get_download_filename

    # 1. With title
    assert (
        get_download_filename(
            title="Αλέξης Τσίπρας: Η νέα εποχή",
            fallback_filename="clip_01_short.mp4",
            extension="mp4",
        )
        == "Αλέξης Τσίπρας - Η νέα εποχή.mp4"
    )
    assert (
        get_download_filename(
            title="Αλέξης Τσίπρας: Η νέα εποχή",
            fallback_filename="seo_clip_01.json",
            extension="json",
        )
        == "Αλέξης Τσίπρας - Η νέα εποχή.json"
    )

    # 2. Without title (fallback)
    assert (
        get_download_filename(
            title=None,
            fallback_filename="clip_02_short.mp4",
            extension="mp4",
        )
        == "clip_02_short.mp4"
    )
    assert (
        get_download_filename(
            title="",
            fallback_filename="clip_02_short.mp4",
            extension="json",
        )
        == "clip_02_short.json"
    )


def test_strip_emojis():
    from shorts_engine.services.seo_generator import strip_emojis

    text = "🔥 Το Μέλλον της Τεχνητής Νοημοσύνης 🤖 στην Ελλάδα 🇬🇷! 🚀"
    cleaned = strip_emojis(text)
    assert cleaned == "Το Μέλλον της Τεχνητής Νοημοσύνης στην Ελλάδα!"
    assert "🔥" not in cleaned
    assert "🤖" not in cleaned
    assert "🇬🇷" not in cleaned
    assert "🚀" not in cleaned


def test_validate_seo_dict_removes_emojis():
    from shorts_engine.services.seo_generator import _validate_seo_dict

    data = {
        "title": "🎬 Η ανάλυση του προέδρου για την οικονομία 📈!",
        "description": "💥 Αναλυτική τοποθέτηση για τις εξελίξεις 🏛️ #Shorts",
        "tags": ["🔥 Τσίπρας", "οικονομία 💶", "Ελλάδα 🇬🇷", "shorts"],
    }
    seo = _validate_seo_dict(data)
    assert seo.title == "🎬 Η ανάλυση του προέδρου για την οικονομία 📈!"
    assert "🎬" in seo.title
    assert "📈" in seo.title
    assert "💥" in seo.description
    assert "🏛️" in seo.description
    assert "🔥" not in seo.tags[0]
    assert "💶" not in seo.tags[1]
    assert "🇬🇷" not in seo.tags[2]


def test_align_words_with_corrected_text_equal_count() -> None:
    from shorts_engine.services.seo_generator import align_words_with_corrected_text

    original_words = [(0.0, 0.4, "χαίρο"), (0.4, 0.8, "με")]
    corrected = "χαίρομαι πολύ"
    aligned = align_words_with_corrected_text(
        original_words=original_words,
        corrected_text=corrected,
        seg_start=0.0,
        seg_end=0.8,
    )
    assert len(aligned) == 2
    assert aligned[0][2] == "χαίρομαι"
    assert aligned[1][2] == "πολύ"


def test_align_words_with_corrected_text_none_words() -> None:
    from shorts_engine.services.seo_generator import align_words_with_corrected_text

    aligned = align_words_with_corrected_text(
        original_words=None,
        corrected_text="Ένα δύο τρία",
        seg_start=1.0,
        seg_end=4.0,
    )
    assert len(aligned) == 3
    assert aligned[0][2] == "Ένα"
    assert aligned[0][0] == 1.0
    assert aligned[2][2] == "τρία"
    assert aligned[2][1] == 4.0


def test_correct_transcript_greek_batching_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from shorts_engine.services.seo_generator import correct_transcript_greek
    from shorts_engine.services.transcriber import TranscriptionSegment

    # Create 70 segments to verify that batching processes beyond the old 60 limit
    segments = [
        TranscriptionSegment(start=float(i), end=float(i + 1), text=f"γραμμή {i}")
        for i in range(70)
    ]

    def fake_call_gemini(*args, **kwargs) -> str:
        contents = kwargs.get("contents") or (args[1] if len(args) > 1 else "")
        # Return properly tagged lines with an intentional correction
        # [k] διορθωμένη γραμμή k
        lines = []
        for line in contents.splitlines():
            line_str = line.strip()
            if line_str.startswith("[") and "]" in line_str:
                idx = line_str[1 : line_str.index("]")]
                lines.append(f"[{idx}] διορθωμένη γραμμή {idx}")
        return "\n".join(lines)

    monkeypatch.setattr(
        "shorts_engine.services.seo_generator._call_gemini_with_fallback",
        fake_call_gemini,
    )

    corrected = correct_transcript_greek(segments, api_key="fake-key")
    assert len(corrected) == 70
    # Both batch 1 (0..49) and batch 2 (50..69) should be corrected!
    assert "διορθωμένη" in corrected[0].text
    assert "διορθωμένη" in corrected[65].text


def test_correct_transcript_greek_acronym_sense_checking(monkeypatch: pytest.MonkeyPatch) -> None:
    from shorts_engine.services.seo_generator import correct_transcript_greek
    from shorts_engine.services.transcriber import TranscriptionSegment

    segments = [
        TranscriptionSegment(start=0.0, end=2.0, text="τα με με αποκρύπτουν την αλήθεια"),
        TranscriptionSegment(start=2.0, end=4.0, text="έλεγχος προστίμων από την α δε"),
        TranscriptionSegment(start=4.0, end=6.0, text="αυξήσεις στους λογαριασμούς της δε η"),
    ]

    def fake_call_gemini(*args, **kwargs) -> str:
        # Simulate Gemini accurately sense-checking and correcting the acronyms
        return (
            "[0] Τα ΜΜΕ αποκρύπτουν την αλήθεια\n"
            "[1] Έλεγχος προστίμων από την ΑΑΔΕ\n"
            "[2] Αυξήσεις στους λογαριασμούς της ΔΕΗ"
        )

    monkeypatch.setattr(
        "shorts_engine.services.seo_generator._call_gemini_with_fallback",
        fake_call_gemini,
    )

    corrected = correct_transcript_greek(segments, api_key="fake-key")
    assert "ΜΜΕ" in corrected[0].text
    assert "ΑΑΔΕ" in corrected[1].text
    assert "ΔΕΗ" in corrected[2].text


def test_clean_seo_title_preserves_complete_titles_under_100_chars() -> None:
    from shorts_engine.services.seo_generator import _clean_seo_title

    # 73-character title that used to be chopped at 60 chars ("στις πλάτ")
    title_73 = "ΒΟΜΒΑ για το ρεύμα: Κάποιοι παντελονιάζουν 7,5 ΔΙΣ στις πλάτες των καταναλωτών"
    cleaned = _clean_seo_title(title_73, max_chars=100)
    assert cleaned == title_73
    assert "πλάτες" in cleaned
    assert not cleaned.endswith("πλάτ")


def test_clean_seo_title_truncates_at_word_boundary_when_exceeding_100_chars() -> None:
    from shorts_engine.services.seo_generator import _clean_seo_title

    long_title = "Αποκάλυψη-σοκ για τα τιμολόγια ρεύματος στην Ελλάδα: Πώς οι μεγάλοι όμιλοι αποκομίζουν δισεκατομμύρια ευρώ κέρδη εις βάρος των πολιτών"
    cleaned = _clean_seo_title(long_title, max_chars=100)
    assert len(cleaned) <= 100
    # Must end on a complete word, not mid-word
    assert not cleaned.endswith((" ", "-", "—", ","))
    last_word = cleaned.split()[-1]
    assert last_word in long_title


def test_align_words_with_corrected_text_clamps_to_segment_bounds() -> None:
    from shorts_engine.services.seo_generator import align_words_with_corrected_text

    original_words = [
        (0.0, 1.0, "γεια"),
        (1.0, 2.0, "σας"),
    ]
    # Corrected text has more words inserted
    corrected_text = "γεια σας κύριες και κύριοι"
    aligned = align_words_with_corrected_text(
        original_words=original_words,
        corrected_text=corrected_text,
        seg_start=0.0,
        seg_end=2.0,
    )
    assert len(aligned) == 5
    assert aligned[0][0] >= 0.0
    assert aligned[-1][1] <= 2.05
    # Strictly monotonic
    for i in range(1, len(aligned)):
        assert aligned[i][0] >= aligned[i - 1][0]




