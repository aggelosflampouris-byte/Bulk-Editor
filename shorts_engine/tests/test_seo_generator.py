"""
tests/test_seo_generator.py — Unit tests for SEO generation and transcript correction helpers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

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


def test_call_gemini_with_fallback_sets_thinking_budget_zero():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Success"
    mock_client.models.generate_content.return_value = mock_resp

    from google.genai import types as genai_types

    config = genai_types.GenerateContentConfig(max_output_tokens=1000)
    assert config.thinking_config is None

    _call_gemini_with_fallback(mock_client, "test", config)
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget == 0


def test_seo_metadata_youtube_tags_display():
    from shorts_engine.services.seo_generator import SeoMetadata

    seo = SeoMetadata(
        title="Τίτλος",
        description="Περιγραφή",
        tags=("Τσίπρας", "πολιτική", "Ελλάδα", "shorts"),
        primary_keyword="keyword",
        pinned_comment="pinned",
        alt_titles=("alt",),
        tiktok_caption="tiktok",
        ig_reels_caption="reels"
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
        alt_titles=("alt",),
        tiktok_caption="tiktok",
        ig_reels_caption="reels"
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


