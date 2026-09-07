"""
tests/test_seo_generator.py — Unit tests for SEO generation and transcript correction helpers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

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
