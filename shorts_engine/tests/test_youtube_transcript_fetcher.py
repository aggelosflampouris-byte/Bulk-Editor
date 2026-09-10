"""
tests/test_youtube_transcript_fetcher.py — Unit tests for youtube-transcript-api integration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shorts_engine.services.transcriber import TranscriptionSegment
from shorts_engine.services.youtube_transcript_fetcher import (
    YouTubeTranscriptUnavailableError,
    extract_youtube_id,
    fetch_youtube_transcript,
    format_transcript_for_llm,
)


def test_extract_youtube_id():
    assert extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s") == "dQw4w9WgXcQ"
    assert extract_youtube_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_id("https://example.com/not-youtube") is None
    assert extract_youtube_id("") is None


def test_fetch_youtube_transcript_success():
    raw_snippets = [
        {"text": "Γεια σας σε όλους", "start": 0.0, "duration": 3.5},
        {"text": "Σήμερα θα μιλήσουμε για AI", "start": 3.5, "duration": 4.0},
    ]

    mock_snippet_1 = MagicMock(text="Γεια σας σε όλους", start=0.0, duration=3.5)
    mock_snippet_2 = MagicMock(text="Σήμερα θα μιλήσουμε για AI", start=3.5, duration=4.0)
    mock_fetched = [mock_snippet_1, mock_snippet_2]

    mock_api = MagicMock()
    mock_api.fetch.return_value = mock_fetched

    with patch("shorts_engine.services.youtube_transcript_fetcher.YouTubeTranscriptApi", return_value=mock_api):
        segments = fetch_youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ", languages=("el", "en"))
        assert len(segments) == 2
        assert segments[0].text == "Γεια σας σε όλους"
        assert segments[0].start == 0.0
        assert segments[0].end == 3.5
        assert segments[1].text == "Σήμερα θα μιλήσουμε για AI"
        assert segments[1].start == 3.5
        assert segments[1].end == 7.5


def test_fetch_youtube_transcript_unavailable():
    from youtube_transcript_api import TranscriptsDisabled

    mock_api = MagicMock()
    mock_api.fetch.side_effect = TranscriptsDisabled("dQw4w9WgXcQ")

    with patch("shorts_engine.services.youtube_transcript_fetcher.YouTubeTranscriptApi", return_value=mock_api):
        with pytest.raises(YouTubeTranscriptUnavailableError):
            fetch_youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_format_transcript_for_llm():
    segments = [
        TranscriptionSegment(start=10.0, end=45.0, text="Πρώτο κομμάτι ομιλίας."),
        TranscriptionSegment(start=45.0, end=90.0, text="Δεύτερο κομμάτι ομιλίας."),
    ]
    formatted = format_transcript_for_llm(segments)
    assert "[00:10 - 00:45]" in formatted
    assert "Πρώτο κομμάτι ομιλίας." in formatted
    assert "[00:45 - 01:30]" in formatted
    assert "Δεύτερο κομμάτι ομιλίας." in formatted
