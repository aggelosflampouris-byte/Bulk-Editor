"""
tests/test_ollama_analyzer.py — Unit tests for Ollama SEO, chapters, and clip highlights analyzer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from shorts_engine.services.ollama_analyzer import (
    analyze_transcript_with_ollama,
    OllamaAnalysisResult,
    VideoChapter,
    HighRetentionHook,
    OllamaServiceError,
)
from shorts_engine.services.transcriber import TranscriptionSegment


def test_analyze_transcript_with_ollama_success():
    mock_json_response = """{
        "target_seo_tags": ["τεχνητή νοημοσύνη", "shorts", "viral greece", "greek tech", "qwen"],
        "video_chapters": [
            {
                "title": "Εισαγωγή στην AI",
                "start_time": 0.0,
                "end_time": 60.0,
                "summary": "Γενική εισαγωγή στα μοντέλα τεχνητής νοημοσύνης."
            },
            {
                "title": "Πρακτική Εφαρμογή",
                "start_time": 60.0,
                "end_time": 180.0,
                "summary": "Πώς χρησιμοποιούμε τα μοντέλα στην πράξη."
            }
        ],
        "high_retention_hooks": [
            {
                "title": "Το Μυστικό της AI",
                "start_time": 15.0,
                "end_time": 55.0,
                "hook_summary": "Αποκάλυψη για το πώς αλλάζει ο κόσμος.",
                "virality_reason": "Ισχυρό hook με άμεση ανατροπή στην αρχή."
            }
        ]
    }"""

    mock_chat_response = MagicMock()
    mock_chat_response.message.content = mock_json_response

    mock_client = MagicMock()
    mock_client.chat.return_value = mock_chat_response

    with patch("shorts_engine.services.ollama_analyzer.ollama.Client", return_value=mock_client):
        result = analyze_transcript_with_ollama(
            transcript_data="[00:00 - 01:00] Εισαγωγή στην AI. [01:00 - 03:00] Πρακτική Εφαρμογή.",
            model="qwen2.5:32b",
            host="http://localhost:11434",
        )

        assert isinstance(result, OllamaAnalysisResult)
        assert len(result.target_seo_tags) == 5
        assert "τεχνητή νοημοσύνη" in result.target_seo_tags
        assert len(result.video_chapters) == 2
        assert result.video_chapters[0].title == "Εισαγωγή στην AI"
        assert result.video_chapters[0].start_time == 0.0
        assert result.video_chapters[0].end_time == 60.0
        assert len(result.high_retention_hooks) == 1
        assert result.high_retention_hooks[0].title == "Το Μυστικό της AI"
        assert result.high_retention_hooks[0].duration == 40.0


def test_analyze_transcript_with_ollama_parses_markdown_wrapped_json():
    mock_markdown_json = """```json
    {
        "target_seo_tags": "greece, tech, ai",
        "video_chapters": [
            {"title": "Intro", "start_time": "00:00", "end_time": "01:00", "summary": "Intro"}
        ],
        "high_retention_hooks": [
            {"title": "Hook 1", "start_time": 10.0, "end_time": 45.0, "hook_summary": "Great hook", "virality_reason": "High retention"}
        ]
    }
    ```"""

    mock_chat_response = MagicMock()
    mock_chat_response.message.content = mock_markdown_json

    mock_client = MagicMock()
    mock_client.chat.return_value = mock_chat_response

    with patch("shorts_engine.services.ollama_analyzer.ollama.Client", return_value=mock_client):
        result = analyze_transcript_with_ollama(
            transcript_data=[TranscriptionSegment(0.0, 60.0, "Intro text")],
            model="mistral",
        )
        assert len(result.target_seo_tags) == 3
        assert result.target_seo_tags == ["greece", "tech", "ai"]
        assert len(result.video_chapters) == 1
        assert result.video_chapters[0].start_time == 0.0
        assert result.video_chapters[0].end_time == 60.0


def test_analyze_transcript_with_ollama_connection_error():
    mock_client = MagicMock()
    mock_client.chat.side_effect = ConnectionError("Could not connect to Ollama daemon at http://localhost:11434")

    with patch("shorts_engine.services.ollama_analyzer.ollama.Client", return_value=mock_client):
        with pytest.raises(OllamaServiceError) as exc_info:
            analyze_transcript_with_ollama(
                transcript_data="dummy transcript",
                model="qwen2.5",
            )
        assert "Ollama" in str(exc_info.value)
