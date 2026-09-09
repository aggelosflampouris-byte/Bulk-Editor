"""
tests/test_cloud_analyzer.py — Unit tests for 100% Cloud-based SEO, chapters, and clip highlights analyzer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from shorts_engine.services.cloud_analyzer import (
    analyze_transcript_cloud,
    CloudAnalysisResult,
    VideoChapter,
    HighRetentionHook,
    CloudAnalyzerError,
    transcribe_audio_cloud,
)
from shorts_engine.services.transcriber import TranscriptionSegment


def test_analyze_transcript_cloud_success():
    mock_json_response = """{
        "target_seo_tags": ["τεχνητή νοημοσύνη", "shorts", "viral greece", "greek tech", "cloud ai"],
        "video_chapters": [
            {
                "title": "Εισαγωγή στην AI",
                "start_time": 0.0,
                "end_time": 60.0,
                "summary": "Γενική εισαγωγή στα cloud μοντέλα."
            },
            {
                "title": "Πρακτική Εφαρμογή",
                "start_time": 60.0,
                "end_time": 180.0,
                "summary": "Πώς χρησιμοποιούμε τα cloud μοντέλα στην πράξη."
            }
        ],
        "high_retention_hooks": [
            {
                "title": "Το Μυστικό της Cloud AI",
                "start_time": 15.0,
                "end_time": 55.0,
                "hook_summary": "Αποκάλυψη για cloud μοντέλα.",
                "virality_reason": "Ισχυρό hook με άμεση ανατροπή στην αρχή."
            },
            {
                "title": "Δεύτερο Hook",
                "start_time": 65.0,
                "end_time": 105.0,
                "hook_summary": "Δεύτερη δυνατή στιγμή.",
                "virality_reason": "Υψηλό ενδιαφέρον."
            },
            {
                "title": "Τρίτο Hook",
                "start_time": 120.0,
                "end_time": 160.0,
                "hook_summary": "Τρίτη δυνατή στιγμή.",
                "virality_reason": "Κορυφαίο retention."
            }
        ]
    }"""

    with patch("shorts_engine.services.cloud_analyzer._call_gemini_with_fallback", return_value=mock_json_response):
        result = analyze_transcript_cloud(
            transcript_data="[00:00 - 01:00] Εισαγωγή στην AI. [01:00 - 03:00] Πρακτική Εφαρμογή.",
            gemini_api_key="valid_fake_key",
        )

        assert isinstance(result, CloudAnalysisResult)
        assert len(result.target_seo_tags) == 5
        assert "τεχνητή νοημοσύνη" in result.target_seo_tags
        assert len(result.video_chapters) == 2
        assert result.video_chapters[0].title == "Εισαγωγή στην AI"
        assert result.video_chapters[0].start_time == 0.0
        assert result.video_chapters[0].end_time == 60.0
        assert len(result.high_retention_hooks) == 3
        assert result.high_retention_hooks[0].title == "Το Μυστικό της Cloud AI"
        assert result.high_retention_hooks[0].duration == 40.0


def test_analyze_transcript_cloud_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(CloudAnalyzerError) as exc_info:
        analyze_transcript_cloud(
            transcript_data="dummy text",
            gemini_api_key="",
        )
    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_transcribe_audio_cloud_mock(tmp_path):
    dummy_audio = tmp_path / "test.mp3"
    dummy_audio.write_bytes(b"dummy audio content")

    mock_transcript_json = """[
        {"start": 0.0, "end": 4.5, "text": "Γεια σας και καλώς ήρθατε."},
        {"start": 4.5, "end": 9.0, "text": "Σήμερα μιλάμε για cloud AI."}
    ]"""

    with patch("shorts_engine.services.cloud_analyzer._call_gemini_with_fallback", return_value=mock_transcript_json):
        segments = transcribe_audio_cloud(dummy_audio, gemini_api_key="fake_key")
        assert len(segments) == 2
        assert segments[0].text == "Γεια σας και καλώς ήρθατε."
        assert segments[0].start == 0.0
        assert segments[0].end == 4.5
