"""
tests/test_retention_and_pacing.py — Test suite for Retention Auditing, Pacing Controller,
Wayin 4-Axis Virality Scoring, and 3-Act Explainer Blueprint.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shorts_engine.services.clip_selector import (
    ClipCandidate,
    _parse_single_clip,
    select_clips,
)
from shorts_engine.services.logic_guardrail import (
    RetentionAudit,
    audit_retention_signals,
)
from shorts_engine.services.seo_generator import SeoMetadata
from shorts_engine.services.transcriber import TranscriptionSegment
from shorts_engine.services.video_engine import apply_pacing_pattern_interrupts


def test_retention_audit_high_retention_signals() -> None:
    """A clip starting immediately with rapid speech, numbers, and curiosity question gets high retention."""
    text = (
        "Γιατί οι τιμές στο ρεύμα αυξήθηκαν κατά 25% αυτόν τον μήνα; "
        "Η αποκάλυψη των στοιχείων δείχνει ότι οι λογαριασμοί κρύβουν τεράστια κέρδη "
        "και οι καταναλωτές πληρώνουν δισεκατομμύρια ευρώ."
    )
    segments = [
        TranscriptionSegment(
            start=0.1,
            end=6.0,
            text=text,
            words=[(0.1, 0.4, "Γιατί"), (0.4, 0.8, "οι"), (0.8, 1.2, "τιμές")],
        )
    ]
    audit = audit_retention_signals(
        segments=segments,
        text=text,
        duration=7.0,
        hook_summary="Shocking power bills increase",
        title="Γιατί αυξήθηκε 25% το ρεύμα;",
    )

    assert isinstance(audit, RetentionAudit)
    assert audit.score >= 80
    assert audit.grade in ("S", "A")
    assert audit.hook_speed_score == 100
    assert audit.curiosity_gap_score >= 80
    assert audit.intro_silence_seconds <= 0.35


def test_retention_audit_sluggish_signals_recommends_action() -> None:
    """A clip with dead air at the start and sluggish monologue receives penalties and advice."""
    text = "Ναι λοιπόν αυτό είναι ένα θέμα."
    segments = [
        TranscriptionSegment(
            start=1.8,
            end=10.0,
            text=text,
            words=[(1.8, 2.5, "Ναι"), (2.5, 3.2, "λοιπόν")],
        )
    ]
    audit = audit_retention_signals(
        segments=segments,
        text=text,
        duration=10.0,
    )

    assert audit.score < 75
    assert audit.intro_silence_seconds >= 1.5
    assert audit.hook_speed_score <= 60
    assert any("νεκρού" in r.lower() or "κενό" in r.lower() or "σιωπή" in r.lower() or "αργός" in r.lower() for r in audit.actionable_recommendations)


def test_wayin_4_axis_virality_score_calculation() -> None:
    """Verify Wayin 4-axis weighted calculation (emotion 30%, narrative 30%, hook 25%, velocity 15%)."""
    seo = SeoMetadata.fallback("Test Title")
    candidate = ClipCandidate(
        index=1,
        start_time=0.0,
        end_time=35.0,
        hook_summary="Tense parliamentary clash",
        seo=seo,
        broll_query="parliament debate",
        emotional_intensity=9,
        standalone_narrative=9,
        hook_potency=10,
        speech_velocity=8,
    )

    # 9*0.30 (2.7) + 9*0.30 (2.7) + 10*0.25 (2.5) + 8*0.15 (1.2) = 9.1
    assert candidate.virality_score == 9.1
    assert candidate.emotional_intensity == 9
    assert candidate.standalone_narrative == 9


def test_parse_single_clip_filters_weak_narrative() -> None:
    """Clips with standalone_narrative < 5 must be discarded to prevent out-of-context cuts."""
    raw_clip = {
        "start_time": 10.0,
        "end_time": 45.0,
        "hook_summary": "Fragment without context",
        "emotional_intensity": 8,
        "standalone_narrative": 3,  # < 5 -> Must be rejected!
        "hook_potency": 8,
        "speech_velocity": 7,
        "seo": {
            "title": "Αποσπασματικό Βίντεο",
            "curiosity_title": "Τι συνέβη;",
            "authority_title": "Η αλήθεια",
            "contrarian_title": "Το λάθος",
            "primary_keyword": "βουλή",
            "description": "Περιγραφή",
            "pinned_comment": "Σχόλιο",
            "tags": ["πολιτική"],
        },
        "broll_query": "greek debate",
    }

    parsed = _parse_single_clip(raw_clip, index=1, min_dur=30.0, max_dur=60.0)
    assert parsed is None


def test_select_clips_ranks_by_virality_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """Candidates returned from select_clips must be ranked highest virality_score first."""
    segments = [
        TranscriptionSegment(start=0.0, end=50.0, text="Πρώτο μέρος ομιλίας.", words=[]),
        TranscriptionSegment(start=50.0, end=100.0, text="Δεύτερο μέρος ομιλίας.", words=[]),
    ]

    mock_json = """{
      "clips": [
        {
          "start_time": 0.0,
          "end_time": 40.0,
          "hook_summary": "Medium potency clip",
          "emotional_intensity": 6,
          "standalone_narrative": 7,
          "hook_potency": 6,
          "speech_velocity": 6,
          "seo": {
            "title": "Μεσαίας δυναμικής κλιπ",
            "curiosity_title": "Τι έγινε;",
            "authority_title": "Στοιχεία",
            "contrarian_title": "Ανατροπή",
            "primary_keyword": "οικονομία",
            "description": "Περιγραφή",
            "pinned_comment": "Σχόλιο",
            "tags": ["νέα"]
          },
          "broll_query": "market street"
        },
        {
          "start_time": 50.0,
          "end_time": 90.0,
          "hook_summary": "Viral explosive clip",
          "emotional_intensity": 10,
          "standalone_narrative": 10,
          "hook_potency": 10,
          "speech_velocity": 9,
          "seo": {
            "title": "Εκρηκτική αποκάλυψη",
            "curiosity_title": "Το μεγάλο σκάνδαλο",
            "authority_title": "Αποδείξεις",
            "contrarian_title": "Όλο το παρασκήνιο",
            "primary_keyword": "σκάνδαλο",
            "description": "Περιγραφή",
            "pinned_comment": "Σχόλιο",
            "tags": ["σκάνδαλο"]
          },
          "broll_query": "court justice"
        }
      ]
    }"""

    monkeypatch.setattr(
        "shorts_engine.services.clip_selector._call_gemini_with_fallback",
        lambda **kwargs: mock_json,
    )
    try:
        monkeypatch.setattr(
            "shorts_engine.services.cache_manager.load_cache_pickle",
            lambda *args: None,
        )
        monkeypatch.setattr(
            "shorts_engine.services.cache_manager.save_cache_pickle",
            lambda *args: None,
        )
    except (AttributeError, ModuleNotFoundError):
        pass
    try:
        monkeypatch.setattr(
            "services.cache_manager.load_cache_pickle",
            lambda *args: None,
        )
        monkeypatch.setattr(
            "services.cache_manager.save_cache_pickle",
            lambda *args: None,
        )
    except (AttributeError, ModuleNotFoundError):
        pass

    clips = select_clips(
        segments=segments,
        gemini_api_key="mock_key",
        max_clips=2,
        min_clips=2,
        min_dur=30.0,
        max_dur=50.0,
    )

    assert len(clips) >= 2
    # The clip with virality_score 9.8 (clip 2) must be ranked as index 1!
    assert clips[0].virality_score > clips[1].virality_score
    assert clips[0].index == 1
    assert "Εκρηκτική" in clips[0].seo.title


def test_apply_pacing_pattern_interrupts_short_video_passthrough() -> None:
    """Videos shorter than cut_interval (<= 3.5s) pass through without alteration."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        in_path = Path(tmp_dir) / "short.mp4"
        in_path.write_bytes(b"dummy video bytes")
        out_path = Path(tmp_dir) / "out.mp4"

        with patch("shorts_engine.services.video_engine.probe_duration", return_value=2.8):
            res = apply_pacing_pattern_interrupts(in_path, out_path, cut_interval=3.5)
            assert res == out_path
            assert out_path.read_bytes() == b"dummy video bytes"


def test_apply_pacing_pattern_interrupts_builds_ffmpeg_filter() -> None:
    """Videos > 3.5s build alternating normal and punch-zoom segments."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        in_path = Path(tmp_dir) / "long.mp4"
        in_path.write_bytes(b"dummy")
        out_path = Path(tmp_dir) / "out.mp4"

        mock_run = MagicMock()
        with (
            patch("shorts_engine.services.video_engine.probe_duration", return_value=10.0),
            patch("shorts_engine.services.video_engine.probe_has_audio", return_value=True),
            patch("shorts_engine.services.video_engine.run_ffmpeg", mock_run),
        ):
            apply_pacing_pattern_interrupts(in_path, out_path, cut_interval=3.5, zoom_factor=1.12)

            assert mock_run.called
            cmd = mock_run.call_args[0][0]
            cmd_str = " ".join(cmd)
            assert "-filter_complex" in cmd
            assert "crop=1080:1920" in cmd_str
            assert "concat=n=" in cmd_str
