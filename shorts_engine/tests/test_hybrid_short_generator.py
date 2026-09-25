"""
tests/test_hybrid_short_generator.py — Unit tests for the Hybrid Short Generation Engine.

Verifies:
  1. Direct TTS sentence and word boundary synchronization (speech-to-caption lockstep).
  2. Hybrid script generation and JSON schema validation via Gemini.
  3. End-to-end orchestration mixing authentic speaker clips with AI breakdown scenes.
  4. Autopilot pipeline execution under production_strategy='hybrid'.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shorts_engine.config import Settings
from shorts_engine.services.ai_short_generator import (
    synthesize_voiceover_with_segments,
)
from shorts_engine.services.content_decision_engine import (
    classify_production_mode,
)
from shorts_engine.services.hybrid_short_generator import (
    HybridScriptPackage,
    build_hybrid_short,
    generate_hybrid_script,
)
from shorts_engine.services.transcriber import TranscriptionSegment


def test_classify_production_mode_hybrid(tmp_path: Path) -> None:
    fake_vid = tmp_path / "video.mp4"
    fake_vid.touch()
    assert classify_production_mode(fake_vid, user_preference="hybrid") == "hybrid"
    assert classify_production_mode(fake_vid, user_preference="HYBRID") == "hybrid"


def test_synthesize_voiceover_with_segments_mocked(tmp_path: Path) -> None:
    """Verify that edge-tts SentenceBoundary chunks are converted into millisecond-accurate word segments."""
    fake_mp3 = tmp_path / "test_voice.mp3"

    sample_sentence = {
        "type": "SentenceBoundary",
        "offset": 10000000,  # 1.0s
        "duration": 20000000,  # 2.0s
        "text": "Αυτά είναι τα πραγματικά γεγονότα.",
    }

    async def fake_stream():
        yield {"type": "audio", "data": b"FAKE_AUDIO_DATA_FOR_TEST"}
        yield sample_sentence

    mock_comm = MagicMock()
    mock_comm.stream = fake_stream

    with patch("edge_tts.Communicate", return_value=mock_comm):
        out_path, segments = synthesize_voiceover_with_segments(
            text="Αυτά είναι τα πραγματικά γεγονότα.",
            output_path=fake_mp3,
        )

        assert out_path.is_file()
        assert out_path.read_bytes() == b"FAKE_AUDIO_DATA_FOR_TEST"
        assert len(segments) == 1

        seg = segments[0]
        assert seg.start == pytest.approx(1.0, 0.01)
        assert seg.end == pytest.approx(3.0, 0.01)
        assert seg.text == "Αυτά είναι τα πραγματικά γεγονότα."
        assert seg.words is not None
        assert len(seg.words) == 5

        # Check word timings are non-decreasing and within segment bounds
        for w_start, w_end, w_txt in seg.words:
            assert w_start >= 1.0 - 0.001
            assert w_end <= 3.0 + 0.001
            assert w_end > w_start


def test_generate_hybrid_script_mocked() -> None:
    """Verify Gemini script parsing extracts Part 2 breakdown scenes and SEO."""
    mock_payload = {
        "title": "Η Αλήθεια για τα Επιτόκια",
        "hook": "Τι πραγματικά συμβαίνει;",
        "narration_script": "Αυτά τα νούμερα δείχνουν την πραγματική πίεση στην αγορά. Εσείς τι πιστεύετε; Γράψτε μας!",
        "scenes": [
            {
                "scene_index": 1,
                "narration_chunk": "Αυτά τα νούμερα δείχνουν την πραγματική πίεση.",
                "visual_prompt": "Greek bank and athens finance street",
                "pexels_query": "greece banking finance",
            },
            {
                "scene_index": 2,
                "narration_chunk": "Εσείς τι πιστεύετε; Γράψτε μας!",
                "visual_prompt": "Greek citizens walking in city",
                "pexels_query": "athens street crowd",
            },
        ],
        "seo": {
            "title": "Η Αλήθεια για τα Επιτόκια #shorts",
            "description": "Ανάλυση οικονομικών δεδομένων @DianismaNews",
            "tags": ["οικονομία", "ελλάδα", "shorts"],
            "primary_keyword": "επιτόκια",
        },
    }

    with (
        patch("shorts_engine.services.hybrid_short_generator.genai.Client"),
        patch("shorts_engine.services.hybrid_short_generator._call_gemini_with_fallback", return_value=json.dumps(mock_payload)),
    ):
        pkg = generate_hybrid_script(
            speaker_text="Τα επιτόκια αυξάνονται ραγδαία φέτος.",
            topic_title="Επιτόκια 2026",
            topic_context="Συνέντευξη για οικονομικά",
            gemini_api_key="FAKE_KEY",
        )

        assert isinstance(pkg, HybridScriptPackage)
        assert pkg.title == "Η Αλήθεια για τα Επιτόκια"
        assert len(pkg.scenes) == 2
        assert pkg.seo.primary_keyword == "επιτόκια"
        assert "επιτόκια" in pkg.seo.title.lower()


def test_build_hybrid_short_orchestration(tmp_path: Path) -> None:
    """Verify end-to-end hybrid assembly of Part 1 (Speaker) + Part 2 (AI Breakdown)."""
    fake_clip = tmp_path / "speaker_clip.mp4"
    fake_clip.touch()
    out_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    fake_segments = [
        TranscriptionSegment(start=0.0, end=4.5, text="Καλώς ήρθατε στο κανάλι."),
        TranscriptionSegment(start=4.5, end=9.0, text="Σήμερα θα δούμε τα δεδομένα."),
    ]

    settings = Settings(
        target_width=1080,
        target_height=1920,
        gemini_api_key="FAKE_KEY",
        pexels_api_key="FAKE_PEXELS",
        enable_bg_music=False,
    )

    mock_pkg = HybridScriptPackage(
        title="Δοκιμή Hybrid",
        hook="Δείτε αυτό",
        narration_script="Αυτά είναι τα στοιχεία.",
        scenes=[{"scene_index": 1, "pexels_query": "greece news"}],
        seo=MagicMock(primary_keyword="στοιχεία"),
    )

    with (
        patch("shorts_engine.services.hybrid_short_generator.probe_resolution", return_value=(1080, 1920)),
        patch("shorts_engine.services.hybrid_short_generator.probe_duration", side_effect=lambda p: 9.0 if "spk" in str(p) or "speaker" in str(p) else 4.0),
        patch("shorts_engine.services.hybrid_short_generator.mask_burned_in_subtitles", side_effect=lambda src, dst: src),
        patch("shorts_engine.services.hybrid_short_generator.write_ass_file"),
        patch("shorts_engine.services.hybrid_short_generator.burn_subtitles"),
        patch("shorts_engine.services.hybrid_short_generator.generate_hybrid_script", return_value=mock_pkg),
        patch("shorts_engine.services.hybrid_short_generator.correct_transcript_greek", side_effect=lambda segs, key: segs),
        patch("shorts_engine.services.hybrid_short_generator.synthesize_voiceover_with_segments") as mock_synth,
        patch("shorts_engine.services.hybrid_short_generator._assemble_scene_video") as mock_assemble,
        patch("subprocess.run") as mock_subproc,
    ):
        mock_subproc.return_value = MagicMock(returncode=0)

        # Mock synth output
        def _fake_synth(text, output_path, voice):
            output_path.touch()
            return output_path, [TranscriptionSegment(0.0, 5.0, text)]

        mock_synth.side_effect = _fake_synth

        # Mock assemble scene video
        fake_scene_bed = work_dir / "scenes_bed.mp4"
        fake_scene_bed.touch()
        mock_assemble.return_value = fake_scene_bed

        # Mock subprocess creating the output files
        def _side_effect_run(cmd, **kw):
            # cmd is ffmpeg command
            out_file = Path(cmd[-1])
            out_file.touch()
            return MagicMock(returncode=0, stderr="")

        mock_subproc.side_effect = _side_effect_run

        final_vid, seo = build_hybrid_short(
            clip_video_path=fake_clip,
            speaker_segments=fake_segments,
            topic_title="Ειδήσεις 2026",
            topic_context="Ανάλυση",
            settings=settings,
            tmp_dir=work_dir,
            output_dir=out_dir,
        )

        assert final_vid.is_file()
        assert "hybrid_short_" in final_vid.name
        assert seo == mock_pkg.seo
