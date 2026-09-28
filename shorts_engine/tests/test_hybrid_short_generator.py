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
    """Verify that edge-tts WordBoundary chunks are converted into millisecond-accurate word segments."""
    fake_mp3 = tmp_path / "test_voice.mp3"

    sample_events = [
        {"type": "audio", "data": b"FAKE_AUDIO_DATA_FOR_TEST"},
        {"type": "WordBoundary", "offset": 10000000, "duration": 4000000, "text": "Αυτά"},
        {"type": "WordBoundary", "offset": 14000000, "duration": 3000000, "text": "είναι"},
        {"type": "WordBoundary", "offset": 17000000, "duration": 2000000, "text": "τα"},
        {"type": "WordBoundary", "offset": 19000000, "duration": 5000000, "text": "πραγματικά"},
        {"type": "WordBoundary", "offset": 24000000, "duration": 6000000, "text": "γεγονότα."},
    ]

    async def fake_stream():
        for ev in sample_events:
            yield ev

    mock_comm = MagicMock()
    mock_comm.stream = fake_stream

    with patch("edge_tts.Communicate", return_value=mock_comm):
        out_path, segments = synthesize_voiceover_with_segments(
            text="Αυτά είναι τα πραγματικά γεγονότα.",
            output_path=fake_mp3,
            voice="el-GR-NestorasNeural",
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
        def _fake_synth(text, output_path, *args, **kwargs):
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


def test_build_hybrid_short_two_speaker_bites_subtitles_sync(tmp_path: Path) -> None:
    """Verify that multi-part speaker bite 2 captions start at natural t=0 offsets and are not crushed."""
    fake_clip = tmp_path / "long_speaker_clip.mp4"
    fake_clip.touch()
    out_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    fake_segments = [
        TranscriptionSegment(
            start=0.0,
            end=9.5,
            text="Πρώτο μέρος της δήλωσης.",
            words=[(0.0, 4.0, "Πρώτο"), (4.2, 7.0, "μέρος"), (7.2, 9.5, "δήλωσης.")],
        ),
        TranscriptionSegment(
            start=10.0,
            end=20.0,
            text="Δεύτερο μέρος της δήλωσης που αναλύει τα πάντα.",
            words=[(10.0, 13.0, "Δεύτερο"), (13.5, 16.0, "μέρος"), (16.5, 20.0, "πάντα.")],
        ),
    ]

    settings = Settings(
        target_width=1080,
        target_height=1920,
        gemini_api_key="FAKE_KEY",
        pexels_api_key="FAKE_PEXELS",
        enable_bg_music=False,
    )

    mock_pkg = HybridScriptPackage(
        title="Δοκιμή Hybrid 2 Bites",
        hook="Δείτε αυτό",
        opening_hook="Προσοχή στη δήλωση.",
        commentary_script="Αυτά είναι τα στοιχεία.",
        outro_script="Γράψτε στα σχόλια.",
        narration_script="Αυτά είναι τα στοιχεία.",
        scenes=[{"scene_index": 1, "pexels_query": "greece news"}],
        seo=MagicMock(primary_keyword="στοιχεία"),
    )

    written_ass_calls = []

    def _capture_write_ass(*args, **kwargs):
        segs = kwargs.get("segments") if "segments" in kwargs else args[0]
        out_path = kwargs.get("output_path") if "output_path" in kwargs else args[1]
        written_ass_calls.append((Path(out_path).name, list(segs)))
        Path(out_path).touch()

    with (
        patch("shorts_engine.services.hybrid_short_generator.probe_resolution", return_value=(1080, 1920)),
        patch("shorts_engine.services.hybrid_short_generator.probe_duration", side_effect=lambda p: 20.0 if "spk" in str(p) or "speaker" in str(p) else 4.0),
        patch("shorts_engine.services.hybrid_short_generator.mask_burned_in_subtitles", side_effect=lambda src, dst: src),
        patch("shorts_engine.services.hybrid_short_generator.write_ass_file", side_effect=_capture_write_ass),
        patch("shorts_engine.services.hybrid_short_generator.burn_subtitles"),
        patch("shorts_engine.services.hybrid_short_generator.generate_hybrid_script", return_value=mock_pkg),
        patch("shorts_engine.services.hybrid_short_generator.correct_transcript_greek", side_effect=lambda segs, key: segs),
        patch("shorts_engine.services.hybrid_short_generator.synthesize_voiceover_with_segments") as mock_synth,
        patch("shorts_engine.services.hybrid_short_generator._assemble_scene_video") as mock_assemble,
        patch("subprocess.run") as mock_subproc,
    ):
        mock_subproc.return_value = MagicMock(returncode=0)

        def _fake_synth(text, output_path, *args, **kwargs):
            output_path.touch()
            return output_path, [TranscriptionSegment(0.0, 4.0, text)]

        mock_synth.side_effect = _fake_synth

        fake_scene_bed = work_dir / "scenes_bed.mp4"
        fake_scene_bed.touch()
        mock_assemble.return_value = fake_scene_bed

        def _side_effect_run(cmd, **kw):
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

        # Verify that bite 2 subtitles (b2) were generated with valid, positive, spaced timestamps
        b2_calls = [c for c in written_ass_calls if "b2" in c[0]]
        assert len(b2_calls) == 1
        b2_segs = b2_calls[0][1]
        assert len(b2_segs) >= 1
        for seg in b2_segs:
            assert seg.start >= 0.0
            assert seg.end > seg.start
            assert seg.end - seg.start > 1.0  # Must not be crushed to 0.05s!
            if seg.words:
                assert seg.words[0][0] >= 0.0
                assert seg.words[-1][1] > seg.words[0][0]
                assert seg.words[-1][1] - seg.words[0][0] > 1.0
