"""
tests/test_content_decision_and_masking.py — Unit tests for:
- subtitle_masker (blur + dark plate ffmpeg filter generation)
- content_decision_engine (speaker vs full AI classification)
- ai_short_generator (data models and scene parsing)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shorts_engine.services.ai_short_generator import (
    AIShortPackage,
    clean_json_markdown,
)
from shorts_engine.services.content_decision_engine import (
    classify_production_mode,
)
from shorts_engine.services.face_tracker import SpeakerTrackingResult
from shorts_engine.services.subtitle_masker import (
    build_subtitle_mask_filter,
    mask_burned_in_subtitles,
)

# --- Subtitle Masker Tests ---


def test_build_subtitle_mask_filter_custom_params() -> None:
    filt = build_subtitle_mask_filter(mask_y=1100, mask_h=500, opacity=0.75)
    assert "1100" in filt
    assert "500" in filt
    assert "boxblur=24:12" in filt
    assert "black@0.75" in filt


def test_mask_burned_in_subtitles_nonexistent_input(tmp_path: Path) -> None:
    non_existent = tmp_path / "ghost.mp4"
    out = tmp_path / "out.mp4"
    with pytest.raises(FileNotFoundError):
        mask_burned_in_subtitles(non_existent, out)


def test_mask_burned_in_subtitles_ffmpeg_invocation(tmp_path: Path) -> None:
    fake_input = tmp_path / "input.mp4"
    fake_input.touch()
    fake_output = tmp_path / "output.mp4"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        res = mask_burned_in_subtitles(fake_input, fake_output)
        assert res == fake_output
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd[0]
        assert "-filter_complex" in cmd


# --- Content Decision Engine Tests ---


def test_classify_production_mode_explicit_preference(tmp_path: Path) -> None:
    fake_vid = tmp_path / "test.mp4"
    fake_vid.touch()

    assert classify_production_mode(fake_vid, user_preference="speaker") == "speaker"
    assert classify_production_mode(fake_vid, user_preference="ai_gen") == "ai_gen"
    assert classify_production_mode(fake_vid, user_preference="SPEAKER") == "speaker"
    assert classify_production_mode(fake_vid, user_preference="AI_GEN") == "ai_gen"


def test_classify_production_mode_missing_file(tmp_path: Path) -> None:
    ghost = tmp_path / "ghost.mp4"
    # Should safely fallback to "speaker" without throwing
    assert classify_production_mode(ghost, user_preference="auto") == "speaker"


def test_classify_production_mode_detects_speaker(tmp_path: Path) -> None:
    fake_vid = tmp_path / "speaker_present.mp4"
    fake_vid.touch()

    mock_tracking = SpeakerTrackingResult(
        has_speaker=True,
        speaker_presence_ratio=0.45,  # > 0.20 threshold
        static_crop_x=100,
        static_crop_y=0,
        crop_expression="",
        crop_y_expression="",
        shot_crop_offsets=[],
    )

    with (
        patch("shorts_engine.services.content_decision_engine.probe_resolution", return_value=(1080, 1920)),
        patch("shorts_engine.services.content_decision_engine.track_active_speaker", return_value=mock_tracking),
    ):
        mode = classify_production_mode(fake_vid, user_preference="auto")
        assert mode == "speaker"


def test_classify_production_mode_detects_no_speaker(tmp_path: Path) -> None:
    fake_vid = tmp_path / "no_speaker.mp4"
    fake_vid.touch()

    mock_tracking = SpeakerTrackingResult(
        has_speaker=False,
        speaker_presence_ratio=0.05,  # < 0.20 threshold
        static_crop_x=0,
        static_crop_y=0,
        crop_expression="",
        crop_y_expression="",
        shot_crop_offsets=[],
    )

    with (
        patch("shorts_engine.services.content_decision_engine.probe_resolution", return_value=(1080, 1920)),
        patch("shorts_engine.services.content_decision_engine.track_active_speaker", return_value=mock_tracking),
    ):
        mode = classify_production_mode(fake_vid, user_preference="auto")
        assert mode == "ai_gen"


# --- AI Short Generator Models & Helpers Tests ---


def test_clean_json_markdown() -> None:
    raw = "```json\n{\"title\": \"Test\"}\n```"
    assert clean_json_markdown(raw) == '{"title": "Test"}'

    raw_no_fence = '{"title": "Direct"}'
    assert clean_json_markdown(raw_no_fence) == '{"title": "Direct"}'


def test_ai_short_package_creation() -> None:
    from shorts_engine.services.seo_generator import SeoMetadata

    scene1 = {
        "scene_index": 1,
        "narration_chunk": "Πρώτη σκηνή για την ενέργεια.",
        "pexels_query": "solar energy",
        "visual_prompt": "Close up of solar panels in Greece under sunny sky, vertical 9:16",
    }

    seo = SeoMetadata(
        title="Τι συμβαίνει με την ενέργεια;",
        description="Ανάλυση για τις τιμές ρεύματος.",
        tags=("ενέργεια", "ρεύμα"),
        primary_keyword="ενέργεια",
        pinned_comment="Σχολιάστε παρακάτω.",
        curiosity_title="curiosity",
        authority_title="authority",
        contrarian_title="contrarian",
    )
    pkg = AIShortPackage(
        title="Τι συμβαίνει με την ενέργεια;",
        hook="Αυτό που συμβαίνει με το ρεύμα δεν έχει προηγούμενο.",
        narration_script="Πρώτη σκηνή για την ενέργεια. Και η συνέχεια.",
        scenes=[scene1],
        seo=seo,
    )

    assert pkg.title == "Τι συμβαίνει με την ενέργεια;"
    assert pkg.hook.startswith("Αυτό")
    assert len(pkg.scenes) == 1
    assert pkg.scenes[0]["pexels_query"] == "solar energy"


def test_build_full_ai_short_flow(tmp_path: Path) -> None:
    from shorts_engine.config import Settings
    from shorts_engine.services.ai_short_generator import build_full_ai_short
    from shorts_engine.services.seo_generator import SeoMetadata
    from shorts_engine.services.transcriber import TranscriptionSegment

    settings = Settings(gemini_api_key="test_key", enable_bg_music=False)
    tmp_dir = tmp_path / "work"
    output_dir = tmp_path / "out"
    tmp_dir.mkdir()
    output_dir.mkdir()

    fake_pkg = AIShortPackage(
        title="AI News",
        hook="Hook",
        narration_script="Narration text in Greek",
        scenes=[{"scene_index": 1, "narration_chunk": "Narration text in Greek", "pexels_query": "news"}],
        seo=SeoMetadata(
            title="Title",
            description="Desc",
            tags=("test",),
            primary_keyword="news",
            pinned_comment="Comment",
            curiosity_title="c",
            authority_title="a",
            contrarian_title="co",
        ),
    )

    def fake_voiceover(text: str, out_path: Path, voice: str = "el-GR-NestorasNeural") -> Path:
        out_path.write_bytes(b"dummy audio")
        return out_path

    def fake_assemble(*args: object, **kwargs: object) -> Path:
        p = tmp_dir / "scene.mp4"
        p.write_bytes(b"dummy video")
        return p

    def fake_burn(input_path: Path, ass_path: Path, output_path: Path) -> Path:
        assert input_path.is_file()
        assert ass_path.is_file()
        output_path.write_bytes(b"dummy subtitled video")
        return output_path

    with (
        patch("shorts_engine.services.ai_short_generator.generate_script_and_scenes", return_value=fake_pkg),
        patch("shorts_engine.services.ai_short_generator.synthesize_voiceover", side_effect=fake_voiceover),
        patch("shorts_engine.services.ai_short_generator._assemble_scene_video", side_effect=fake_assemble),
        patch("shorts_engine.services.ai_short_generator.probe_duration", return_value=12.0),
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
        patch("shorts_engine.services.ai_short_generator.transcribe", return_value=[
            TranscriptionSegment(start=0.0, end=10.0, text="Narration text in Greek", words=[])
        ]),
        patch("shorts_engine.services.ai_short_generator.burn_subtitles", side_effect=fake_burn) as mock_burn,
    ):
        out_file, seo = build_full_ai_short(
            topic_title="Greek Energy",
            topic_context="Details",
            settings=settings,
            tmp_dir=tmp_dir,
            output_dir=output_dir,
        )

        assert out_file.is_file()
        assert seo.title == "Title"
        mock_burn.assert_called_once()
        assert "input_path" in mock_burn.call_args.kwargs
        assert "ass_path" in mock_burn.call_args.kwargs
        assert "output_path" in mock_burn.call_args.kwargs

