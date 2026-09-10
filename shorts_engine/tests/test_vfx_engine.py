"""
tests/test_vfx_engine.py — Unit tests for services/vfx_engine.py.

Tests are structured in two groups:
  1. choose_vfx_preset — pure logic, no I/O, no YOLO, fast.
  2. apply_vfx         — requires FFmpeg; uses synthetic 2-second video
     (same fixture pattern as test_video_engine.py).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

try:
    from shorts_engine.services.vfx_engine import (
        SceneAnalysis,
        VfxPreset,
        apply_vfx,
        choose_vfx_preset,
    )
except ModuleNotFoundError:
    from services.vfx_engine import (
        SceneAnalysis,
        VfxPreset,
        apply_vfx,
        choose_vfx_preset,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_video(path: Path, duration: float = 2.0, with_audio: bool = True) -> Path:
    """Create a minimal synthetic MP4 for FFmpeg testing."""
    video_filter = f"color=c=blue:size=1080x1920:rate=30,trim=duration={duration}"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", video_filter,
    ]
    if with_audio:
        cmd += [
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration}",
            "-c:a", "aac", "-b:a", "128k",
        ]
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "ultrafast", "-crf", "28",
        "-movflags", "+faststart",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


# ── choose_vfx_preset tests ────────────────────────────────────────────────────


class TestChooseVfxPreset:
    """Pure logic tests — no I/O required."""

    def _empty_scene(self) -> SceneAnalysis:
        return SceneAnalysis()

    def test_dramatic_keyword_takes_priority(self) -> None:
        transcript = "Πρόβλημα στην πόλη, κίνδυνος για όλους!"
        preset = choose_vfx_preset(transcript, self._empty_scene())
        assert preset is VfxPreset.DRAMATIC

    def test_high_energy_keyword_triggers_vibrance(self) -> None:
        transcript = "Απίστευτο! Τέλειο αποτέλεσμα, φοβερό!"
        # No dramatic keywords, so VIBRANCE should win.
        preset = choose_vfx_preset(transcript, self._empty_scene())
        assert preset is VfxPreset.VIBRANCE

    def test_dramatic_beats_high_energy(self) -> None:
        # Both dramatic and high-energy keywords present — dramatic has priority.
        transcript = "Απίστευτο τέλος, αυτό ήταν η αποκάλυψη!"
        preset = choose_vfx_preset(transcript, self._empty_scene())
        assert preset is VfxPreset.DRAMATIC

    def test_person_dominant_scene_triggers_warmth(self) -> None:
        scene = SceneAnalysis(
            person_frame_ratio=0.75,
            detected_classes={"person"},
            dominant_class=None,
        )
        preset = choose_vfx_preset("Μιλάμε για τη ζωή.", scene)
        assert preset is VfxPreset.WARMTH

    def test_no_person_triggers_cinematic(self) -> None:
        scene = SceneAnalysis(
            person_frame_ratio=0.0,
            detected_classes={"car", "truck"},
            dominant_class="car",
        )
        preset = choose_vfx_preset("Αυτοκίνητα στον δρόμο.", scene)
        assert preset is VfxPreset.CINEMATIC

    def test_empty_scene_empty_transcript_falls_back_to_subtle(self) -> None:
        preset = choose_vfx_preset("", self._empty_scene())
        assert preset is VfxPreset.SUBTLE

    def test_partial_person_ratio_below_threshold_falls_back(self) -> None:
        # person_frame_ratio=0.40 is below 0.60 threshold — SUBTLE
        scene = SceneAnalysis(
            person_frame_ratio=0.40,
            detected_classes={"person"},
        )
        preset = choose_vfx_preset("Κάποιος πέρασε από εκεί.", scene)
        assert preset is VfxPreset.SUBTLE

    def test_transcript_none_equivalent_empty_string(self) -> None:
        # Ensure passing empty string doesn't raise.
        preset = choose_vfx_preset("", SceneAnalysis())
        assert isinstance(preset, VfxPreset)


# ── apply_vfx tests ────────────────────────────────────────────────────────────


class TestApplyVfx:
    """Integration tests that call FFmpeg — require ffmpeg on PATH."""

    @pytest.fixture()
    def video_with_audio(self, tmp_path: Path) -> Path:
        return _make_video(tmp_path / "src_audio.mp4", duration=2.0, with_audio=True)

    @pytest.fixture()
    def video_no_audio(self, tmp_path: Path) -> Path:
        return _make_video(tmp_path / "src_silent.mp4", duration=2.0, with_audio=False)

    @pytest.mark.parametrize("preset", list(VfxPreset))
    def test_all_presets_produce_output_file(
        self, preset: VfxPreset, video_with_audio: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / f"out_{preset.name}.mp4"
        result = apply_vfx(video_with_audio, out, preset)
        assert result == out
        assert out.is_file()
        assert out.stat().st_size > 0

    def test_audio_stream_preserved_after_vfx(
        self, video_with_audio: Path, tmp_path: Path
    ) -> None:
        """Audio must be stream-copied through apply_vfx unchanged."""
        out = tmp_path / "out_audio_check.mp4"
        apply_vfx(video_with_audio, out, VfxPreset.SUBTLE)

        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(out),
            ],
            capture_output=True, text=True,
        )
        assert "audio" in probe.stdout

    def test_silent_video_passes_through_without_error(
        self, video_no_audio: Path, tmp_path: Path
    ) -> None:
        """apply_vfx must not crash on a video with no audio track."""
        out = tmp_path / "out_silent_vfx.mp4"
        result = apply_vfx(video_no_audio, out, VfxPreset.CINEMATIC)
        assert result.is_file()

    def test_raises_file_not_found_for_missing_input(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            apply_vfx(tmp_path / "nonexistent.mp4", tmp_path / "out.mp4", VfxPreset.SUBTLE)
