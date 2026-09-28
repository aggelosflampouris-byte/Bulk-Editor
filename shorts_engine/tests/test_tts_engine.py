"""
tests/test_tts_engine.py — Unit tests for Piper & Edge TTS Engine.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from shorts_engine.config import Settings
from shorts_engine.services.tts_engine import (
    DEFAULT_EDGE_VOICE,
    DEFAULT_PIPER_VOICE,
    EdgeTTSEngine,
    PiperTTSEngine,
    get_tts_engine,
    sanitize_voiceover_speech,
    synthesize_voiceover,
    synthesize_voiceover_with_segments,
)


def test_sanitize_voiceover_speech_greek_brand_mentions() -> None:
    """Verify channel names and brand handles are normalized to natural Greek pronunciation."""
    assert sanitize_voiceover_speech("Κάντε εγγραφή στο κανάλι @dianismanews!") == "Κάντε εγγραφή στο κανάλι Διάνυσμα!"
    assert sanitize_voiceover_speech("Ακολουθήστε το @dianisma") == "Ακολουθήστε το Διάνυσμα"
    assert sanitize_voiceover_speech("Ειδήσεις από το dianisma") == "Ειδήσεις από το Διάνυσμα"
    assert sanitize_voiceover_speech("**Σημαντικό!** #dianismanews") == "Σημαντικό! κανάλι Διάνυσμα"


class FakeAudioChunk:
    def __init__(self, duration: float = 2.0, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        num_samples = int(sample_rate * duration)
        self.audio_float_array = np.zeros(num_samples, dtype=np.float32)
        self.audio_int16_bytes = b"\x00\x00" * num_samples


def test_piper_tts_synthesize_real_or_mocked(tmp_path: Path) -> None:
    """Verify PiperTTSEngine computes millisecond-accurate segments and word timestamps."""
    out_wav = tmp_path / "piper_test.wav"
    engine = PiperTTSEngine()

    mock_voice = MagicMock()
    mock_voice.synthesize.return_value = [
        FakeAudioChunk(duration=2.0),
        FakeAudioChunk(duration=2.0),
    ]

    with patch.object(engine, "get_piper_voice", return_value=mock_voice):
        res_path, segments = engine.synthesize(
            text="Πρώτη πρόταση εδώ. Δεύτερη πρόταση εδώ.",
            output_path=out_wav,
            voice=DEFAULT_PIPER_VOICE,
            speed=1.0,
        )

        assert res_path == out_wav
        assert out_wav.is_file()
        assert len(segments) == 2

        # First segment: 0.0 -> 2.0s
        seg1 = segments[0]
        assert seg1.start == pytest.approx(0.0, 0.01)
        assert seg1.end == pytest.approx(2.0, 0.01)
        assert len(seg1.words) == 3
        assert seg1.words[0][2] == "Πρώτη"
        assert seg1.words[1][2] == "πρόταση"
        assert seg1.words[2][2] == "εδώ."

        # Second segment: 2.0 -> 4.0s
        seg2 = segments[1]
        assert seg2.start == pytest.approx(2.0, 0.01)
        assert seg2.end == pytest.approx(4.0, 0.01)
        assert len(seg2.words) == 3
        assert seg2.words[0][2] == "Δεύτερη"


def test_piper_tts_synthesize_mp3_conversion(tmp_path: Path) -> None:
    """Verify synthesis with .mp3 output triggers ffmpeg conversion safely."""
    out_mp3 = tmp_path / "piper_test.mp3"
    engine = PiperTTSEngine()

    mock_voice = MagicMock()
    mock_voice.synthesize.return_value = [FakeAudioChunk(duration=1.0)]

    with patch.object(engine, "get_piper_voice", return_value=mock_voice), \
         patch("subprocess.run") as mock_run:
        # Create output file in subprocess mock
        def fake_ffmpeg(*args, **kwargs):
            out_mp3.write_bytes(b"FAKE_MP3_DATA")
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_ffmpeg
        res_path, segments = engine.synthesize(
            text="Μια απλή δοκιμή.",
            output_path=out_mp3,
        )

        assert res_path == out_mp3
        assert out_mp3.is_file()
        assert len(segments) == 1
        assert segments[0].text == "Μια απλή δοκιμή."


def test_edge_tts_synthesize_mocked(tmp_path: Path) -> None:
    """Verify EdgeTTSEngine creates aligned segments from SentenceBoundary events."""
    out_mp3 = tmp_path / "edge_test.mp3"
    engine = EdgeTTSEngine()

    sample_sentence = {
        "type": "SentenceBoundary",
        "offset": 5000000,  # 0.5s
        "duration": 15000000,  # 1.5s
        "text": "Επίσημη δήλωση.",
    }

    async def fake_stream():
        yield {"type": "audio", "data": b"EDGE_AUDIO"}
        yield sample_sentence

    mock_comm = MagicMock()
    mock_comm.stream = fake_stream

    with patch("edge_tts.Communicate", return_value=mock_comm):
        res_path, segments = engine.synthesize(
            text="Επίσημη δήλωση.",
            output_path=out_mp3,
            voice=DEFAULT_EDGE_VOICE,
        )
        assert res_path == out_mp3
        assert out_mp3.is_file()
        assert len(segments) == 1
        assert segments[0].start == pytest.approx(0.5, 0.01)
        assert segments[0].end == pytest.approx(2.0, 0.01)


def test_synthesize_voiceover_fallback(tmp_path: Path) -> None:
    """Verify graceful fallback from Piper to Edge-TTS on model exception."""
    out_wav = tmp_path / "fallback_test.wav"

    mock_piper = MagicMock()
    mock_piper.synthesize.side_effect = RuntimeError("Piper ONNX model error")

    mock_edge = MagicMock()
    mock_edge.synthesize.return_value = (out_wav, [])

    with patch("shorts_engine.services.tts_engine.get_tts_engine") as mock_get:
        def fake_get_engine(name):
            return mock_piper if name == "piper" else mock_edge

        mock_get.side_effect = fake_get_engine
        path, segments = synthesize_voiceover_with_segments(
            text="Δοκιμή fallback.",
            output_path=out_wav,
            engine="piper",
        )
        assert mock_edge.synthesize.called


def test_settings_tts_validation() -> None:
    """Verify Settings validates tts_engine and tts_speed constraints."""
    settings = Settings()
    assert settings.validate() == []

    settings.tts_engine = "invalid_engine"
    errors = settings.validate()
    assert any("Invalid tts_engine" in err for err in errors)

    settings.tts_engine = "piper"
    settings.tts_speed = 0.2  # too slow
    errors = settings.validate()
    assert any("tts_speed" in err for err in errors)

    settings.tts_speed = 1.05
    assert settings.validate() == []
