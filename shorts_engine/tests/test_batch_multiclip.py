"""
tests/test_batch_multiclip.py — Unit tests for batch multi-clip extraction on long uploaded video files.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shorts_engine.config import Settings
from shorts_engine.pipeline import process_batch, ProcessingResult
from shorts_engine.services.clip_selector import ClipCandidate
from shorts_engine.services.seo_generator import SeoMetadata
from shorts_engine.services.transcriber import TranscriptionSegment


def test_process_batch_long_video_triggers_multiclip_extraction(tmp_path: Path):
    dummy_video = tmp_path / "long_speech.mp4"
    dummy_video.write_bytes(b"dummy")

    settings = Settings(
        min_clips=3,
        max_clips=5,
        clip_min_duration=30.0,
        clip_max_duration=50.0,
        output_dir=tmp_path / "output",
        enable_bg_music=False,
    )

    mock_segments = [
        TranscriptionSegment(start=float(i * 10), end=float((i + 1) * 10), text=f"Ομιλία {i}")
        for i in range(12)  # 120s total
    ]

    mock_candidates = [
        ClipCandidate(
            index=i + 1,
            start_time=float(i * 35),
            end_time=float((i + 1) * 35),
            hook_summary=f"Hook {i + 1}",
            seo=SeoMetadata(title=f"Clip {i + 1}", description=f"Desc {i + 1}", tags=["tag"]),
            broll_query="greek speech",
        )
        for i in range(3)
    ]

    def mock_process_clip(**kwargs):
        clip = kwargs["clip"]
        out_file = kwargs["run_output_dir"] / f"clip_{clip.index:02d}_short.mp4"
        out_file.write_bytes(b"short")
        return ProcessingResult(
            input_file=kwargs["source_path"],
            output_file=out_file,
            seo=clip.seo,
            success=True,
            clip_index=clip.index,
        )

    with patch("shorts_engine.pipeline.assert_system_binaries"), \
         patch("shorts_engine.pipeline.probe_duration", return_value=120.0), \
         patch("shorts_engine.pipeline.transcribe", return_value=mock_segments), \
         patch("shorts_engine.pipeline.select_clips", return_value=mock_candidates), \
         patch("shorts_engine.pipeline.snap_to_silence", side_effect=lambda start_time, end_time, **kw: (start_time, end_time)), \
         patch("shorts_engine.pipeline.process_url_clip", side_effect=mock_process_clip):

        results = process_batch(
            video_paths=[dummy_video],
            settings=settings,
        )

        # Must return at least 3 clips from the 120s video
        assert len(results) == 3
        for i, res in enumerate(results):
            assert res.success is True
            assert res.clip_index == i + 1
            assert res.output_file is not None
            assert res.output_file.exists()
