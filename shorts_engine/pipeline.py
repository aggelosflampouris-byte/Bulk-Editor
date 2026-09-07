"""
pipeline.py — Batch orchestrator for the Greek Shorts Processing Engine.

Responsibilities:
  1. Accept a list of input video paths and a Settings object.
  2. Wrap the entire batch in a TemporaryDirectory for scratch-disk discipline.
  3. For each video: transcribe → fetch B-roll → crop → overlay → burn subs
     → concat outro → generate SEO → write output files.
  4. Report per-video progress via an optional callback.
  5. Return a list of typed ProcessingResult objects summarising each outcome.

This module contains zero UI logic.  All FFmpeg calls, API calls, and model
inference are delegated to the services layer.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from config import Settings, assert_system_binaries
from services.broll_fetcher import (
    BRollClip,
    download_clip,
    extract_broll_query,
    search_broll,
)
from services.seo_generator import SeoMetadata, generate_seo, generate_broll_query, seo_to_dict
from services.transcriber import (
    TranscriptionSegment,
    full_transcript_text,
    transcribe,
    write_ass_file,
)
from services.video_engine import (
    burn_subtitles,
    concatenate_with_outro,
    crop_to_9_16,
    overlay_broll,
    probe_duration,
)

logger = logging.getLogger(__name__)

# ── Progress Callback Type ─────────────────────────────────────────────────────

# Signature: (current_index: int, total: int, stage: str) -> None
ProgressCallback = Callable[[int, int, str], None]


# ── Result Types ───────────────────────────────────────────────────────────────

@dataclass
class ProcessingResult:
    """
    Typed summary of the outcome for a single video processed through
    the full pipeline.
    """

    input_file: Path
    output_file: Optional[Path] = None
    seo: Optional[SeoMetadata] = None
    success: bool = False
    # Human-readable error message; set on failure
    error: Optional[str] = None
    # Individual stage warnings (non-fatal, e.g. B-roll skipped)
    warnings: list[str] = field(default_factory=list)


# ── Single-Video Pipeline ──────────────────────────────────────────────────────

def process_single(
    video_path: Path,
    settings: Settings,
    tmp_dir: Path,
    run_output_dir: Path,
    progress_cb: Optional[ProgressCallback] = None,
    item_index: int = 0,
    total_items: int = 1,
) -> ProcessingResult:
    """
    Run the complete processing pipeline for a single input video.

    Intermediate files are written to *tmp_dir* (a temporary scratch space).
    The final output video and SEO JSON are written to *run_output_dir*.

    Stage order:
      1. Transcribe (faster-whisper)
      2. Search B-roll (Pexels) — skipped if key absent/quota exceeded
      3. Download B-roll clip — skipped if search returned None
      4. Crop to 9:16
      5. Overlay B-roll — skipped if no clip downloaded
      6. Burn ASS subtitles
      7. Concatenate outro — skipped if outro_path is None
      8. Generate SEO metadata (Gemini)
      9. Write output files

    Args:
        video_path:      Absolute path to the source video.
        settings:        Validated Settings instance.
        tmp_dir:         Scratch directory for intermediate files.
        run_output_dir:  Final output directory for this batch run.
        progress_cb:     Optional callback called at each stage.
        item_index:      0-based index of this item in the batch (for callback).
        total_items:     Total items in the batch (for callback).

    Returns:
        A ProcessingResult summarising the outcome.
    """
    result = ProcessingResult(input_file=video_path)

    def _report(stage: str) -> None:
        logger.info("[%d/%d] %s — %s", item_index + 1, total_items, video_path.name, stage)
        if progress_cb:
            progress_cb(item_index, total_items, stage)

    stem = video_path.stem
    warnings: list[str] = []

    try:
        # ── Stage 1: Transcription ────────────────────────────────────────────
        _report("Transcribing Greek speech...")
        segments: list[TranscriptionSegment] = transcribe(
            video_path,
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        transcript_text: str = full_transcript_text(segments)
        logger.info("Transcript (%d chars): %s...", len(transcript_text), transcript_text[:80])

        # Write ASS subtitle file to scratch dir
        ass_path: Path = tmp_dir / f"{stem}.ass"
        write_ass_file(segments, ass_path)

        # ── Stage 2: B-Roll Search ────────────────────────────────────────────
        broll_clip: Optional[BRollClip] = None
        if settings.pexels_api_key:
            _report("Generating B-roll search query...")

            # Prefer a Gemini-generated query (semantically aware, English);
            # fall back to the stopword-based extractor when Gemini is absent.
            query: str = (
                generate_broll_query(transcript_text, settings.gemini_api_key)
                or extract_broll_query(transcript_text)
            )
            logger.info("B-roll search query: '%s'", query)

            _report(f"Searching for B-roll: '{query}'...")
            broll_clip = search_broll(query, settings.pexels_api_key)
            if broll_clip is None:
                msg = f"No suitable B-roll found for query '{query}' — skipping overlay."
                logger.warning(msg)
                warnings.append(msg)
        else:
            msg = "Pexels API key not provided — B-roll step skipped."
            logger.warning(msg)
            warnings.append(msg)

        # ── Stage 3: B-Roll Download ──────────────────────────────────────────
        broll_path: Optional[Path] = None
        if broll_clip is not None:
            _report("Downloading B-roll clip...")
            broll_dest = tmp_dir / f"{stem}_broll.mp4"
            try:
                broll_path = download_clip(broll_clip, broll_dest)
            except RuntimeError as exc:
                msg = f"B-roll download failed: {exc} — skipping overlay."
                logger.warning(msg)
                warnings.append(msg)
                broll_path = None

        # ── Stage 4: Crop to 9:16 ────────────────────────────────────────────
        _report("Cropping to 9:16 (1080×1920)...")
        cropped_path: Path = tmp_dir / f"{stem}_cropped.mp4"
        crop_to_9_16(
            video_path,
            cropped_path,
            target_width=settings.target_width,
            target_height=settings.target_height,
        )

        # ── Stage 5: B-Roll Overlay ───────────────────────────────────────────
        current_path = cropped_path
        if broll_path is not None:
            _report("Applying B-roll overlay...")
            main_duration = probe_duration(cropped_path)
            # Clamp start offset so overlay fits within the video
            safe_start = min(
                settings.broll_start_offset,
                max(0.0, main_duration - settings.broll_overlay_duration),
            )
            overlaid_path: Path = tmp_dir / f"{stem}_overlaid.mp4"
            overlay_broll(
                main_path=cropped_path,
                broll_path=broll_path,
                start_time=safe_start,
                overlay_duration=settings.broll_overlay_duration,
                output_path=overlaid_path,
                target_width=settings.target_width,
                target_height=settings.target_height,
            )
            current_path = overlaid_path

        # ── Stage 6: Subtitle Burn-in ─────────────────────────────────────────
        _report("Burning subtitles...")
        burned_path: Path = tmp_dir / f"{stem}_burned.mp4"
        burn_subtitles(current_path, ass_path, burned_path)
        current_path = burned_path

        # ── Stage 7: Outro Concatenation ──────────────────────────────────────
        if settings.outro_path is not None:
            _report("Concatenating outro...")
            final_tmp: Path = tmp_dir / f"{stem}_with_outro.mp4"
            concatenate_with_outro(
                main_path=current_path,
                outro_path=settings.outro_path,
                output_path=final_tmp,
                target_width=settings.target_width,
                target_height=settings.target_height,
            )
            current_path = final_tmp
        else:
            warnings.append("No outro provided — concatenation step skipped.")

        # ── Stage 8: SEO Generation ───────────────────────────────────────────
        _report("Generating SEO metadata...")
        seo = generate_seo(transcript_text, settings.gemini_api_key)

        # ── Stage 9: Write Final Output ───────────────────────────────────────
        _report("Writing output files...")
        run_output_dir.mkdir(parents=True, exist_ok=True)

        final_output = run_output_dir / f"{stem}_short.mp4"
        seo_json_path = run_output_dir / f"seo_{stem}.json"

        # Move final video from scratch to output dir
        import shutil
        shutil.copy2(str(current_path), str(final_output))

        # Write SEO JSON alongside the video
        seo_json_path.write_text(
            json.dumps(seo_to_dict(seo), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        result.output_file = final_output
        result.seo = seo
        result.success = True
        result.warnings = warnings
        _report("Done ✓")

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Pipeline failed for '%s': %s", video_path.name, error_msg, exc_info=True)
        result.error = error_msg
        result.success = False
        result.warnings = warnings

    return result


# ── Batch Orchestrator ─────────────────────────────────────────────────────────

def run_batch(
    video_paths: list[Path],
    settings: Settings,
    progress_cb: Optional[ProgressCallback] = None,
) -> list[ProcessingResult]:
    """
    Process a list of video files through the complete pipeline.

    The entire batch shares a single TemporaryDirectory for scratch files,
    which is automatically cleaned up (even on failure) when the batch ends.

    A timestamped output subdirectory is created under settings.output_dir
    to avoid collisions between batch runs.

    Args:
        video_paths: Ordered list of source video paths.
        settings:    Validated Settings instance.
        progress_cb: Optional callback — called per stage per video.

    Returns:
        List of ProcessingResult, one per input video, in input order.

    Raises:
        RuntimeError: If system binary assertions fail (ffmpeg/ffprobe missing).
    """
    if not video_paths:
        return []

    # Fail fast before spending time on transcription
    assert_system_binaries()

    # Timestamped output directory for this batch run
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = settings.output_dir / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Batch output directory: '%s'", run_output_dir)

    results: list[ProcessingResult] = []
    total = len(video_paths)

    with tempfile.TemporaryDirectory(prefix="shorts_engine_") as tmp_root:
        tmp_dir = Path(tmp_root)
        logger.info("Scratch directory: '%s'", tmp_dir)

        for idx, video_path in enumerate(video_paths):
            logger.info("─── Processing [%d/%d]: '%s' ───", idx + 1, total, video_path.name)

            # Each video gets its own subdirectory in the scratch space
            per_video_tmp = tmp_dir / f"item_{idx:04d}"
            per_video_tmp.mkdir(parents=True, exist_ok=True)

            result = process_single(
                video_path=video_path,
                settings=settings,
                tmp_dir=per_video_tmp,
                run_output_dir=run_output_dir,
                progress_cb=progress_cb,
                item_index=idx,
                total_items=total,
            )
            results.append(result)

    successful = sum(1 for r in results if r.success)
    logger.info(
        "Batch complete: %d/%d succeeded. Output: '%s'",
        successful, total, run_output_dir,
    )
    return results
