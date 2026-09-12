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
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_PKG_DIR = Path(__file__).parent.resolve()
if str(_PKG_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR.parent))
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

try:
    from config import Settings, assert_system_binaries
except ImportError:
    from shorts_engine.config import Settings, assert_system_binaries
from services.cache_manager import log_project_history
from services.broll_fetcher import (
    BRollClip,
    download_clip,
    extract_broll_query,
    search_broll,
)
from services.clip_selector import ClipCandidate, select_clips
from services.compositor import compose_timeline
from services.downloader import download_video, probe_url_metadata
from services.face_tracker import track_active_speaker
from services.seo_generator import (
    SeoMetadata,
    correct_transcript_greek,
    generate_broll_query,
    generate_seo,
    seo_to_dict,
)
from services.timeline_utils import slice_segments, snap_to_silence
from services.transcriber import (
    TranscriptionSegment,
    full_transcript_text,
    transcribe,
    write_ass_file,
)
from services.vfx_engine import (
    SceneAnalysis,
    VfxPreset,
    analyse_scene_objects,
    apply_vfx,
    choose_vfx_preset,
)
from services.video_engine import (
    burn_subtitles,
    concatenate_with_outro,
    crop_to_9_16,
    is_already_9_16,
    mix_background_music,
    probe_duration,
    probe_resolution,
    slice_video,
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
    output_file: Path | None = None
    seo: SeoMetadata | None = None
    success: bool = False
    # Human-readable error message; set on failure
    error: str | None = None
    # Individual stage warnings (non-fatal, e.g. B-roll skipped)
    warnings: list[str] = field(default_factory=list)
    # The Pexels search query that was used for B-roll (for UI display)
    broll_query: str | None = None
    # Viral hook text and score detected by Qwen 2.5
    hook_text: str | None = None
    virality_score: float | None = None
    # Clip index when processing multi-clip URL mode
    clip_index: int | None = None


# ── Single-Video Pipeline ──────────────────────────────────────────────────────

def process_single(
    video_path: Path,
    settings: Settings,
    tmp_dir: Path,
    run_output_dir: Path,
    progress_cb: ProgressCallback | None = None,
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
        _report(f"Transcribing Greek speech (faster-whisper {settings.whisper_model_size})...")

        def _item_transcribe_cb(pct: float, msg: str) -> None:
            _report(msg)

        segments: list[TranscriptionSegment] = transcribe(
            video_path,
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            beam_size=settings.whisper_beam_size,
            progress_cb=_item_transcribe_cb,
        )
        transcript_text: str = full_transcript_text(segments)
        logger.info("Transcript (%d chars): %s...", len(transcript_text), transcript_text[:80])


        # ── Stage 1c: Gemini transcript correction ────────────────────────────
        # Fix Whisper transcription errors in the Greek text while keeping all
        # word-level timing intact (correction replaces text only, not timing).
        if settings.gemini_api_key:
            _report("Correcting transcript with Gemini...")
            segments = correct_transcript_greek(segments, settings.gemini_api_key)
            transcript_text = full_transcript_text(segments)
            logger.info("Corrected transcript: %s...", transcript_text[:80])

        # ── Stage 1d: SEO Generation (moved early for subtitle keyword injection) ──
        _report("Generating SEO metadata...")
        seo = generate_seo(
            transcript_text,
            settings.gemini_api_key,
            source_title=video_path.stem,
            brand_voice=settings.brand_voice,
        )

        # Write ASS subtitle file to scratch dir
        ass_path: Path = tmp_dir / f"{stem}.ass"
        write_ass_file(
            segments,
            ass_path,
            primary_keyword=seo.primary_keyword if seo else None,
        )

        # ── Stage 2: B-Roll Search ────────────────────────────────────────────
        broll_clip: BRollClip | None = None
        if settings.pexels_api_key:
            _report("Generating B-roll search query...")

            # Prefer a Gemini-generated query (semantically aware, English);
            # fall back to the stopword-based extractor when Gemini is absent.
            query: str = (
                generate_broll_query(transcript_text, settings.gemini_api_key)
                or extract_broll_query(transcript_text)
            )
            result.broll_query = query
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
        broll_path: Path | None = None
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

        # ── Stage 4: Crop to 9:16 (active speaker tracking) ───────────────────
        cropped_path: Path = tmp_dir / f"{stem}_cropped.mp4"

        # Validation Check: Ensure input produced a valid video stream
        try:
            probe_resolution(video_path)
        except Exception as exc:
            msg = f"Input video is invalid or missing video stream: {exc}"
            logger.error("URL pipeline aborting video: %s", msg)
            return ProcessingResult(
                input_file=video_path,
                clip_index=1,
                status="failed",
                error=msg
            )

        has_speaker: bool = False
        crop_x_offset: int | None = None
        crop_x_expr: str | None = None
        crop_y_offset: int | None = None
        crop_y_expr: str | None = None

        if settings.enable_face_tracking:
            try:
                src_w, src_h = probe_resolution(video_path)
                tracking_info = track_active_speaker(
                    video_path=video_path,
                    source_width=src_w,
                    source_height=src_h,
                    target_width=settings.target_width,
                    target_height=settings.target_height,
                )
                has_speaker = tracking_info.has_speaker
                crop_x_offset = tracking_info.static_crop_x
                crop_x_expr = tracking_info.crop_expression
                crop_y_offset = getattr(tracking_info, "static_crop_y", None)
                crop_y_expr = getattr(tracking_info, "crop_y_expression", None)
            except Exception as exc:
                logger.warning("Active speaker tracking failed: %s — using center-crop.", exc)

        if not is_already_9_16(video_path, settings.target_width, settings.target_height):
            _report("Cropping to 9:16 (active speaker tracking)...")
            crop_to_9_16(
                video_path,
                cropped_path,
                target_width=settings.target_width,
                target_height=settings.target_height,
                crop_x_offset=crop_x_offset,
                crop_x_expr=crop_x_expr,
                crop_y_offset=crop_y_offset,
                crop_y_expr=crop_y_expr,
            )
        else:
            import shutil as _shutil
            _shutil.copy2(str(video_path), str(cropped_path))
            _report("Crop skipped — video is already 9:16.")
            logger.info("Crop stage skipped for '%s' (already 9:16).", video_path.name)

        # ── Stage 5: B-Roll Overlay & Dynamic Zoom ────────────────────────────
        current_path = cropped_path
        broll_to_apply = broll_path
        if has_speaker:
            logger.info("Speaker recognized on screen — keeping speaker centered at all times; suppressing B-roll overlay.")
            warnings.append("Speaker recognized on screen — B-roll overlay suppressed to keep speaker in center at all times.")
            broll_to_apply = None

        if segments or broll_to_apply is not None:
            _report("Applying timeline effects (Dynamic Zoom / B-Roll)...")
            main_duration = probe_duration(cropped_path)
            actual_broll_dur = min(
                settings.broll_overlay_duration,
                max(1.0, main_duration - 1.0),
            )
            safe_start = min(
                settings.broll_start_offset,
                max(0.0, main_duration - actual_broll_dur),
            )
            overlaid_path: Path = tmp_dir / f"{stem}_overlaid.mp4"
            compose_timeline(
                main_video_path=cropped_path,
                output_path=overlaid_path,
                broll_video_path=broll_to_apply,
                broll_start=safe_start,
                broll_duration=actual_broll_dur,
                target_width=settings.target_width,
                target_height=settings.target_height,
                segments=segments,
                clip_start_offset=0.0,
                ken_burns=settings.broll_ken_burns,
                split_screen=settings.broll_split_screen,
            )
            current_path = overlaid_path

        # ── Stage 6: Subtitle Burn-in ─────────────────────────────────────────
        _report("Burning subtitles...")
        burned_path: Path = tmp_dir / f"{stem}_burned.mp4"
        burn_subtitles(current_path, ass_path, burned_path)
        current_path = burned_path

        # ── Stage 6c: VFX / Colour Grading ────────────────────────────────
        if settings.enable_vfx:
            _report("Analysing scene for VFX / colour grading...")
            try:
                scene: SceneAnalysis = analyse_scene_objects(
                    current_path,
                    model_path=settings.vfx_yolo_model,
                )
                preset: VfxPreset = choose_vfx_preset(transcript_text, scene)
                _report(f"Applying VFX preset: {preset.name}...")
                vfx_path: Path = tmp_dir / f"{stem}_vfx.mp4"
                apply_vfx(current_path, vfx_path, preset)
                current_path = vfx_path
                logger.info("VFX stage complete: preset=%s", preset.name)
            except Exception as exc:
                msg = f"VFX stage skipped: {exc}"
                logger.warning(msg)
                warnings.append(msg)

        # ── Stage 6d: Background Music ────────────────────────────────────────
        bg_music_path = settings.resolve_bg_music_path()
        if bg_music_path is not None:
            _report("Mixing background music...")
            bgm_path: Path = tmp_dir / f"{stem}_bgm.mp4"
            mix_background_music(
                video_path=current_path,
                music_path=bg_music_path,
                output_path=bgm_path,
                volume=settings.bg_music_volume,
                ducking=settings.bg_music_ducking,
            )
            current_path = bgm_path

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
                transition=settings.transition_type,
                transition_duration=settings.transition_duration,
            )
            current_path = final_tmp
        else:
            warnings.append("No outro provided — concatenation step skipped.")


        # ── Stage 9: Write Final Output ───────────────────────────────────────
        _report("Writing output files...")
        run_output_dir.mkdir(parents=True, exist_ok=True)

        final_output = run_output_dir / f"{stem}_short.mp4"
        seo_json_path = run_output_dir / f"seo_{stem}.json"

        # Move final video from scratch to output dir
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
        
        # Log successful project history
        try:
            log_project_history(result, run_output_dir)
        except Exception as e:
            logger.warning("Failed to log project history: %s", e)
            
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
    progress_cb: ProgressCallback | None = None,
) -> list[ProcessingResult]:
    return _run_batch_impl(video_paths, settings, progress_cb)




def _run_batch_impl(
    video_paths: list[Path],
    settings: Settings,
    progress_cb: ProgressCallback | None = None,
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

            main_duration = probe_duration(video_path)
            # If the video is longer than clip_max_duration, perform multi-clip extraction
            # guaranteeing minimum settings.min_clips (default: 3)
            if main_duration > settings.clip_max_duration:
                logger.info(
                    "Video '%s' duration (%.1fs) exceeds max clip duration (%.1fs) — running multi-clip extraction (minimum: %d)...",
                    video_path.name, main_duration, settings.clip_max_duration, settings.min_clips,
                )
                if progress_cb:
                    progress_cb(idx, total, f"[{video_path.name}] Transcribing audio...")

                def _item_transcribe_cb(pct: float, msg: str) -> None:
                    if progress_cb:
                        progress_cb(idx, total, f"[{video_path.name}] {msg}")

                all_segments = transcribe(
                    video_path,
                    model_size=settings.whisper_model_size,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                    beam_size=settings.whisper_beam_size,
                    progress_cb=_item_transcribe_cb,
                )

                if not all_segments:
                    logger.info("No speech segments detected in '%s' — generating synthetic segments across duration (%.1fs)...", video_path.name, main_duration)
                    step = main_duration / max(settings.min_clips, 1)
                    all_segments = [
                        TranscriptionSegment(
                            start=round(k * step, 2),
                            end=round(min((k + 1) * step, main_duration), 2),
                            text=f"{video_path.stem} clip {k + 1}",
                            words=[],
                        )
                        for k in range(settings.min_clips)
                    ]
                elif settings.gemini_api_key:
                    if progress_cb:
                        progress_cb(idx, total, f"[{video_path.name}] Correcting transcript with Gemini...")
                    all_segments = correct_transcript_greek(all_segments, settings.gemini_api_key)

                if progress_cb:
                    progress_cb(idx, total, f"[{video_path.name}] Selecting best clips with AI...")

                raw_candidates = select_clips(
                    segments=all_segments,
                    gemini_api_key=settings.gemini_api_key,
                    max_clips=settings.max_clips,
                    min_clips=settings.min_clips,
                    min_dur=settings.clip_min_duration,
                    max_dur=settings.clip_max_duration,
                    source_title=video_path.stem,
                    brand_voice=settings.brand_voice,
                )

                snapped: list[ClipCandidate] = []
                for cand in raw_candidates:
                    s_start, s_end = snap_to_silence(
                        start_time=cand.start_time,
                        end_time=cand.end_time,
                        segments=all_segments,
                        min_dur=settings.clip_min_duration,
                        max_dur=settings.clip_max_duration,
                    )
                    snapped.append(
                        ClipCandidate(
                            index=cand.index,
                            start_time=s_start,
                            end_time=s_end,
                            hook_summary=cand.hook_summary,
                            seo=cand.seo,
                            broll_query=cand.broll_query,
                        )
                    )

                logger.info(
                    "Selected %d clip candidates from '%s' (minimum guaranteed: %d):",
                    len(snapped), video_path.name, settings.min_clips,
                )
                print(f"\n{'='*65}\n🎬 SELECTED {len(snapped)} SHORTS FROM '{video_path.name}' (MINIMUM: {settings.min_clips}):\n{'='*65}")
                for c in snapped:
                    dur = c.end_time - c.start_time
                    title = c.seo.title if c.seo else "Clip"
                    logger.info("  [#%d] [%.1fs - %.1fs] (%.1fs): %s", c.index, c.start_time, c.end_time, dur, title)
                    print(f"  [#{c.index}] {c.start_time:.1f}s - {c.end_time:.1f}s ({dur:.1f}s) — \"{title}\"")
                print(f"{'='*65}\n")

                stem_prefix = video_path.stem
                item_results = []
                for c_idx, clip in enumerate(snapped):
                    per_clip_tmp = per_video_tmp / f"clip_{clip.index:02d}"
                    per_clip_tmp.mkdir(parents=True, exist_ok=True)
                    res = process_url_clip(
                        clip=clip,
                        source_path=video_path,
                        all_segments=all_segments,
                        settings=settings,
                        tmp_dir=per_clip_tmp,
                        run_output_dir=run_output_dir,
                        progress_cb=progress_cb,
                        item_index=c_idx,
                        total_items=len(snapped),
                        stem_prefix=stem_prefix or None,
                    )
                    item_results.append(res)
                    results.append(res)

                item_succ = sum(1 for r in item_results if r.success)
                print(f"\n{'='*65}\n✅ RENDERED {item_succ}/{len(snapped)} SHORTS FROM '{video_path.name}':\n{'='*65}")
                for idx_r, r in enumerate(item_results, 1):
                    c_num = r.clip_index if r.clip_index is not None else idx_r
                    st_icon = "✓" if r.success else "✗"
                    c_title = (r.seo.title if r.seo else None) or (r.output_file.name if r.output_file else f"Clip #{c_num}")
                    c_out = str(r.output_file) if r.output_file else "None"
                    print(f"  [{st_icon}] Clip #{c_num}: \"{c_title}\" -> {c_out}")
                print(f"{'='*65}\n")
            else:
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
        successful, len(results), run_output_dir,
    )
    return results


# ── URL Pipeline ───────────────────────────────────────────────────────────────


def process_url_clip(
    clip: ClipCandidate,
    source_path: Path,
    all_segments: list[TranscriptionSegment],
    settings: Settings,
    tmp_dir: Path,
    run_output_dir: Path,
    progress_cb: ProgressCallback | None = None,
    item_index: int = 0,
    total_items: int = 1,
    stem_prefix: str | None = None,
) -> ProcessingResult:
    """
    Run the full assembly pipeline for a single AI-selected clip.

    The source video is already downloaded and transcribed; this function
    slices the specific [start, end] window, re-bases subtitles, and runs
    the existing crop → B-roll → subtitle → outro chain.

    Args:
        clip:          The AI-selected clip (start/end times, SEO, B-roll query).
        source_path:   Path to the downloaded source video.
        all_segments:  Full transcription of the source (used for subtitle slicing).
        settings:      Validated Settings instance.
        tmp_dir:       Per-clip scratch directory for intermediate files.
        run_output_dir: Final output directory for this batch run.
        progress_cb:   Optional progress callback.
        item_index:    0-based clip index (for callback display).
        total_items:   Total clips being processed (for callback display).
        stem_prefix:   Optional prefix for output filename.

    Returns:
        A ProcessingResult summarising the outcome.
    """
    # Use a deterministic stem so files don't collide across clips
    if stem_prefix:
        stem = f"{stem_prefix}_clip_{clip.index:02d}"
    else:
        stem = f"clip_{clip.index:02d}"
    result = ProcessingResult(input_file=source_path, clip_index=clip.index)
    warnings: list[str] = []

    def _report(stage: str) -> None:
        label = f"[Clip {clip.index}] {stage}"
        logger.info("[%d/%d] %s", item_index + 1, total_items, label)
        if progress_cb:
            progress_cb(item_index, total_items, label)

    try:
        # ── Stage 1: Slice raw clip from source ────────────────────────────────
        _report(f"Slicing [{clip.start_display} → {clip.end_display}]...")
        raw_clip_path = tmp_dir / f"{stem}_raw.mp4"
        slice_video(
            source_path=source_path,
            start_time=clip.start_time,
            end_time=clip.end_time,
            output_path=raw_clip_path,
        )

        # Validation Check: Ensure extraction produced a valid video stream
        try:
            probe_resolution(raw_clip_path)
        except Exception as exc:
            msg = f"Extracted clip is invalid or missing video stream: {exc}"
            logger.error("Batch pipeline aborting clip %d: %s", clip.index, msg)
            return ProcessingResult(
                input_file=source_path,
                clip_index=clip.index,
                status="failed",
                error=msg
            )

        # ── Stage 2: Build re-based subtitle file ──────────────────────────────
        _report("Building subtitles...")
        clip_segments = slice_segments(all_segments, clip.start_time, clip.end_time)
        ass_path = tmp_dir / f"{stem}.ass"
        if clip_segments:
            write_ass_file(
                clip_segments,
                ass_path,
                primary_keyword=clip.seo.primary_keyword if clip.seo else None,
            )
        else:
            warnings.append(f"Clip {clip.index}: no transcript segments in window — subtitles skipped.")
            ass_path = None  # type: ignore[assignment]

        # ── Stage 3: B-Roll Search (use per-clip query from AI) ────────────────
        broll_clip: BRollClip | None = None
        if settings.pexels_api_key:
            _report(f"Searching B-roll: '{clip.broll_query}'...")
            result.broll_query = clip.broll_query
            broll_clip = search_broll(clip.broll_query, settings.pexels_api_key)
            if broll_clip is None:
                msg = f"Clip {clip.index}: no B-roll found for '{clip.broll_query}' — skipping overlay."
                logger.warning(msg)
                warnings.append(msg)
        else:
            warnings.append(f"Clip {clip.index}: Pexels key absent — B-roll skipped.")

        # ── Stage 4: B-Roll Download ───────────────────────────────────────────
        broll_path: Path | None = None
        if broll_clip is not None:
            _report("Downloading B-roll...")
            broll_dest = tmp_dir / f"{stem}_broll.mp4"
            try:
                broll_path = download_clip(broll_clip, broll_dest)
            except RuntimeError as exc:
                msg = f"Clip {clip.index}: B-roll download failed: {exc}"
                logger.warning(msg)
                warnings.append(msg)

        # ── Stage 4: Crop to 9:16 (active speaker tracking) ───────────────────
        cropped_path = tmp_dir / f"{stem}_cropped.mp4"
        

        has_speaker: bool = False
        crop_x_offset: int | None = None
        crop_x_expr: str | None = None
        crop_y_offset: int | None = None
        crop_y_expr: str | None = None

        if settings.enable_face_tracking:
            try:
                src_w, src_h = probe_resolution(raw_clip_path)
                tracking_info = track_active_speaker(
                    video_path=raw_clip_path,
                    source_width=src_w,
                    source_height=src_h,
                    target_width=settings.target_width,
                    target_height=settings.target_height,
                )
                has_speaker = tracking_info.has_speaker
                crop_x_offset = tracking_info.static_crop_x
                crop_x_expr = tracking_info.crop_expression
                crop_y_offset = tracking_info.static_crop_y
                crop_y_expr = tracking_info.crop_y_expression
            except Exception as exc:
                logger.warning("Active speaker tracking failed: %s — using center-crop.", exc)

        if is_already_9_16(raw_clip_path, settings.target_width, settings.target_height):
            shutil.copy2(str(raw_clip_path), str(cropped_path))
            _report("Crop skipped — already 9:16.")
        else:
            _report("Cropping to 9:16 (active speaker tracking)...")
            crop_to_9_16(
                raw_clip_path,
                cropped_path,
                target_width=settings.target_width,
                target_height=settings.target_height,
                crop_x_offset=crop_x_offset,
                crop_x_expr=crop_x_expr,
                crop_y_offset=crop_y_offset,
                crop_y_expr=crop_y_expr,
            )

        # ── Stage 6: B-Roll Overlay & Dynamic Zoom ─────────────────────────────────────────
        current_path = cropped_path
        
        broll_to_apply = broll_path
        if has_speaker:
            logger.info(
                "Speaker recognized on screen for clip %d — centering speaker at all times; suppressing B-roll overlay.",
                clip.index,
            )
            warnings.append(f"Clip {clip.index}: speaker recognized on screen — B-roll overlay suppressed to keep speaker in center at all times.")
            broll_to_apply = None

        if all_segments or broll_to_apply is not None:
            _report("Applying timeline effects (Dynamic Zoom / B-Roll)...")
            main_dur = probe_duration(cropped_path)
            
            actual_broll_dur = min(
                settings.broll_overlay_duration,
                max(1.0, main_dur - 1.0),
            )
            safe_start = min(
                settings.broll_start_offset,
                max(0.0, main_dur - actual_broll_dur),
            )
            overlaid_path = tmp_dir / f"{stem}_overlaid.mp4"
            compose_timeline(
                main_video_path=cropped_path,
                output_path=overlaid_path,
                broll_video_path=broll_to_apply,
                broll_start=safe_start,
                broll_duration=actual_broll_dur,
                target_width=settings.target_width,
                target_height=settings.target_height,
                segments=clip_segments,
                clip_start_offset=clip.start_time,
                ken_burns=settings.broll_ken_burns,
                split_screen=settings.broll_split_screen,
            )
            current_path = overlaid_path

        # ── Stage 7: Subtitle Burn-in ──────────────────────────────────────────
        if ass_path is not None and ass_path.is_file():
            _report("Burning subtitles...")
            burned_path = tmp_dir / f"{stem}_burned.mp4"
            burn_subtitles(current_path, ass_path, burned_path)
            current_path = burned_path
        else:
            warnings.append(f"Clip {clip.index}: subtitle burn skipped (no .ass file).")

        # ── Stage 7b: VFX / Colour Grading ────────────────────────────────────
        if settings.enable_vfx:
            clip_transcript = " ".join(s.text for s in clip_segments) if clip_segments else ""
            _report("Analysing scene for VFX / colour grading...")
            try:
                scene: SceneAnalysis = analyse_scene_objects(
                    current_path,
                    model_path=settings.vfx_yolo_model,
                )
                preset: VfxPreset = choose_vfx_preset(clip_transcript, scene)
                _report(f"Applying VFX preset: {preset.name}...")
                vfx_path = tmp_dir / f"{stem}_vfx.mp4"
                apply_vfx(current_path, vfx_path, preset)
                current_path = vfx_path
                logger.info(
                    "VFX stage complete for clip %d: preset=%s", clip.index, preset.name
                )
            except Exception as exc:
                msg = f"Clip {clip.index}: VFX stage skipped: {exc}"
                logger.warning(msg)
                warnings.append(msg)

        # ── Stage 7c: Background Music ─────────────────────────────────────────
        bg_music_path = settings.resolve_bg_music_path()
        if bg_music_path is not None:
            _report("Mixing background music...")
            bgm_path = tmp_dir / f"{stem}_bgm.mp4"
            mix_background_music(
                video_path=current_path,
                music_path=bg_music_path,
                output_path=bgm_path,
                volume=settings.bg_music_volume,
                ducking=settings.bg_music_ducking,
            )
            current_path = bgm_path

        # ── Stage 8: Outro Concatenation ───────────────────────────────────────
        if settings.outro_path is not None:
            _report("Concatenating outro...")
            final_tmp = tmp_dir / f"{stem}_with_outro.mp4"
            concatenate_with_outro(
                main_path=current_path,
                outro_path=settings.outro_path,
                output_path=final_tmp,
                target_width=settings.target_width,
                target_height=settings.target_height,
                transition=settings.transition_type,
                transition_duration=settings.transition_duration,
            )
            current_path = final_tmp
        else:
            warnings.append(f"Clip {clip.index}: no outro — concatenation skipped.")

        # ── Stage 9: Write Final Output ────────────────────────────────────────
        _report("Writing output files...")
        run_output_dir.mkdir(parents=True, exist_ok=True)

        final_output = run_output_dir / f"{stem}_short.mp4"
        seo_json_path = run_output_dir / f"seo_{stem}.json"

        shutil.copy2(str(current_path), str(final_output))

        seo_json_path.write_text(
            json.dumps(seo_to_dict(clip.seo), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        result.output_file = final_output
        result.seo = clip.seo
        result.success = True
        result.warnings = warnings
        _report("Done ✓")

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error(
            "URL clip pipeline failed for clip %d: %s", clip.index, error_msg, exc_info=True
        )
        result.error = error_msg
        result.success = False
        result.warnings = warnings

    return result


def run_url_pipeline(
    url: str,
    settings: Settings,
    clip_indices: list[int] | None = None,
    progress_cb: ProgressCallback | None = None,
) -> tuple[list[ClipCandidate], list[ProcessingResult]]:
    """
    Run the complete URL-driven pipeline: download → transcribe → select → assemble.

    This is the primary entry point for the URL input mode. It differs from
    run_batch in two key ways:
      1. A single source URL (not a list of uploaded files) is the input.
      2. The Gemini clip selector decides which moments to extract; the user
         can pre-filter via *clip_indices* (from the review step in the UI).

    Stage order:
      1. Validate settings and assert system binaries.
      2. Probe URL metadata (title, channel, duration).
      3. Download source video into scratch directory.
      4. Single-pass transcription (faster-whisper) of the source video.
      5. Optional Gemini transcript correction.
      6. Gemini clip selection → list[ClipCandidate].
      7. Boundary snapping for each candidate.
      8. For each selected (and user-approved) clip: full assembly pipeline.
      9. Return (all_candidates, assembly_results).

    Args:
        url:           Source video URL (YouTube, Vimeo, direct MP4, etc.).
        settings:      Validated Settings instance.
        clip_indices:  If provided, only assemble clips at these 1-based indices.
                       None = assemble all selected candidates.
        progress_cb:   Optional callback called at each stage.

    Returns:
        A tuple of:
          - all_candidates: list[ClipCandidate] (all AI-selected candidates,
            regardless of clip_indices, for UI display purposes).
          - results: list[ProcessingResult] (one per assembled clip).

    Raises:
        RuntimeError: If system binary assertions fail.
        ValueError:   If the URL is empty or the source video is too long.
    """
    assert_system_binaries()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = settings.output_dir / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("URL pipeline output directory: '%s'", run_output_dir)

    all_candidates: list[ClipCandidate] = []
    results: list[ProcessingResult] = []

    def _report(stage: str) -> None:
        logger.info("[URL Pipeline] %s", stage)
        if progress_cb:
            progress_cb(0, 1, stage)

    with tempfile.TemporaryDirectory(prefix="shorts_url_") as tmp_root:
        tmp_dir = Path(tmp_root)

        # ── Phase 1: Download ──────────────────────────────────────────────────
        _report("Probing URL metadata...")
        url_meta = probe_url_metadata(url, settings.max_source_duration_seconds)
        logger.info(
            "Source: '%s' by '%s' (%s)",
            url_meta.title, url_meta.channel, url_meta.duration_display,
        )

        _report(f"Downloading '{url_meta.title}'...")
        download_dir = tmp_dir / "source"
        source_path = download_video(url, download_dir, settings.max_source_duration_seconds)

        # ── Phase 2: Transcription ─────────────────────────────────────────────
        _report("Transcribing audio (loading Whisper model)...")

        def _url_transcribe_cb(pct: float, msg: str) -> None:
            _report(msg)

        all_segments = transcribe(
            source_path,
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            beam_size=settings.whisper_beam_size,
            progress_cb=_url_transcribe_cb,
        )

        if settings.gemini_api_key:
            _report("Correcting transcript with Gemini...")
            all_segments = correct_transcript_greek(all_segments, settings.gemini_api_key)

        # ── Phase 3: AI Clip Selection ─────────────────────────────────────────
        _report("Selecting best clips with AI...")
        raw_candidates = select_clips(
            segments=all_segments,
            gemini_api_key=settings.gemini_api_key,
            max_clips=settings.max_clips,
            min_clips=settings.min_clips,
            min_dur=settings.clip_min_duration,
            max_dur=settings.clip_max_duration,
            source_title=url_meta.title,
        )

        # ── Phase 4: Boundary Snapping ─────────────────────────────────────────
        snapped: list[ClipCandidate] = []
        for cand in raw_candidates:
            snapped_start, snapped_end = snap_to_silence(
                start_time=cand.start_time,
                end_time=cand.end_time,
                segments=all_segments,
                min_dur=settings.clip_min_duration,
                max_dur=settings.clip_max_duration,
            )
            snapped.append(
                ClipCandidate(
                    index=cand.index,
                    start_time=snapped_start,
                    end_time=snapped_end,
                    hook_summary=cand.hook_summary,
                    seo=cand.seo,
                    broll_query=cand.broll_query,
                )
            )

        all_candidates = snapped

        # Print and log the selected clips with minimum guaranteed count
        logger.info(
            "Selected %d clip candidates (minimum guaranteed: %d):",
            len(all_candidates), settings.min_clips,
        )
        print(f"\n{'='*65}\n🎬 SELECTED {len(all_candidates)} SHORTS (MINIMUM: {settings.min_clips}):\n{'='*65}")
        for c in all_candidates:
            dur = c.end_time - c.start_time
            title = c.seo.title if c.seo else "Clip"
            logger.info("  [#%d] [%.1fs - %.1fs] (%.1fs): %s", c.index, c.start_time, c.end_time, dur, title)
            print(f"  [#{c.index}] {c.start_time:.1f}s - {c.end_time:.1f}s ({dur:.1f}s) — \"{title}\"")
        print(f"{'='*65}\n")

        # Filter to user-approved indices if provided
        if clip_indices is not None:
            to_process = [c for c in snapped if c.index in clip_indices]
        else:
            to_process = snapped

        total_clips = len(to_process)
        logger.info("Assembling %d/%d selected clips...", total_clips, len(snapped))

        # ── Phase 5: Per-clip Assembly ─────────────────────────────────────────
        for idx, clip in enumerate(to_process):
            per_clip_tmp = tmp_dir / f"clip_{clip.index:02d}"
            per_clip_tmp.mkdir(parents=True, exist_ok=True)

            result = process_url_clip(
                clip=clip,
                source_path=source_path,
                all_segments=all_segments,
                settings=settings,
                tmp_dir=per_clip_tmp,
                run_output_dir=run_output_dir,
                progress_cb=progress_cb,
                item_index=idx,
                total_items=total_clips,
            )
            results.append(result)

    successful = sum(1 for r in results if r.success)
    logger.info(
        "URL pipeline complete: %d/%d clips succeeded. Output: '%s'",
        successful, total_clips, run_output_dir,
    )
    print(f"\n{'='*65}\n✅ RENDERED {successful}/{total_clips} SHORTS:\n{'='*65}")
    for idx, r in enumerate(results, 1):
        c_idx = r.clip_index if r.clip_index is not None else idx
        status_icon = "✓" if r.success else "✗"
        title = (r.seo.title if r.seo else None) or (r.output_file.name if r.output_file else f"Clip #{c_idx}")
        out_path = str(r.output_file) if r.output_file else "None"
        print(f"  [{status_icon}] Clip #{c_idx}: \"{title}\" -> {out_path}")
    print(f"{'='*65}\n")
    return all_candidates, results
