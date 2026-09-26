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
from datetime import datetime, timezone
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
if str(_PKG_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR.parent))

try:
    from config import Settings, assert_system_binaries
except ImportError:
    from shorts_engine.config import Settings, assert_system_binaries
from services.broll_fetcher import download_clip, extract_broll_query, search_broll
from services.cache_manager import log_project_history
from services.clip_selector import ClipCandidate, select_clips
from services.compositor import compose_timeline
from services.downloader import download_video, probe_url_metadata
from services.face_tracker import track_active_speaker
from services.logic_guardrail import RetentionAudit, audit_retention_signals
from services.ocr_engine import OCREngine
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
    # Source YouTube video ID if known
    source_video_id: str | None = None
    # Inflowave-inspired hook & retention audit diagnostic
    retention_audit: RetentionAudit | None = None


# ── Shared Stage Helpers ──────────────────────────────────────────────────────


def _apply_crop_stage(
    source_path: Path,
    settings: Settings,
    output_path: Path,
    report_fn: Callable[[str], None],
) -> bool:
    """
    Run the 9:16 crop and active speaker face-tracking stage.

    Encapsulates the probe → track → crop (or copy) sequence that is shared
    between `process_single` and `process_url_clip`.  Returns whether a
    speaker was detected, so the caller can decide whether to suppress B-roll.

    Args:
        source_path: Path to the input video clip.
        settings:    Validated Settings instance (provides target dimensions,
                     face-tracking toggle, and YOLO model path).
        output_path: Destination path for the cropped output.
        report_fn:   Progress-reporting callback (calls `_report` internally).

    Returns:
        True if an active speaker was detected (B-roll should be suppressed),
        False otherwise.

    Raises:
        RuntimeError: Propagated from `probe_resolution` if the video is
            unreadable; the caller is responsible for handling this.
    """
    has_speaker: bool = False
    crop_x_offset: int | None = None
    crop_x_expr: str | None = None
    crop_y_offset: int | None = None
    crop_y_expr: str | None = None

    if settings.enable_face_tracking:
        try:
            src_w, src_h = probe_resolution(source_path)
            tracking_info = track_active_speaker(
                video_path=source_path,
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

    src_w, src_h = probe_resolution(source_path)
    # Strict 9:16 check: only skip crop if resolution matches exact target dimensions (within 2px)
    if (
        is_already_9_16(source_path, settings.target_width, settings.target_height)
        and abs(src_w - settings.target_width) <= 2
        and abs(src_h - settings.target_height) <= 2
    ):
        shutil.copy2(str(source_path), str(output_path))
        report_fn("Crop skipped — video is already 1080x1920 (9:16).")
        logger.info("Crop stage skipped for '%s' (already exact 9:16).", source_path.name)
    else:
        report_fn("Cropping to 9:16 (active speaker tracking)...")
        crop_to_9_16(
            source_path,
            output_path,
            target_width=settings.target_width,
            target_height=settings.target_height,
            crop_x_offset=crop_x_offset,
            crop_x_expr=crop_x_expr,
            crop_y_offset=crop_y_offset,
            crop_y_expr=crop_y_expr,
        )

    return has_speaker


# ── Single-Video Pipeline ──────────────────────────────────────────────────────


def build_short_from_clip(
    video_path: Path,
    stem: str,
    segments: list[TranscriptionSegment],
    transcript_text: str,
    seo,
    broll_query: str | None,
    settings: Settings,
    tmp_dir: Path,
    run_output_dir: Path,
    _report,
    warnings: list[str],
    custom_broll_path: Path | None = None,
    clip_start_offset: float = 0.0,
):
    """Core assembly logic used by both local files and downloaded URL clips."""
    # Write ASS subtitle file
    ass_path: Path = tmp_dir / f"{stem}.ass"
    if segments:
        chosen_style = getattr(settings, "caption_style", "auto")
        if chosen_style in ("auto", "", None):
            try:
                from services.caption_styles import recommend_caption_style
            except ImportError:
                from shorts_engine.services.caption_styles import recommend_caption_style
            rec_template = recommend_caption_style(
                niche=getattr(settings, "niche_template", ""),
                topic_or_title=seo.title if seo else "",
                transcript_sample=" ".join(s.text for s in segments[:10]),
            )
            chosen_style = rec_template.id

        write_ass_file(
            segments,
            ass_path,
            primary_keyword=seo.primary_keyword if seo else None,
            subtitle_position=getattr(settings, "subtitle_position", "lower_third"),
            subtitle_mode=getattr(settings, "subtitle_mode", "dynamic"),
            caption_style=chosen_style,
        )
    else:
        warnings.append("No transcript segments provided — subtitles skipped.")
        ass_path = None

    # B-Roll Download
    broll_path: Path | None = None
    if custom_broll_path:
        _report("Using custom B-roll")
        broll_path = custom_broll_path
    elif broll_query and settings.pexels_api_key:
        _report(f"Searching for B-roll: '{broll_query}'...")
        broll_clip = search_broll(broll_query, settings.pexels_api_key)
        if broll_clip is None:
            msg = f"No suitable B-roll found for '{broll_query}' — skipping."
            logger.warning(msg)
            warnings.append(msg)
        else:
            _report("Downloading B-roll clip...")
            broll_dest = tmp_dir / f"{stem}_broll.mp4"
            try:
                broll_path = download_clip(broll_clip, broll_dest)
            except RuntimeError as exc:
                msg = f"B-roll download failed: {exc} — skipping overlay."
                logger.warning(msg)
                warnings.append(msg)
                broll_path = None

    # Crop to 9:16 (active speaker tracking)
    cropped_path: Path = tmp_dir / f"{stem}_cropped.mp4"
    try:
        probe_resolution(video_path)
    except Exception as exc:
        raise RuntimeError(f"Input video invalid or missing stream: {exc}")

    has_speaker = _apply_crop_stage(video_path, settings, cropped_path, _report)

    # B-Roll Overlay & Dynamic Zoom
    current_path = cropped_path
    broll_to_apply = broll_path
    if has_speaker:
        logger.info("Speaker recognized — suppressing B-roll overlay.")
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
            clip_start_offset=clip_start_offset,
            ken_burns=settings.broll_ken_burns,
            split_screen=settings.broll_split_screen,
            enable_dynamic_zoom=getattr(settings, "enable_dynamic_zoom", False),
        )
        current_path = overlaid_path

    # Subtitle Burn-in
    if ass_path is not None and ass_path.is_file():
        if getattr(settings, "mask_old_subtitles", True):
            _report("Masking old burned-in subtitles with frosted plate...")
            masked_path: Path = tmp_dir / f"{stem}_masked.mp4"
            try:
                from services.subtitle_masker import mask_burned_in_subtitles
                current_path = mask_burned_in_subtitles(current_path, masked_path)
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning("Subtitle masking skipped: %s", exc)

        _report("Burning subtitles...")
        burned_path: Path = tmp_dir / f"{stem}_burned.mp4"
        burn_subtitles(current_path, ass_path, burned_path)
        current_path = burned_path
    else:
        warnings.append("Subtitle burn skipped (no .ass file).")

    # VFX / Colour Grading
    if settings.enable_vfx:
        _report("Analysing scene for VFX / colour grading...")
        try:
            scene = analyse_scene_objects(
                current_path,
                model_path=settings.vfx_yolo_model,
            )
            preset = choose_vfx_preset(transcript_text, scene)
            _report(f"Applying VFX preset: {preset.name}...")
            vfx_path: Path = tmp_dir / f"{stem}_vfx.mp4"
            apply_vfx(current_path, vfx_path, preset)
            current_path = vfx_path
            logger.info("VFX stage complete: preset=%s", preset.name)
        except Exception as exc:
            msg = f"VFX stage skipped: {exc}"
            logger.warning(msg)
            warnings.append(msg)

    # Background Music
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

    # Outro Concatenation
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

    # Write Final Output
    _report("Writing output files...")
    run_output_dir.mkdir(parents=True, exist_ok=True)

    final_output = run_output_dir / f"{stem}_short.mp4"
    
    shutil.copy2(str(current_path), str(final_output))

    if seo:
        seo_json_path = run_output_dir / f"seo_{stem}.json"
        seo_json_path.write_text(
            json.dumps(seo_to_dict(seo), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return final_output


def process_single(
    video_path: Path,
    settings: Settings,
    tmp_dir: Path,
    run_output_dir: Path,
    progress_cb: ProgressCallback | None = None,
    item_index: int = 0,
    total_items: int = 1,
) -> ProcessingResult:
    result = ProcessingResult(input_file=video_path)

    def _report(stage: str) -> None:
        logger.info("[%d/%d] %s — %s", item_index + 1, total_items, video_path.name, stage)
        if progress_cb:
            progress_cb(item_index, total_items, stage)

    stem = video_path.stem
    warnings: list[str] = []

    try:
        _report(f"Transcribing Greek speech (faster-whisper {settings.whisper_model_size})...")

        def _item_transcribe_cb(pct: float, msg: str) -> None:
            _report(msg)

        segments = transcribe(
            video_path,
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            beam_size=settings.whisper_beam_size,
            progress_cb=_item_transcribe_cb,
            context_hint=settings.whisper_context_hint or None,
            source_title=stem,
        )
        transcript_text: str = full_transcript_text(segments)

        if settings.gemini_api_key:
            _report("Correcting transcript with Gemini...")
            segments = correct_transcript_greek(segments, settings.gemini_api_key)
            transcript_text = full_transcript_text(segments)

        _report("Generating SEO metadata...")
        seo = generate_seo(
            transcript_text,
            settings.gemini_api_key,
            source_title=video_path.stem,
            brand_voice=settings.brand_voice,
        )
        
        query = None
        if settings.pexels_api_key:
            query = (
                generate_broll_query(transcript_text, settings.gemini_api_key)
                or extract_broll_query(transcript_text)
            )
            result.broll_query = query

        final_output = build_short_from_clip(
            video_path=video_path,
            stem=stem,
            segments=segments,
            transcript_text=transcript_text,
            seo=seo,
            broll_query=query,
            settings=settings,
            tmp_dir=tmp_dir,
            run_output_dir=run_output_dir,
            _report=_report,
            warnings=warnings,
        )

        result.output_file = final_output
        result.seo = seo
        result.success = True
        result.warnings = warnings

        try:
            from services.video_engine import probe_duration
            out_dur = probe_duration(final_output)
            result.retention_audit = audit_retention_signals(
                segments=segments,
                text=transcript_text,
                duration=out_dur,
                hook_summary=seo.title if seo else "",
                title=seo.title if seo else "",
            )
            result.virality_score = round(result.retention_audit.score / 10.0, 1)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            logger.warning("Retention audit failed for single video: %s", exc)

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


# ── Shared Transcribe + Select Helper ────────────────────────────────────────


def _transcribe_and_select(
    video_path: Path,
    settings: Settings,
    progress_cb: Callable[[float, str], None] | None,
    source_title: str,
    main_duration: float,
) -> tuple[list[TranscriptionSegment], list[ClipCandidate]]:
    """
    Shared helper: transcribe → correct → OCR → select → snap.

    Extracted to eliminate the identical block that previously lived in both
    ``run_batch`` (multi-clip path) and ``run_url_pipeline``.

    Args:
        video_path:    Path to the source video.
        settings:      Active Settings instance.
        progress_cb:   Per-segment transcription progress callback (pct, msg).
        source_title:  Title injected into the Whisper prompt.
        main_duration: Pre-probed duration of the source video (seconds).

    Returns:
        (all_segments, snapped_candidates)
    """
    all_segments = transcribe(
        video_path,
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        beam_size=settings.whisper_beam_size,
        progress_cb=progress_cb,
        context_hint=settings.whisper_context_hint or None,
        source_title=source_title,
    )

    if not all_segments:
        logger.info(
            "No speech segments in '%s' — synthesising %d placeholder segments (%.1fs).",
            video_path.name, settings.min_clips, main_duration,
        )
        step = main_duration / max(settings.min_clips, 1)
        all_segments = [
            TranscriptionSegment(
                start=round(k * step, 2),
                end=round(min((k + 1) * step, main_duration), 2),
                text=f"{source_title} clip {k + 1}",
                words=[],
            )
            for k in range(settings.min_clips)
        ]
    elif settings.gemini_api_key:
        all_segments = correct_transcript_greek(all_segments, settings.gemini_api_key)

    ocr_text = ""
    try:
        ocr_engine = OCREngine(gemini_api_key=settings.gemini_api_key)
        ocr_text = ocr_engine.extract_text_from_video(video_path, sample_rate_sec=5)
    except Exception as exc:
        logger.warning("OCR extraction failed for '%s': %s", video_path.name, exc)

    raw_candidates = select_clips(
        segments=all_segments,
        gemini_api_key=settings.gemini_api_key,
        max_clips=settings.max_clips,
        min_clips=settings.min_clips,
        min_dur=settings.clip_min_duration,
        max_dur=settings.clip_max_duration,
        source_title=source_title,
        brand_voice=settings.brand_voice,
        channel_niche=settings.whisper_context_hint or "",
        ocr_text=ocr_text,
        niche_template=getattr(settings, "niche_template", "custom"),
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
                emotional_intensity=cand.emotional_intensity,
                standalone_narrative=cand.standalone_narrative,
                hook_potency=cand.hook_potency,
                speech_velocity=cand.speech_velocity,
            )
        )

    return all_segments, snapped


def render_selected_clips(
    source_path: Path,
    all_segments: list[TranscriptionSegment],
    clips: list[ClipCandidate],
    settings: Settings,
    progress_cb: ProgressCallback | None = None,
) -> list[ProcessingResult]:
    """
    Assemble a pre-selected list of ClipCandidates from an already-downloaded
    source video, without repeating download or transcription.

    This is the dedicated render-phase entry point for the URL tab, which
    caches ``source_path`` and ``all_segments`` from the analysis phase and
    calls this function on "Render" to avoid re-downloading/re-transcribing.

    Args:
        source_path:  Path to the already-downloaded source video.
        all_segments: Transcription segments from the analysis phase.
        clips:        AI-selected ClipCandidates to assemble.
        settings:     Active Settings instance.
        progress_cb:  Optional progress callback.

    Returns:
        List of ProcessingResult, one per assembled clip.
    """
    assert_system_binaries()

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_output_dir = settings.output_dir / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Render-phase output directory: '%s'", run_output_dir)

    results: list[ProcessingResult] = []
    total = len(clips)

    with tempfile.TemporaryDirectory(prefix="shorts_render_") as tmp_root:
        tmp_dir = Path(tmp_root)
        for idx, clip in enumerate(clips):
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
                total_items=total,
            )
            results.append(result)

    successful = sum(1 for r in results if r.success)
    logger.info("Render complete: %d/%d clips succeeded.", successful, total)
    return results


# ── Batch Orchestrator ─────────────────────────────────────────────────────────




def run_batch(
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
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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

                def _batch_progress_cb(pct: float, msg: str) -> None:
                    if progress_cb:
                        progress_cb(idx, total, f"[{video_path.name}] {msg}")

                if progress_cb:
                    progress_cb(idx, total, f"[{video_path.name}] Transcribing audio...")

                all_segments, snapped = _transcribe_and_select(
                    video_path=video_path,
                    settings=settings,
                    progress_cb=_batch_progress_cb,
                    source_title=video_path.stem,
                    main_duration=main_duration,
                )

                logger.info(
                    "Selected %d clip candidates from '%s' (minimum guaranteed: %d):",
                    len(snapped), video_path.name, settings.min_clips,
                )
                for c in snapped:
                    dur = c.end_time - c.start_time
                    title = c.seo.title if c.seo else "Clip"
                    logger.info("  [#%d] [%.1fs - %.1fs] (%.1fs): %s", c.index, c.start_time, c.end_time, dur, title)

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
                        stem_prefix=stem_prefix,
                    )
                    item_results.append(res)
                    results.append(res)

                item_succ = sum(1 for r in item_results if r.success)
                logger.info(
                    "Rendered %d/%d shorts from '%s'.",
                    item_succ, len(snapped), video_path.name,
                )
                for idx_r, r in enumerate(item_results, 1):
                    c_num = r.clip_index if r.clip_index is not None else idx_r
                    icon = "✓" if r.success else "✗"
                    c_title = (r.seo.title if r.seo else None) or (r.output_file.name if r.output_file else f"Clip #{c_num}")
                    logger.info("  [%s] Clip #%d: '%s'", icon, c_num, c_title)
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
    custom_broll_path: str | Path | None = None,
    source_is_section: bool = False,
    source_offset: float = 0.0,
    source_video_id: str | None = None,
) -> ProcessingResult:
    if stem_prefix:
        stem = f"{stem_prefix}_clip_{clip.index:02d}"
    else:
        stem = f"clip_{clip.index:02d}"
    result = ProcessingResult(
        input_file=source_path,
        clip_index=clip.index,
        source_video_id=source_video_id,
    )
    warnings: list[str] = []

    def _report(stage: str) -> None:
        label = f"[Clip {clip.index}] {stage}"
        logger.info("[%d/%d] %s", item_index + 1, total_items, label)
        if progress_cb:
            progress_cb(item_index, total_items, label)

    try:
        _report(f"Slicing [{clip.start_display} → {clip.end_display}]...")
        raw_clip_path = tmp_dir / f"{stem}_raw.mp4"
        if source_is_section:
            s_start = source_offset
            s_end = source_offset + (clip.end_time - clip.start_time)
        else:
            s_start = clip.start_time
            s_end = clip.end_time

        slice_video(
            source_path=source_path,
            start_time=s_start,
            end_time=s_end,
            output_path=raw_clip_path,
        )

        try:
            probe_resolution(raw_clip_path)
        except Exception as exc:
            msg = f"Extracted clip is invalid or missing video stream: {exc}"
            logger.error("Batch pipeline aborting clip %d: %s", clip.index, msg)
            return ProcessingResult(input_file=source_path, clip_index=clip.index, success=False, error=msg)

        clip_segments = slice_segments(all_segments, clip.start_time, clip.end_time)

        if settings.gemini_api_key and clip_segments:
            _report("AI logic scan: correcting captions for grammar and gibberish...")
            clip_segments = correct_transcript_greek(clip_segments, settings.gemini_api_key)
            
        clip_transcript = " ".join(s.text for s in clip_segments) if clip_segments else ""
        
        final_output = build_short_from_clip(
            video_path=raw_clip_path,
            stem=stem,
            segments=clip_segments,
            transcript_text=clip_transcript,
            seo=clip.seo,
            broll_query=clip.broll_query,
            settings=settings,
            tmp_dir=tmp_dir,
            run_output_dir=run_output_dir,
            _report=_report,
            warnings=warnings,
            custom_broll_path=Path(custom_broll_path) if custom_broll_path else None,
            clip_start_offset=clip.start_time,
        )

        result.output_file = final_output
        result.seo = clip.seo
        result.success = True
        result.warnings = warnings
        result.virality_score = clip.virality_score
        result.hook_text = clip.hook_summary

        try:
            from services.video_engine import probe_duration
            actual_dur = probe_duration(final_output)
            result.retention_audit = audit_retention_signals(
                segments=clip_segments,
                text=clip_transcript,
                duration=actual_dur,
                hook_summary=clip.hook_summary,
                title=clip.seo.title if clip.seo else "",
            )
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            logger.warning("Retention audit failed for clip %d: %s", clip.index, exc)

        try:
            log_project_history(result, run_output_dir)
        except Exception as exc:
            logger.warning("Failed to log project history for clip %d: %s", clip.index, exc)

        _report("Done ✓")

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("URL clip pipeline failed for clip %d: %s", clip.index, error_msg, exc_info=True)
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

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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

        ocr_text = ""
        try:
            _report("Extracting OCR visual context...")
            ocr_engine = OCREngine(gemini_api_key=settings.gemini_api_key)
            ocr_text = ocr_engine.extract_text_from_video(source_path, sample_rate_sec=5)
        except Exception as exc:
            logger.warning("OCR extraction failed for %s: %s", source_path.name, exc)

        # ── Phase 3: AI Clip Selection (Single Best Clip Workflow) ─────────────
        _report("Selecting the single best clip with AI...")
        raw_candidates = select_clips(
            segments=all_segments,
            gemini_api_key=settings.gemini_api_key,
            max_clips=1,
            min_clips=1,
            min_dur=20.0,
            max_dur=45.0,
            source_title=url_meta.title,
            brand_voice=settings.brand_voice,
            channel_niche=settings.whisper_context_hint or "",
            ocr_text=ocr_text,
            niche_template=getattr(settings, "niche_template", "custom"),
        )

        # ── Phase 4: Boundary Snapping ─────────────────────────────────────────
        snapped: list[ClipCandidate] = []
        for cand in raw_candidates:
            snapped_start, snapped_end = snap_to_silence(
                start_time=cand.start_time,
                end_time=cand.end_time,
                segments=all_segments,
                min_dur=20.0,
                max_dur=45.0,
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

        logger.info(
            "Selected %d clip candidates (minimum guaranteed: %d):",
            len(all_candidates), settings.min_clips,
        )
        for c in all_candidates:
            dur = c.end_time - c.start_time
            title = c.seo.title if c.seo else "Clip"
            logger.info("  [#%d] [%.1fs - %.1fs] (%.1fs): %s", c.index, c.start_time, c.end_time, dur, title)

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
    for idx, r in enumerate(results, 1):
        c_idx = r.clip_index if r.clip_index is not None else idx
        icon = "✓" if r.success else "✗"
        title = (r.seo.title if r.seo else None) or (r.output_file.name if r.output_file else f"Clip #{c_idx}")
        logger.info("  [%s] Clip #%d: '%s'", icon, c_idx, title)
    return all_candidates, results
