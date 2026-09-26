"""
services/autopilot.py — End-to-End Autopilot Orchestration
"""

import logging
import math
import tempfile
import uuid
from collections.abc import Generator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import Settings
from google.genai.errors import APIError
from pipeline import process_url_clip

try:
    from services.cache_manager import (
        is_video_already_processed,
        record_processed_video,
    )
except ImportError:
    from shorts_engine.services.cache_manager import (
        is_video_already_processed,
        record_processed_video,
    )
from services.channel_analyzer import (
    DIANISMA_CHANNEL_URL,
    VideoMeta,
    fetch_youtube_videos,
    find_viral_recent_videos,
)
from services.clip_selector import ClipCandidate, select_clips
from services.downloader import download_video, download_video_section
from services.seo_generator import generate_seo
from services.timeline_utils import snap_to_silence
from services.transcriber import transcribe

try:
    from shorts_engine.services.youtube_transcript_fetcher import (
        YouTubeTranscriptError,
        fetch_youtube_transcript,
    )
except ImportError:
    from services.youtube_transcript_fetcher import (
        YouTubeTranscriptError,
        fetch_youtube_transcript,
    )
from services.youtube_uploader import (
    YouTubeAuthError,
    authenticate,
    fetch_my_recent_videos,
    is_authenticated,
)

try:
    from shorts_engine.services.niche_sourcing import find_niche_trend_videos
except ImportError:
    from services.niche_sourcing import find_niche_trend_videos

try:
    from shorts_engine.services.ai_short_generator import build_full_ai_short
except ImportError:
    from services.ai_short_generator import build_full_ai_short

logger = logging.getLogger(__name__)


try:
    from services.traffic_scheduler import (
        get_optimal_schedule_slot,
        get_optimal_schedule_time,
    )
except ImportError:
    from shorts_engine.services.traffic_scheduler import (
        get_optimal_schedule_slot,
        get_optimal_schedule_time,
    )


def run_autopilot_pipeline(
    target_url: str = "",
    settings: Settings = None,
    broll_path: str = "",
    num_videos: int = 1,
    selected_video: VideoMeta | None = None,
    production_strategy: str = "auto",
    niche_query: str = "",
    local_video_paths: list[Path] | None = None,
) -> Generator[tuple[str, int, Any], None, None]:
    """
    Runs the entire pipeline end-to-end for the top `num_videos` videos.
    If local_video_paths is provided, processes those local video files directly.
    If selected_video is provided, processes that exact video from the library.
    By default (empty target_url), searches YouTube for fresh viral videos in our niche (≤ 3 weeks old)
    and strictly excludes @DianismaNews to avoid re-using our own channel's older content.

    Supports production modes:
      - "hybrid" (Recommended: Authentic speaker clip + AI breakdown & 9:16 B-roll)
      - "speaker" (Preserves speaker footage, obscures old captions with frosted plate, overlays B-roll)
      - "ai_gen"  (Full AI Short generation with original Greek script, voiceover, 9:16 media)
      - "auto" (Dynamically detects on-camera speaker presence)

    Yields (status_message, progress_percentage, result_data).
    """
    if local_video_paths:
        best_videos = []
        for p in local_video_paths[:num_videos]:
            p = Path(p)
            best_videos.append(
                VideoMeta(
                    video_id=f"local_{p.stem}_{abs(hash(str(p)))}",
                    title=p.stem.replace("_", " ").title(),
                    url=str(p.resolve()),
                    duration_seconds=0,
                    view_count=0,
                    upload_date=datetime.now(timezone.utc).strftime("%Y%m%d"),
                    channel_title="Local Video",
                    description=f"Local video: {p.name}",
                )
            )
        yield (f"Loaded {len(best_videos)} local video file(s) for intelligent autopilot processing.", 8, None)
    elif selected_video is not None:
        best_videos = [selected_video]
        yield (f"Selected video from Dianisma library: {selected_video.title}", 8, None)
    elif target_url and any(
        x in target_url.lower() for x in ("watch?v=", "youtu.be/", "/shorts/")
    ):
        yield (f"Analyzing specific video from URL: {target_url}...", 5, None)
        videos = fetch_youtube_videos(target_url, max_videos=1)
        best_videos = videos[:num_videos]
    elif (
        target_url
        and target_url.strip()
        and target_url.strip() != DIANISMA_CHANNEL_URL
        and not any(x in target_url.lower() for x in ("uczmnsmxzae4m_hzh6g1jkg",))
    ):
        # Specific channel URL or search term explicitly requested by caller
        videos = []
        is_dianisma = target_url.strip().lower() == "@dianismanews"
        if is_dianisma and is_authenticated():
            try:
                yield (
                    "Sourcing uploads from Dianisma library via YouTube Data API v3...",
                    5,
                    None,
                )
                client = authenticate()
                videos = fetch_my_recent_videos(client, max_videos=30)
            except (
                YouTubeAuthError,
                RuntimeError,
                OSError,
                ValueError,
                KeyError,
                AttributeError,
            ):
                videos = []

        if not videos:
            yield (f"Analyzing external channel uploads from {target_url}...", 5, None)
            videos = fetch_youtube_videos(target_url, max_videos=30)

        viral_picks = find_viral_recent_videos(
            videos, max_results=max(10, num_videos * 2)
        )
        candidates = [vp.video for vp in viral_picks] if viral_picks else list(videos)
        unprocessed = [
            v
            for v in candidates
            if not is_video_already_processed(
                v.video_id, settings.output_dir if settings else None
            )
            and not is_video_already_processed(
                v.url, settings.output_dir if settings else None
            )
        ]
        best_videos = (unprocessed if unprocessed else candidates)[:num_videos]
    else:
        # Default Autopilot Mode (also active when DIANISMA_CHANNEL_URL or empty URL is passed):
        # Search YouTube for fresh viral videos in our niche (≤ 3 weeks old)
        # Excludes @DianismaNews to avoid re-using our own channel's older content
        yield (
            "Searching YouTube for fresh viral videos in our niche (≤ 3 weeks old)...",
            5,
            None,
        )

        yt_client = None
        if is_authenticated():
            try:
                yt_client = authenticate()
            except (YouTubeAuthError, RuntimeError, OSError, ValueError):
                yt_client = None

        search_q = niche_query.strip() if niche_query.strip() else None

        candidates = find_niche_trend_videos(
            query=search_q,
            max_videos=max(10, num_videos * 3),
            max_age_days=21,
            youtube_client=yt_client,
            output_dir=settings.output_dir if settings else None,
        )

        unprocessed = [
            v
            for v in candidates
            if not is_video_already_processed(
                v.video_id, settings.output_dir if settings else None
            )
            and not is_video_already_processed(
                v.url, settings.output_dir if settings else None
            )
        ]

        if len(unprocessed) < num_videos:
            # Resilient fallback: search broader niche topics across YouTube (never our own channel)
            yield (
                "Searching broader niche topics across YouTube (≤ 3 weeks old)...",
                7,
                None,
            )
            try:
                from services.niche_sourcing import (
                    DEFAULT_NICHE_QUERIES,
                    fetch_niche_videos_via_ytdlp,
                )
            except ImportError:
                from shorts_engine.services.niche_sourcing import (
                    DEFAULT_NICHE_QUERIES,
                    fetch_niche_videos_via_ytdlp,
                )

            import random

            shuffled_fallback = list(DEFAULT_NICHE_QUERIES)
            random.shuffle(shuffled_fallback)

            existing_ids = {v.video_id for v in unprocessed}
            for fallback_query in shuffled_fallback:
                fallback_videos = fetch_niche_videos_via_ytdlp(
                    query=fallback_query,
                    max_results=max(10, num_videos * 3),
                    max_age_days=21,
                    output_dir=settings.output_dir if settings else None,
                )
                for fv in fallback_videos:
                    if (
                        fv.video_id not in existing_ids
                        and not is_video_already_processed(
                            fv.video_id, settings.output_dir if settings else None
                        )
                        and not is_video_already_processed(
                            fv.url, settings.output_dir if settings else None
                        )
                    ):
                        existing_ids.add(fv.video_id)
                        unprocessed.append(fv)
                        if len(unprocessed) >= num_videos:
                            break
                if len(unprocessed) >= num_videos:
                    break

        best_videos = unprocessed[:num_videos]

    if not best_videos:
        raise ValueError(
            "Could not find any new unprocessed videos in the niche (≤ 3 weeks old). "
            "All top candidates have already been processed. Please provide a custom topic or keyword focus."
        )

    yield (f"Selected {len(best_videos)} highly viral videos for processing.", 10, None)

    # Configure Autopilot production settings:
    # 1. Strictly 9:16 ratio (1080x1920)
    # 2. Less VFX and post-prod (enable_vfx=False, clean framing, no split screens)
    # 3. Dynamic and fluid subtitles (subtitle_mode="dynamic")
    # 4. Mask burned-in subtitles to prevent caption overlapping (mask_old_subtitles=True)
    # 5. Instrumental royalty-free background music bed with voice ducking (enable_bg_music=True)
    active_bg_track = getattr(settings, "bg_music_track", "ambient_calm")
    if not active_bg_track or active_bg_track in ("none", ""):
        active_bg_track = "ambient_calm"

    ap_settings = replace(
        settings,
        target_width=1080,
        target_height=1920,
        enable_vfx=False,
        enable_dynamic_zoom=False,
        broll_split_screen=False,
        broll_ken_burns=False,
        subtitle_mode="dynamic",
        mask_old_subtitles=getattr(settings, "mask_old_subtitles", False),
        enable_bg_music=True,
        bg_music_track=active_bg_track,
        bg_music_volume=getattr(settings, "bg_music_volume", 0.22)
        if getattr(settings, "bg_music_volume", 0) > 0
        else 0.22,
        bg_music_ducking=True,
    )

    results = []
    total_target_shorts = max(1, num_videos)

    for idx, best_video in enumerate(best_videos):
        if len(results) >= total_target_shorts:
            break

        remaining_shorts = total_target_shorts - len(results)
        remaining_videos = len(best_videos) - idx
        clips_needed = max(1, int(math.ceil(remaining_shorts / remaining_videos)))
        clips_needed = min(clips_needed, remaining_shorts)

        base_pct = 10 + int(85 * (len(results) / total_target_shorts))
        step_pct = max(1, int(85 / total_target_shorts))

        def _p(offset: float) -> int:
            return min(98, int(base_pct + (step_pct * offset)))

        yield (
            f"Processing video {idx + 1}/{len(best_videos)}: '{best_video.title}' (Extracting up to {clips_needed} short(s))...",
            _p(0.05),
            None,
        )

        download_dir = Path(tempfile.gettempdir()) / f"autopilot_{uuid.uuid4().hex}"
        download_dir.mkdir(parents=True, exist_ok=True)

        # Mode 2: If user explicitly forced Full AI Generation from topic
        if production_strategy == "ai_gen":
            for ai_i in range(clips_needed):
                if len(results) >= total_target_shorts:
                    break
                short_num = len(results) + 1
                yield (
                    f"[{short_num}/{total_target_shorts}] Strategy: Full AI Short Generation ({best_video.title[:30]} - Angle {ai_i + 1})...",
                    _p(0.2),
                    None,
                )
                try:
                    final_output, seo = build_full_ai_short(
                        topic_title=f"{best_video.title} (Part {ai_i + 1})" if clips_needed > 1 else best_video.title,
                        topic_context=f"Video Title: {best_video.title}\nDescription: {best_video.description}\nClip {ai_i + 1} of {clips_needed}",
                        settings=ap_settings,
                        tmp_dir=download_dir,
                        output_dir=ap_settings.output_dir,
                        report_cb=lambda msg: None,
                    )
                    record_processed_video(
                        video_id=f"{best_video.video_id}_ai_{ai_i + 1}",
                        url=best_video.url,
                        title=best_video.title,
                        output_dir=ap_settings.output_dir,
                        mode="ai_gen",
                        seo=seo,
                        output_file=final_output,
                    )
                    slot = get_optimal_schedule_slot(slot_index=len(results))
                    results.append(
                        {
                            "seo": seo,
                            "path": final_output,
                            "publish_at": slot.utc_datetime,
                            "schedule_display": slot.display_str,
                            "local_publish_time": slot.local_datetime,
                        }
                    )
                except (RuntimeError, OSError, ValueError, KeyError, APIError) as exc:
                    logger.error("Full AI Short generation failed for %s: %s", best_video.title, exc)
                    yield (
                        f"[{short_num}/{total_target_shorts}] Generation failed for '{best_video.title[:35]}': {exc}",
                        _p(0.9),
                        None,
                    )
            continue

        yield (
            f"[Video {idx + 1}/{len(best_videos)}] Pre-download triage: fetching YouTube transcript...",
            _p(0.1),
            None,
        )

        # 1. Attempt pre-download transcript triage
        transcript = None
        if not Path(best_video.url).is_file():
            try:
                transcript = fetch_youtube_transcript(best_video.url)
                if transcript:
                    logger.info(
                        "Retrieved %d transcript segments via YouTube API for %s — skipping full video download.",
                        len(transcript),
                        best_video.url,
                    )
            except (
                YouTubeTranscriptError,
                RuntimeError,
                OSError,
                ValueError,
                KeyError,
                AttributeError,
            ) as exc:
                logger.debug(
                    "YouTube transcript fetch failed for %s: %s", best_video.url, exc
                )
                transcript = None

        # Auto-detect content niche if template is custom or unconfigured
        if getattr(ap_settings, "niche_template", "custom") in ("custom", "", "auto"):
            try:
                from services.niche_templates import auto_detect_niche, get_template
                det_niche = auto_detect_niche(
                    title=best_video.title,
                    description=getattr(best_video, "description", ""),
                    text_sample=" ".join(s.text for s in transcript[:10]) if transcript else "",
                )
                if det_niche != "custom":
                    det_tpl = get_template(det_niche)
                    ap_settings = det_tpl.apply_to(ap_settings)
                    yield (
                        f"[Video {idx + 1}/{len(best_videos)}] Auto-detected Niche: {det_tpl.sidebar_label}",
                        _p(0.18),
                        None,
                    )
            except (ImportError, RuntimeError, ValueError) as n_err:
                logger.debug("Auto-detect niche skipped: %s", n_err)

        candidate_clips: list[ClipCandidate] = []
        cached_full_video: Path | None = None

        if transcript:
            yield (
                f"[Video {idx + 1}/{len(best_videos)}] AI analyzing transcript to select top {clips_needed} viral moments from '{best_video.title[:30]}'...",
                _p(0.25),
                None,
            )
            total_dur = (
                (transcript[-1].end - transcript[0].start) if transcript else 0.0
            )
            eff_min = (
                min(ap_settings.clip_min_duration, max(15.0, total_dur * 0.9))
                if total_dur > 0
                else ap_settings.clip_min_duration
            )
            raw_clips = select_clips(
                segments=transcript,
                gemini_api_key=settings.gemini_api_key,
                max_clips=clips_needed,
                min_clips=min(1, clips_needed),
                min_dur=eff_min,
                max_dur=ap_settings.clip_max_duration,
                source_title=best_video.title,
                channel_niche=getattr(ap_settings, "niche_template", ""),
            )
            for c_raw in raw_clips[:clips_needed]:
                try:
                    snapped_s, snapped_e = snap_to_silence(
                        start_time=c_raw.start_time,
                        end_time=c_raw.end_time,
                        segments=transcript,
                        min_dur=ap_settings.clip_min_duration,
                        max_dur=ap_settings.clip_max_duration,
                    )
                    candidate_clips.append(
                        ClipCandidate(
                            index=c_raw.index,
                            start_time=snapped_s,
                            end_time=snapped_e,
                            hook_summary=c_raw.hook_summary,
                            seo=c_raw.seo,
                            broll_query=c_raw.broll_query,
                        )
                    )
                except (RuntimeError, ValueError, KeyError) as snap_exc:
                    logger.debug("Boundary snapping skipped in autopilot: %s", snap_exc)
                    candidate_clips.append(c_raw)
        else:
            # Fallback path: Full download + Whisper local transcription + OCR
            yield (
                f"[Video {idx + 1}/{len(best_videos)}] Downloading video for Whisper analysis...",
                _p(0.15),
                None,
            )
            if Path(best_video.url).is_file():
                video_path = Path(best_video.url)
            else:
                try:
                    video_path = download_video(
                        best_video.url, download_dir, settings.max_source_duration_seconds
                    )
                except (RuntimeError, OSError, ValueError) as dl_exc:
                    logger.warning(
                        "Video download failed for %s (%s). Attempting Full AI Short fallback.",
                        best_video.url,
                        dl_exc,
                    )
                    video_path = None

            if video_path is None:
                yield (
                    f"[Video {idx + 1}/{len(best_videos)}] Video download unavailable -> Switching to Full AI Short Generation...",
                    _p(0.2),
                    None,
                )
                for ai_i in range(clips_needed):
                    if len(results) >= total_target_shorts:
                        break
                    short_num = len(results) + 1
                    try:
                        final_output, seo = build_full_ai_short(
                            topic_title=f"{best_video.title} (Part {ai_i + 1})" if clips_needed > 1 else best_video.title,
                            topic_context=f"Video Title: {best_video.title}\nDescription: {best_video.description}",
                            settings=ap_settings,
                            tmp_dir=download_dir,
                            output_dir=ap_settings.output_dir,
                            report_cb=lambda msg: None,
                        )
                        record_processed_video(
                            video_id=f"{best_video.video_id}_ai_{ai_i + 1}",
                            url=best_video.url,
                            title=best_video.title,
                            output_dir=ap_settings.output_dir,
                            mode="ai_gen",
                            seo=seo,
                            output_file=final_output,
                        )
                        slot = get_optimal_schedule_slot(slot_index=len(results))
                        results.append(
                            {
                                "seo": seo,
                                "path": final_output,
                                "publish_at": slot.utc_datetime,
                                "schedule_display": slot.display_str,
                                "local_publish_time": slot.local_datetime,
                            }
                        )
                    except Exception as ai_exc:
                        logger.error("Full AI fallback failed for %s: %s", best_video.title, ai_exc)
                continue

            cached_full_video = video_path
            yield (
                f"[Video {idx + 1}/{len(best_videos)}] Transcribing audio with Whisper...",
                _p(0.35),
                None,
            )
            transcript = transcribe(
                video_path=video_path,
                model_size=settings.whisper_model_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                beam_size=settings.whisper_beam_size,
                context_hint=settings.whisper_context_hint,
            )

            # Extract visual context via OCR
            ocr_text = ""
            try:
                from services.ocr_engine import OCREngine

                ocr_engine = OCREngine(gemini_api_key=settings.gemini_api_key)
                ocr_text = ocr_engine.extract_text_from_video(
                    video_path, sample_rate_sec=5
                )
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning("OCR extraction failed for %s: %s", video_path, exc)

            yield (
                f"[Video {idx + 1}/{len(best_videos)}] AI analyzing transcription to select top {clips_needed} viral moments...",
                _p(0.55),
                None,
            )
            total_dur = (
                (transcript[-1].end - transcript[0].start) if transcript else 0.0
            )
            eff_min = (
                min(ap_settings.clip_min_duration, max(15.0, total_dur * 0.9))
                if total_dur > 0
                else ap_settings.clip_min_duration
            )
            raw_clips = select_clips(
                segments=transcript,
                gemini_api_key=settings.gemini_api_key,
                max_clips=clips_needed,
                min_clips=min(1, clips_needed),
                min_dur=eff_min,
                max_dur=ap_settings.clip_max_duration,
                ocr_text=ocr_text,
                source_title=best_video.title,
                channel_niche=getattr(ap_settings, "niche_template", ""),
            )
            for c_raw in raw_clips[:clips_needed]:
                try:
                    snapped_s, snapped_e = snap_to_silence(
                        start_time=c_raw.start_time,
                        end_time=c_raw.end_time,
                        segments=transcript,
                        min_dur=ap_settings.clip_min_duration,
                        max_dur=ap_settings.clip_max_duration,
                    )
                    candidate_clips.append(
                        ClipCandidate(
                            index=c_raw.index,
                            start_time=snapped_s,
                            end_time=snapped_e,
                            hook_summary=c_raw.hook_summary,
                            seo=c_raw.seo,
                            broll_query=c_raw.broll_query,
                        )
                    )
                except (RuntimeError, ValueError, KeyError) as snap_exc:
                    logger.debug("Boundary snapping skipped in autopilot fallback: %s", snap_exc)
                    candidate_clips.append(c_raw)

        if not candidate_clips:
            logger.warning(
                "AI could not find viral moments in transcript for %s. Skipping.",
                best_video.title,
            )
            continue

        # If hybrid, conduct wide & deep research once per video topic
        dossier = None
        if production_strategy == "hybrid":
            try:
                from services.research_engine import conduct_wide_and_deep_research

                dossier = conduct_wide_and_deep_research(
                    topic_title=best_video.title,
                    topic_context=f"Video Title: {best_video.title}\nDescription: {best_video.description}",
                    gemini_api_key=settings.gemini_api_key,
                )
            except (OSError, RuntimeError, ValueError, KeyError) as r_exc:
                logger.warning("Autopilot research step skipped: %s", r_exc)

        # Loop through all selected viral moments for this video
        for clip_pos, best_clip in enumerate(candidate_clips):
            if len(results) >= total_target_shorts:
                break

            short_num = len(results) + 1
            yield (
                f"[{short_num}/{total_target_shorts}] Selected Moment #{best_clip.index}: {best_clip.seo.title if best_clip.seo else best_clip.hook_summary} [{best_clip.start_display} → {best_clip.end_display}]",
                _p(0.60),
                None,
            )

            # Sourcing video media for this clip
            source_is_section = False
            source_offset = 0.0
            video_path = None

            if Path(best_video.url).is_file():
                video_path = Path(best_video.url)
                source_is_section = False
                source_offset = 0.0
            elif cached_full_video:
                video_path = cached_full_video
                source_is_section = False
                source_offset = 0.0
            else:
                # Fast section download: download only this chosen clip segment (with 2s padding)
                pad = 2.0
                clip_dl_dir = download_dir / f"clip_{best_clip.index}"
                clip_dl_dir.mkdir(parents=True, exist_ok=True)
                try:
                    video_path = download_video_section(
                        best_video.url,
                        clip_dl_dir,
                        start_time=best_clip.start_time,
                        end_time=best_clip.end_time,
                        padding=pad,
                    )
                    source_is_section = True
                    source_offset = min(best_clip.start_time, pad)
                except Exception as exc:
                    logger.warning(
                        "Section download failed (%s), falling back to full download.", exc
                    )
                    try:
                        cached_full_video = download_video(
                            best_video.url,
                            download_dir,
                            settings.max_source_duration_seconds,
                        )
                        video_path = cached_full_video
                        source_is_section = False
                        source_offset = 0.0
                    except Exception as dl_exc:
                        logger.warning(
                            "Full download also failed for %s: %s", best_video.url, dl_exc
                        )
                        video_path = None

            if not video_path:
                logger.error(
                    "No valid video media available for clip %d of '%s'. Skipping.",
                    best_clip.index,
                    best_video.title,
                )
                continue

            active_strategy = production_strategy
            if active_strategy == "auto":
                try:
                    from services.content_decision_engine import classify_production_mode

                    detected_mode = classify_production_mode(
                        video_path, user_preference="auto"
                    )
                    active_strategy = detected_mode
                except Exception as c_err:
                    logger.debug("Mode classification skipped: %s", c_err)
                    active_strategy = "speaker"

            if active_strategy == "ai_gen":
                try:
                    final_output, seo = build_full_ai_short(
                        topic_title=best_clip.seo.title if (best_clip.seo and best_clip.seo.title) else best_video.title,
                        topic_context=f"Video Title: {best_video.title}\nHook: {best_clip.hook_summary}",
                        settings=ap_settings,
                        tmp_dir=download_dir,
                        output_dir=ap_settings.output_dir,
                        report_cb=lambda msg: None,
                    )
                    record_processed_video(
                        video_id=f"{best_video.video_id}_c{best_clip.index}",
                        url=best_video.url,
                        title=best_video.title,
                        output_dir=ap_settings.output_dir,
                        mode="ai_gen",
                        seo=seo,
                        output_file=final_output,
                    )
                    slot = get_optimal_schedule_slot(slot_index=len(results))
                    results.append(
                        {
                            "seo": seo,
                            "path": final_output,
                            "publish_at": slot.utc_datetime,
                            "schedule_display": slot.display_str,
                            "local_publish_time": slot.local_datetime,
                        }
                    )
                    continue
                except Exception as ai_exc:
                    logger.error("AI gen for clip failed: %s", ai_exc)
                    continue

            if active_strategy == "hybrid":
                yield (
                    f"[{short_num}/{total_target_shorts}] Strategy: Back-and-Forth Hybrid Short (Speaker Clip + Deep Research AI Breakdown)...",
                    _p(0.68),
                    None,
                )
                try:
                    from services.hybrid_short_generator import build_hybrid_short
                    from services.timeline_utils import slice_segments
                    from services.video_engine import slice_video
                except ImportError:
                    from shorts_engine.services.hybrid_short_generator import (
                        build_hybrid_short,
                    )
                    from shorts_engine.services.timeline_utils import slice_segments
                    from shorts_engine.services.video_engine import slice_video

                clip_dur = max(6.0, best_clip.end_time - best_clip.start_time)
                part1_dur = min(20.0, clip_dur)
                s_start = source_offset if source_is_section else best_clip.start_time
                s_end = s_start + part1_dur

                part1_raw = download_dir / f"hybrid_part1_raw_{best_clip.index}_{idx}.mp4"
                slice_video(
                    source_path=video_path,
                    start_time=s_start,
                    end_time=s_end,
                    output_path=part1_raw,
                )

                part1_segments = slice_segments(
                    transcript, best_clip.start_time, best_clip.start_time + part1_dur
                )

                try:
                    final_output, seo = build_hybrid_short(
                        clip_video_path=part1_raw,
                        speaker_segments=part1_segments,
                        topic_title=best_clip.seo.title if (best_clip.seo and best_clip.seo.title) else best_video.title,
                        topic_context=f"Video Title: {best_video.title}\nHook: {best_clip.hook_summary}\nExcerpt: {best_clip.seo.title if best_clip.seo else ''}",
                        settings=ap_settings,
                        tmp_dir=download_dir,
                        output_dir=ap_settings.output_dir,
                        report_cb=lambda msg: None,
                        research_dossier=dossier,
                    )
                    record_processed_video(
                        video_id=f"{best_video.video_id}_c{best_clip.index}",
                        url=best_video.url,
                        title=best_video.title,
                        output_dir=ap_settings.output_dir,
                        mode="hybrid",
                        seo=seo,
                        output_file=final_output,
                    )
                    slot = get_optimal_schedule_slot(slot_index=len(results))
                    results.append(
                        {
                            "seo": seo,
                            "path": final_output,
                            "publish_at": slot.utc_datetime,
                            "schedule_display": slot.display_str,
                            "local_publish_time": slot.local_datetime,
                        }
                    )

                    try:
                        from services.project_memory import ProjectRecord, save_project_record
                        p_rec = ProjectRecord(
                            project_id=f"proj_{best_video.video_id}_{best_clip.index}_{int(datetime.now(timezone.utc).timestamp())}",
                            created_at=datetime.now(timezone.utc).isoformat(),
                            source_title=best_video.title,
                            source_url=best_video.url,
                            duration_seconds=clip_dur,
                            niche=ap_settings.niche_template,
                            video_type=getattr(best_video, "channel_title", "video"),
                            caption_style=getattr(ap_settings, "caption_style", "auto"),
                            hook_summary=best_clip.hook_summary,
                            hook_text=seo.title if seo else "",
                            virality_score=best_clip.virality_score or 8.5,
                            tags=seo.tags if seo else [],
                            status="approved",
                            output_path=str(final_output),
                        )
                        save_project_record(p_rec)
                    except Exception as p_err:
                        logger.debug("Project memory saving skipped: %s", p_err)
                    continue
                except Exception as exc:
                    logger.error(
                        "Hybrid generation failed for clip %d of %s: %s",
                        best_clip.index,
                        best_video.title,
                        exc,
                    )
                    continue

            # Standard / Speaker Short compose
            yield (
                f"[{short_num}/{total_target_shorts}] Assembling Short for Clip #{best_clip.index}...",
                _p(0.75),
                None,
            )
            stem_pfx = f"v{idx + 1}" if len(best_videos) > 1 else None
            result = process_url_clip(
                clip=best_clip,
                source_path=video_path,
                all_segments=transcript,
                settings=ap_settings,
                tmp_dir=download_dir,
                run_output_dir=ap_settings.output_dir,
                custom_broll_path=broll_path if broll_path else None,
                source_is_section=source_is_section,
                source_offset=source_offset,
                source_video_id=f"{best_video.video_id}_c{best_clip.index}",
                stem_prefix=stem_pfx,
            )

            if result.error or not result.output_file:
                logger.error(
                    f"Clip processing failed for video {best_video.title} clip {best_clip.index}: {result.error}"
                )
                continue

            # SEO
            yield (
                f"[{short_num}/{total_target_shorts}] Generating SEO Metadata for Clip #{best_clip.index}...",
                _p(0.9),
                None,
            )
            try:
                from services.timeline_utils import slice_segments
                from services.transcriber import full_transcript_text
            except ImportError:
                from shorts_engine.services.timeline_utils import slice_segments
                from shorts_engine.services.transcriber import full_transcript_text

            sub_segments = slice_segments(
                transcript, best_clip.start_time, best_clip.end_time
            )
            sub_transcript_text = full_transcript_text(sub_segments)
            seo = best_clip.seo if (best_clip.seo and best_clip.seo.title) else generate_seo(
                transcript_text=sub_transcript_text, api_key=settings.gemini_api_key
            )

            record_processed_video(
                video_id=f"{best_video.video_id}_c{best_clip.index}",
                url=best_video.url,
                title=best_video.title,
                output_dir=ap_settings.output_dir,
                mode="clip",
                seo=seo,
                output_file=result.output_file,
            )

            try:
                from services.project_memory import ProjectRecord, save_project_record
                clip_dur = max(1.0, best_clip.end_time - best_clip.start_time)
                p_rec = ProjectRecord(
                    project_id=f"proj_{best_video.video_id}_{best_clip.index}_{int(datetime.now(timezone.utc).timestamp())}",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    source_title=best_video.title,
                    source_url=best_video.url,
                    duration_seconds=clip_dur,
                    niche=ap_settings.niche_template,
                    video_type=getattr(best_video, "channel_title", "video"),
                    caption_style=getattr(ap_settings, "caption_style", "auto"),
                    hook_summary=best_clip.hook_summary,
                    hook_text=seo.title if seo else "",
                    virality_score=best_clip.virality_score or 8.5,
                    tags=seo.tags if seo else [],
                    status="approved",
                    output_path=str(result.output_file),
                )
                save_project_record(p_rec)
            except Exception as p_err:
                logger.debug("Project memory saving skipped: %s", p_err)

            slot = get_optimal_schedule_slot(slot_index=len(results))
            results.append(
                {
                    "seo": seo,
                    "path": result.output_file,
                    "publish_at": slot.utc_datetime,
                    "schedule_display": slot.display_str,
                    "local_publish_time": slot.local_datetime,
                }
            )

    if not results:
        raise RuntimeError("Failed to generate any videos successfully.")

    # Review
    yield ("Ready for manual review!", 100, results)
