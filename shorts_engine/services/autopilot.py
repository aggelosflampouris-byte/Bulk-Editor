"""
services/autopilot.py — End-to-End Autopilot Orchestration
"""

import logging
import tempfile
from collections.abc import Generator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import Settings
from google.genai.errors import APIError
from pipeline import process_url_clip
from services.cache_manager import is_video_already_processed
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
from services.youtube_transcript_fetcher import fetch_youtube_transcript
from services.youtube_uploader import (
    YouTubeAuthError,
    authenticate,
    fetch_my_recent_videos,
    is_authenticated,
)

try:
    from services.niche_sourcing import find_niche_trend_videos
except ImportError:
    from shorts_engine.services.niche_sourcing import find_niche_trend_videos

logger = logging.getLogger(__name__)


def get_optimal_schedule_time() -> datetime:
    """
    Returns the next optimal time to schedule a Short.
    Defaults to the next available 6:00 PM local time (converted to UTC).
    """
    now = datetime.now(timezone.utc)
    # Convert UTC to a roughly representative local time, or just schedule for UTC 22:00 (which is 6 PM EST)
    # We will pick the next available 22:00 UTC.
    target = now.replace(hour=22, minute=0, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return target


def run_autopilot_pipeline(
    target_url: str = "",
    settings: Settings = None,
    broll_path: str = "",
    num_videos: int = 1,
    selected_video: VideoMeta | None = None,
    production_strategy: str = "auto",
    niche_query: str = "",
) -> Generator[tuple[str, int, Any], None, None]:
    """
    Runs the entire pipeline end-to-end for the top `num_videos` videos.
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
    if selected_video is not None:
        best_videos = [selected_video]
        yield (f"Selected video from Dianisma library: {selected_video.title}", 8, None)
    elif target_url and any(x in target_url.lower() for x in ("watch?v=", "youtu.be/", "/shorts/")):
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
                yield ("Sourcing uploads from Dianisma library via YouTube Data API v3...", 5, None)
                client = authenticate()
                videos = fetch_my_recent_videos(client, max_videos=30)
            except (YouTubeAuthError, RuntimeError, OSError, ValueError, KeyError, AttributeError):
                videos = []

        if not videos:
            yield (f"Analyzing external channel uploads from {target_url}...", 5, None)
            videos = fetch_youtube_videos(target_url, max_videos=30)

        viral_picks = find_viral_recent_videos(videos, max_results=max(10, num_videos * 2))
        candidates = [vp.video for vp in viral_picks] if viral_picks else list(videos)
        unprocessed = [v for v in candidates if not is_video_already_processed(v.url, settings.output_dir if settings else None)]
        best_videos = (unprocessed if unprocessed else candidates)[:num_videos]
    else:
        # Default Autopilot Mode (also active when DIANISMA_CHANNEL_URL or empty URL is passed):
        # Search YouTube for fresh viral videos in our niche (≤ 3 weeks old)
        # Excludes @DianismaNews to avoid re-using our own channel's older content
        yield ("Searching YouTube for fresh viral videos in our niche (≤ 3 weeks old)...", 5, None)

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

        if not candidates:
            # Resilient fallback: search broader niche topics across YouTube (never our own channel)
            yield ("Searching broader niche topics across YouTube (≤ 3 weeks old)...", 7, None)
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

            for fallback_query in DEFAULT_NICHE_QUERIES:
                fallback_videos = fetch_niche_videos_via_ytdlp(
                    query=fallback_query,
                    max_results=max(10, num_videos * 3),
                    max_age_days=21,
                    output_dir=settings.output_dir if settings else None,
                )
                if fallback_videos:
                    candidates = fallback_videos
                    break

        unprocessed = [v for v in candidates if not is_video_already_processed(v.url, settings.output_dir if settings else None)]
        best_videos = (unprocessed if unprocessed else candidates)[:num_videos]

    if not best_videos:
        raise ValueError("Could not find any suitable videos in the niche for Autopilot processing.")

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
        mask_old_subtitles=True,
        enable_bg_music=True,
        bg_music_track=active_bg_track,
        bg_music_volume=getattr(settings, "bg_music_volume", 0.22) if getattr(settings, "bg_music_volume", 0) > 0 else 0.22,
        bg_music_ducking=True,
    )

    results = []

    for idx, best_video in enumerate(best_videos):
        base_pct = 10 + (90 * idx // num_videos)
        step_pct = 90 // num_videos

        def _p(offset: float, b: int = base_pct, s: int = step_pct) -> int:
            return int(b + (s * offset))

        yield (f"[Video {idx+1}/{num_videos}] Sourced viral video: {best_video.title}", _p(0.05), None)

        import uuid
        download_dir = Path(tempfile.gettempdir()) / f"autopilot_{uuid.uuid4().hex}"
        download_dir.mkdir(parents=True, exist_ok=True)

        # Mode 2: If user explicitly forced Full AI Generation from topic
        if production_strategy == "ai_gen":
            yield (
                f"[Video {idx+1}/{num_videos}] Strategy: Full AI Short Generation (AI Script + Greek Voiceover + 9:16 Media)...",
                _p(0.2),
                None,
            )
            from services.ai_short_generator import build_full_ai_short
            try:
                final_output, seo = build_full_ai_short(
                    topic_title=best_video.title,
                    topic_context=f"Video Title: {best_video.title}\nDescription: {best_video.description}",
                    settings=ap_settings,
                    tmp_dir=download_dir,
                    output_dir=ap_settings.output_dir,
                    report_cb=lambda msg: None,
                )
                results.append({
                    "seo": seo,
                    "path": final_output,
                    "publish_at": get_optimal_schedule_time() + timedelta(days=idx),
                })
                continue
            except (RuntimeError, OSError, ValueError, KeyError, APIError) as exc:
                logger.error("Full AI Short generation failed for %s: %s", best_video.title, exc)
                yield (
                    f"[Video {idx+1}/{num_videos}] Generation failed for '{best_video.title[:35]}': {exc}",
                    _p(0.9),
                    None,
                )
                continue

        yield (f"[Video {idx+1}/{num_videos}] Pre-download triage: fetching YouTube transcript...", _p(0.1), None)

        # 1. Attempt pre-download transcript triage
        transcript = None
        try:
            transcript = fetch_youtube_transcript(best_video.url)
            if transcript:
                logger.info(
                    "Retrieved %d transcript segments via YouTube API for %s — skipping full video download.",
                    len(transcript),
                    best_video.url,
                )
        except (RuntimeError, OSError, ValueError) as exc:
            logger.debug("YouTube transcript fetch failed for %s: %s", best_video.url, exc)
            transcript = None

        source_is_section = False
        source_offset = 0.0

        if transcript:
            # AI selects viral clips directly from pre-fetched transcript
            yield (f"[Video {idx+1}/{num_videos}] AI analyzing transcript for viral clips...", _p(0.25), None)
            total_dur = (transcript[-1].end - transcript[0].start) if transcript else 0.0
            eff_min = min(ap_settings.clip_min_duration, max(15.0, total_dur * 0.9)) if total_dur > 0 else ap_settings.clip_min_duration
            clips = select_clips(
                segments=transcript,
                gemini_api_key=settings.gemini_api_key,
                max_clips=3,
                min_clips=1,
                min_dur=eff_min,
                max_dur=ap_settings.clip_max_duration,
            )
            if not clips:
                logger.warning("AI could not find viral moments in transcript for %s. Skipping.", best_video.title)
                continue

            best_clip = clips[0]
            try:
                snapped_s, snapped_e = snap_to_silence(
                    start_time=best_clip.start_time,
                    end_time=best_clip.end_time,
                    segments=transcript,
                    min_dur=ap_settings.clip_min_duration,
                    max_dur=ap_settings.clip_max_duration,
                )
                best_clip = ClipCandidate(
                    index=best_clip.index,
                    start_time=snapped_s,
                    end_time=snapped_e,
                    hook_summary=best_clip.hook_summary,
                    seo=best_clip.seo,
                    broll_query=best_clip.broll_query,
                )
            except (RuntimeError, ValueError, KeyError) as snap_exc:
                logger.debug("Boundary snapping skipped in autopilot: %s", snap_exc)

            yield (
                f"[Video {idx+1}/{num_videos}] Selected clip: {best_clip.seo.title if best_clip.seo else 'Viral Hook'} (Rank: {best_clip.index})",
                _p(0.4),
                None,
            )

            # Fast section download: only download the chosen clip segment (with 2s padding)
            yield (
                f"[Video {idx+1}/{num_videos}] Sourcing video section [{best_clip.start_display} → {best_clip.end_display}]...",
                _p(0.55),
                None,
            )
            pad = 2.0
            try:
                video_path = download_video_section(
                    best_video.url,
                    download_dir,
                    start_time=best_clip.start_time,
                    end_time=best_clip.end_time,
                    padding=pad,
                )
                source_is_section = True
                source_offset = min(best_clip.start_time, pad)
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning("Section download failed (%s), falling back to full download.", exc)
                video_path = download_video(best_video.url, download_dir, settings.max_source_duration_seconds)
                source_is_section = False
                source_offset = 0.0

            # Content Classifier: if auto mode, inspect video for on-camera speaker
            if production_strategy == "auto":
                from services.content_decision_engine import classify_production_mode
                detected_mode = classify_production_mode(video_path, user_preference="auto")
                if detected_mode == "ai_gen":
                    yield (
                        f"[Video {idx+1}/{num_videos}] No clear on-camera speaker -> Switching to Full AI Short Generation...",
                        _p(0.6),
                        None,
                    )
                    from services.ai_short_generator import build_full_ai_short
                    try:
                        final_output, seo = build_full_ai_short(
                            topic_title=best_video.title,
                            topic_context=f"Video Title: {best_video.title}\nDescription: {best_video.description}",
                            settings=ap_settings,
                            tmp_dir=download_dir,
                            output_dir=ap_settings.output_dir,
                            report_cb=lambda msg: None,
                        )
                        results.append({
                            "seo": seo,
                            "path": final_output,
                            "publish_at": get_optimal_schedule_time() + timedelta(days=idx),
                        })
                        continue
                    except (RuntimeError, OSError, ValueError, KeyError, APIError) as exc:
                        logger.error("Full AI fallback failed for %s: %s", best_video.title, exc)
        else:
            # Fallback path: Full download + Whisper local transcription + OCR
            yield (f"[Video {idx+1}/{num_videos}] Downloading video for Whisper analysis...", _p(0.15), None)
            video_path = download_video(best_video.url, download_dir, settings.max_source_duration_seconds)

            # Content Classifier: if auto mode, inspect video for on-camera speaker
            if production_strategy == "auto":
                from services.content_decision_engine import classify_production_mode
                detected_mode = classify_production_mode(video_path, user_preference="auto")
                if detected_mode == "ai_gen":
                    yield (
                        f"[Video {idx+1}/{num_videos}] No clear on-camera speaker -> Switching to Full AI Short Generation...",
                        _p(0.25),
                        None,
                    )
                    from services.ai_short_generator import build_full_ai_short
                    try:
                        final_output, seo = build_full_ai_short(
                            topic_title=best_video.title,
                            topic_context=f"Video Title: {best_video.title}\nDescription: {best_video.description}",
                            settings=ap_settings,
                            tmp_dir=download_dir,
                            output_dir=ap_settings.output_dir,
                            report_cb=lambda msg: None,
                        )
                        results.append({
                            "seo": seo,
                            "path": final_output,
                            "publish_at": get_optimal_schedule_time() + timedelta(days=idx),
                        })
                        continue
                    except (RuntimeError, OSError, ValueError, KeyError, APIError) as exc:
                        logger.error("Full AI fallback failed for %s: %s", best_video.title, exc)

            yield (f"[Video {idx+1}/{num_videos}] Transcribing audio with Whisper...", _p(0.35), None)
            transcript = transcribe(
                video_path=video_path,
                model_size=settings.whisper_model_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                beam_size=settings.whisper_beam_size,
                context_hint=settings.whisper_context_hint,
            )

            # Extract visual context via OCR
            yield (f"[Video {idx+1}/{num_videos}] Extracting visual context (OCR)...", _p(0.45), None)
            ocr_text = ""
            try:
                from services.ocr_engine import OCREngine
                ocr_engine = OCREngine(gemini_api_key=settings.gemini_api_key)
                ocr_text = ocr_engine.extract_text_from_video(video_path, sample_rate_sec=5)
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning("OCR extraction failed for %s: %s", video_path, exc)

            yield (f"[Video {idx+1}/{num_videos}] AI analyzing transcription for viral clips...", _p(0.55), None)
            total_dur = (transcript[-1].end - transcript[0].start) if transcript else 0.0
            eff_min = min(ap_settings.clip_min_duration, max(15.0, total_dur * 0.9)) if total_dur > 0 else ap_settings.clip_min_duration
            clips = select_clips(
                segments=transcript,
                gemini_api_key=settings.gemini_api_key,
                max_clips=3,
                min_clips=1,
                min_dur=eff_min,
                max_dur=ap_settings.clip_max_duration,
                ocr_text=ocr_text,
            )
            if not clips:
                logger.warning("AI could not find any good clips in video %s. Skipping.", best_video.title)
                continue

            best_clip = clips[0]
            try:
                snapped_s, snapped_e = snap_to_silence(
                    start_time=best_clip.start_time,
                    end_time=best_clip.end_time,
                    segments=transcript,
                    min_dur=ap_settings.clip_min_duration,
                    max_dur=ap_settings.clip_max_duration,
                )
                best_clip = ClipCandidate(
                    index=best_clip.index,
                    start_time=snapped_s,
                    end_time=snapped_e,
                    hook_summary=best_clip.hook_summary,
                    seo=best_clip.seo,
                    broll_query=best_clip.broll_query,
                )
            except (RuntimeError, ValueError, KeyError) as snap_exc:
                logger.debug("Boundary snapping skipped in autopilot fallback: %s", snap_exc)

            yield (
                f"[Video {idx+1}/{num_videos}] Selected clip: {best_clip.seo.title if best_clip.seo else 'Viral Hook'} (Rank: {best_clip.index})",
                _p(0.65),
                None,
            )
            source_is_section = False
            source_offset = 0.0

        # Check if production strategy is hybrid
        if production_strategy == "hybrid":
            yield (
                f"[Video {idx+1}/{num_videos}] Strategy: Back-and-Forth Hybrid Short (Speaker Clip + Deep Research AI Breakdown)...",
                _p(0.68),
                None,
            )
            # Conduct wide & deep research across news and data
            dossier = None
            try:
                from services.research_engine import conduct_wide_and_deep_research
                dossier = conduct_wide_and_deep_research(
                    topic_title=best_video.title,
                    topic_context=f"Video Title: {best_video.title}\nDescription: {best_video.description}",
                    gemini_api_key=settings.gemini_api_key,
                )
            except (OSError, RuntimeError, ValueError, KeyError) as r_exc:
                logger.warning("Autopilot research step skipped: %s", r_exc)

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

            part1_raw = download_dir / f"hybrid_part1_raw_{best_clip.index}.mp4"
            slice_video(
                source_path=video_path,
                start_time=s_start,
                end_time=s_end,
                output_path=part1_raw,
            )

            part1_segments = slice_segments(transcript, best_clip.start_time, best_clip.start_time + part1_dur)

            try:
                final_output, seo = build_hybrid_short(
                    clip_video_path=part1_raw,
                    speaker_segments=part1_segments,
                    topic_title=best_video.title,
                    topic_context=f"Video Title: {best_video.title}\nDescription: {best_video.description}",
                    settings=ap_settings,
                    tmp_dir=download_dir,
                    output_dir=ap_settings.output_dir,
                    report_cb=lambda msg: None,
                    research_dossier=dossier,
                )
                results.append({
                    "seo": seo,
                    "path": final_output,
                    "publish_at": get_optimal_schedule_time() + timedelta(days=idx),
                })
                continue
            except (RuntimeError, OSError, ValueError, KeyError, APIError) as exc:
                logger.error("Hybrid generation failed for %s: %s", best_video.title, exc)
                yield (
                    f"[Video {idx+1}/{num_videos}] Hybrid generation failed for '{best_video.title[:35]}': {exc}",
                    _p(0.9),
                    None,
                )
                continue

        # Compose Clip
        yield (f"[Video {idx+1}/{num_videos}] Assembling Short...", _p(0.75), None)
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
            source_video_id=best_video.video_id,
        )
        
        if result.error:
            logger.error(f"Clip processing failed for video {best_video.title}: {result.error}")
            continue
            
        # SEO
        yield (f"[Video {idx+1}/{num_videos}] Generating SEO Metadata...", _p(0.9), None)
        # crop transcript to clip bounds
        try:
            from services.timeline_utils import slice_segments
            from services.transcriber import full_transcript_text
        except ImportError:
            from shorts_engine.services.timeline_utils import slice_segments
            from shorts_engine.services.transcriber import full_transcript_text
        sub_segments = slice_segments(transcript, best_clip.start_time, best_clip.end_time)
        sub_transcript_text = full_transcript_text(sub_segments)
        seo = generate_seo(
            transcript_text=sub_transcript_text,
            api_key=settings.gemini_api_key
        )
        
        results.append({
            "seo": seo,
            "path": result.output_file,
            # Add an offset to publish_at so they aren't scheduled at the exact same time
            "publish_at": get_optimal_schedule_time() + timedelta(days=idx)
        })

    if not results:
        raise RuntimeError("Failed to generate any videos successfully.")

    # Review
    yield ("Ready for manual review!", 100, results)
