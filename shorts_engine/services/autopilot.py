"""
services/autopilot.py — End-to-End Autopilot Orchestration
"""

import logging
import tempfile
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import Settings
from pipeline import process_url_clip
from services.channel_analyzer import fetch_view_velocity_top, fetch_youtube_videos
from services.clip_selector import select_clips
from services.downloader import download_video
from services.seo_generator import generate_seo
from services.transcriber import transcribe

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
    target_url: str,
    settings: Settings,
    broll_path: str = "",
    num_videos: int = 1,
) -> Generator[tuple[str, int, Any], None, None]:
    """
    Runs the entire pipeline end-to-end for the top `num_videos` videos.
    Yields (status_message, progress_percentage, result_data)
    """
    yield ("Analyzing channel / niche...", 5, None)
    
    videos = fetch_youtube_videos(target_url, max_videos=30)
    if not videos:
        raise ValueError("Could not find any videos on this channel.")
        
    velocity_picks = fetch_view_velocity_top(videos, top_n=max(5, num_videos))
    if not velocity_picks:
        raise ValueError("Could not calculate velocity for videos.")
        
    best_videos = velocity_picks[:num_videos]
    
    yield (f"Selected {len(best_videos)} highly viral videos for processing.", 10, None)
    
    results = []
    
    for idx, best_video in enumerate(best_videos):
        base_pct = 10 + (90 * idx // num_videos)
        step_pct = 90 // num_videos
        
        def _p(offset: float) -> int:
            return int(base_pct + (step_pct * offset))
            
        yield (f"[Video {idx+1}/{num_videos}] Selected highly viral video: {best_video.title}", _p(0.05), None)
        
        # Download
        yield (f"[Video {idx+1}/{num_videos}] Downloading video for analysis...", _p(0.1), None)
        import uuid
        download_dir = Path(tempfile.gettempdir()) / f"autopilot_{uuid.uuid4().hex}"
        download_dir.mkdir(parents=True, exist_ok=True)
        video_path = download_video(best_video.url, download_dir, settings.max_source_duration_seconds)
        
        # Transcribe
        yield (f"[Video {idx+1}/{num_videos}] Transcribing audio...", _p(0.3), None)
        transcript = transcribe(
            video_path=video_path,
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            beam_size=settings.whisper_beam_size,
            context_hint=settings.whisper_context_hint
        )
        
        # Extract OCR Text
        yield (f"[Video {idx+1}/{num_videos}] Extracting visual context (OCR)...", _p(0.4), None)
        ocr_text = ""
        try:
            from services.ocr_engine import OCREngine
            ocr_engine = OCREngine(gemini_api_key=settings.gemini_api_key)
            ocr_text = ocr_engine.extract_text_from_video(video_path, sample_rate_sec=5)
        except Exception as e:
            logger.warning(f"OCR extraction failed for {video_path}: {e}")

        # Select Clips
        yield (f"[Video {idx+1}/{num_videos}] AI analyzing transcription for viral clips...", _p(0.5), None)
        clips = select_clips(
            segments=transcript,
            gemini_api_key=settings.gemini_api_key,
            max_clips=3,
            min_clips=3,
            min_dur=30.0,
            max_dur=60.0,
            ocr_text=ocr_text,
        )
        if not clips:
            logger.warning(f"AI could not find any good clips in video {best_video.title}. Skipping.")
            continue
            
        # Pick highest ranked clip
        best_clip = clips[0]
        yield (f"[Video {idx+1}/{num_videos}] Selected clip: {best_clip.seo.title} (Rank: {best_clip.index})", _p(0.6), None)
        
        # Compose Clip
        yield (f"[Video {idx+1}/{num_videos}] Downloading clip segment & assembling Short...", _p(0.7), None)
        # process_url_clip handles downloading the sub-segment and running video_engine
        result = process_url_clip(
            clip=best_clip,
            source_path=video_path,
            all_segments=transcript,
            settings=settings,
            tmp_dir=download_dir,
            run_output_dir=settings.output_dir,
            custom_broll_path=broll_path if broll_path else None
        )
        
        if result.error:
            logger.error(f"Clip processing failed for video {best_video.title}: {result.error}")
            continue
            
        # SEO
        yield (f"[Video {idx+1}/{num_videos}] Generating SEO Metadata...", _p(0.9), None)
        # crop transcript to clip bounds
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
