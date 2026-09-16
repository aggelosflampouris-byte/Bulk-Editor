"""
services/autopilot.py — End-to-End Autopilot Orchestration
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Any

from config import Settings
from services.channel_analyzer import fetch_youtube_videos, fetch_view_velocity_top
from services.downloader import download_video
from services.transcriber import transcribe
import tempfile
from services.clip_selector import select_clips
from pipeline import process_url_clip
from services.seo_generator import generate_seo

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
    broll_path: str = ""
) -> Generator[tuple[str, int, Any], None, None]:
    """
    Runs the entire pipeline end-to-end.
    Yields (status_message, progress_percentage, result_data)
    """
    yield ("Analyzing channel / niche...", 5, None)
    
    videos = fetch_youtube_videos(target_url, max_videos=30)
    if not videos:
        raise ValueError("Could not find any videos on this channel.")
        
    velocity_picks = fetch_view_velocity_top(videos, top_n=5)
    if not velocity_picks:
        raise ValueError("Could not calculate velocity for videos.")
        
    best_video = velocity_picks[0]
    
    yield (f"Selected highly viral video: {best_video.title}", 15, None)
    
    # Download
    yield ("Downloading video for analysis...", 20, None)
    import uuid
    download_dir = Path(tempfile.gettempdir()) / f"autopilot_{uuid.uuid4().hex}"
    download_dir.mkdir(parents=True, exist_ok=True)
    video_path = download_video(best_video.url, download_dir, settings.max_source_duration_seconds)
    
    # Transcribe
    yield ("Transcribing audio...", 35, None)
    transcript = transcribe(
        video_path=video_path,
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        beam_size=settings.whisper_beam_size,
        context_hint=settings.whisper_context_hint
    )
    
    # Select Clips
    yield ("AI analyzing transcription for viral clips...", 50, None)
    clips = select_clips(
        segments=transcript,
        gemini_api_key=settings.gemini_api_key,
        max_clips=3,
        min_clips=3,
        min_dur=30.0,
        max_dur=60.0
    )
    if not clips:
        raise ValueError("AI could not find any good clips in this video.")
        
    # Pick highest ranked clip
    best_clip = clips[0]
    yield (f"Selected clip: {best_clip.seo.title} (Rank: {best_clip.index})", 60, None)
    
    # Compose Clip
    yield ("Downloading clip segment & assembling Short...", 70, None)
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
        raise RuntimeError(f"Clip processing failed: {result.error}")
        
    # SEO
    yield ("Generating SEO Metadata...", 85, None)
    # crop transcript to clip bounds
    from services.timeline_utils import slice_segments
    from services.transcriber import full_transcript_text
    sub_segments = slice_segments(transcript, best_clip.start_time, best_clip.end_time)
    sub_transcript_text = full_transcript_text(sub_segments)
    seo = generate_seo(
        transcript_text=sub_transcript_text,
        api_key=settings.gemini_api_key
    )
    
    # Review
    yield ("Ready for manual review!", 100, {
        "seo": seo,
        "path": result.output_path,
        "publish_at": get_optimal_schedule_time()
    })
