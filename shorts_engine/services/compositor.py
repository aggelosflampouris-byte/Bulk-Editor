"""
services/compositor.py — Multi-layer generative compositor using MoviePy.

Responsibilities:
  1. Build a timeline from multiple layers (Main video, B-Roll, VFX overlay, Text).
  2. Implement Ken Burns effects for still images.
  3. Perform dynamic audio ducking (background music drops when speech is active).
"""

import logging
from pathlib import Path

import cv2
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
)
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut, MultiplyVolume
from moviepy.video.fx import FadeIn, FadeOut, Resize

logger = logging.getLogger(__name__)

def compute_zoom_intervals(
    duration: float,
    segments: list | None = None,
    clip_start_offset: float = 0.0,
    chunk_target: float = 3.5,
) -> list[tuple[float, float]]:
    """
    Compute alternating punch-in zoom intervals (3.0s - 4.5s) for high-retention jump cuts.

    Ensures that every short has energetic, visible camera punch-ins during speech,
    even when Whisper returns only a single coarse segment or when segments are absent.

    Returns:
        List of (start_sec, end_sec) tuples relative to the clip start (0.0 to duration).
    """
    rel_speech: list[tuple[float, float]] = []
    offset = clip_start_offset
    # If segments appear already relative (e.g. all starts <= duration and offset > duration),
    # do not subtract offset again.
    if segments and clip_start_offset > duration:
        all_small = all(
            getattr(s, "start", s[0] if isinstance(s, (list, tuple)) else 0.0) <= (duration + 1.0)
            for s in segments
        )
        if all_small:
            offset = 0.0

    if segments:
        for s in segments:
            start = getattr(s, "start", s[0] if isinstance(s, (list, tuple)) else 0.0)
            end = getattr(s, "end", s[1] if isinstance(s, (list, tuple)) else duration)
            r_s = max(0.0, start - offset)
            r_e = min(duration, end - offset)
            if r_e > r_s + 0.5:
                rel_speech.append((r_s, r_e))

    all_blocks: list[tuple[float, float]] = []
    if not rel_speech:
        # Fallback: divide entire duration into alternating pacing blocks
        t = 0.0
        while t < duration:
            all_blocks.append((t, min(t + chunk_target, duration)))
            t += chunk_target
    else:
        for s_start, s_end in rel_speech:
            seg_len = s_end - s_start
            if seg_len > 4.5:
                t = s_start
                while t < s_end:
                    all_blocks.append((t, min(t + chunk_target, s_end)))
                    t += chunk_target
            else:
                all_blocks.append((s_start, s_end))

    # Zoom is active on odd-indexed blocks (every second chunk punches in)
    zoom_intervals: list[tuple[float, float]] = []
    for i, (b_s, b_e) in enumerate(all_blocks):
        if i % 2 != 0 and (b_e - b_s) >= 0.8:
            zoom_intervals.append((round(b_s, 2), round(b_e, 2)))

    return zoom_intervals


def apply_dynamic_zoom_ffmpeg(
    video_path: Path,
    output_path: Path,
    zoom_intervals: list[tuple[float, float]],
    target_width: int = 1080,
    target_height: int = 1920,
    zoom_factor: float = 1.15,
) -> Path:
    """
    Apply speech dynamic zoom jump-cuts natively and rapidly via FFmpeg overlay timeline expressions.
    """
    import shutil
    import subprocess

    if not zoom_intervals:
        # No intervals: copy directly
        shutil.copy2(str(video_path), str(output_path))
        return output_path

    # Build between condition for FFmpeg overlay timeline enable
    cond_expr = "+".join(f"between(t,{zs:.2f},{ze:.2f})" for zs, ze in zoom_intervals)
    fc = (
        f"[0:v]split=2[base][to_zoom];"
        f"[to_zoom]crop=iw/{zoom_factor:.3f}:ih/{zoom_factor:.3f}:(iw-out_w)/2:(ih-out_h)/2,"
        f"scale={target_width}:{target_height}:flags=bicubic[zoomed];"
        f"[base]scale={target_width}:{target_height}:flags=bicubic[basescaled];"
        f"[basescaled][zoomed]overlay=0:0:enable='{cond_expr}'[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-filter_complex", fc,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "copy",
        str(output_path),
    ]

    logger.info("Executing native FFmpeg dynamic speech zoom (%d intervals)...", len(zoom_intervals))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.warning("FFmpeg dynamic zoom failed (%s), falling back...", res.stderr)
        raise RuntimeError(f"FFmpeg dynamic zoom failed: {res.stderr}")

    return output_path


def compose_timeline(
    main_video_path: Path,
    output_path: Path,
    broll_video_path: Path | None = None,
    broll_start: float = 0.0,
    broll_duration: float = 3.0,
    target_width: int = 1080,
    target_height: int = 1920,
    segments: list | None = None,
    clip_start_offset: float = 0.0,
) -> Path:
    """
    Compose the final video using a multi-layer NLE approach.
    """
    logger.info("Compositing timeline for %s", main_video_path.name)

    # Fast path: If only dynamic zoom on main speaker is needed (no B-roll)
    if not broll_video_path or not broll_video_path.exists():
        try:
            from services.video_engine import probe_duration
            dur = probe_duration(main_video_path)
            zoom_intervals = compute_zoom_intervals(dur, segments, clip_start_offset)
            return apply_dynamic_zoom_ffmpeg(
                video_path=main_video_path,
                output_path=output_path,
                zoom_intervals=zoom_intervals,
                target_width=target_width,
                target_height=target_height,
                zoom_factor=1.15,
            )
        except Exception as exc:
            logger.error("FFmpeg dynamic zoom failed: %s", exc)
            raise

    # Layer 0: Main Speaker
    main_clip = VideoFileClip(str(main_video_path))
    duration = main_clip.duration
    
    layers = [main_clip]
    
    # Layer 1: B-Roll Video
    broll_clip = None
    if broll_video_path and broll_video_path.exists():
        logger.info("Adding B-Roll layer: %s", broll_video_path.name)
        broll_clip = VideoFileClip(str(broll_video_path))
        broll_clip = Resize(width=target_width).apply(broll_clip)
            
        broll_clip = broll_clip.with_start(broll_start).with_position("center")
        
        # Transitions
        broll_clip = FadeIn(0.3).apply(broll_clip)
        broll_clip = FadeOut(0.3).apply(broll_clip)
        layers.append(broll_clip)
        
    final_video = CompositeVideoClip(layers, size=(target_width, target_height))
    
    # Audio compositing (preserve original audio only, background mixed elsewhere)
    audio_layers = []
    if main_clip.audio:
        audio_layers.append(main_clip.audio)
        
    if audio_layers:
        final_audio = CompositeAudioClip(audio_layers)
        final_video = final_video.with_audio(final_audio)
        
    logger.info("Rendering final composite to %s...", output_path.name)
    final_video.write_videofile(
        str(output_path),
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="fast",
        threads=4,
        logger=None 
    )
    
    main_clip.close()
    if broll_clip: broll_clip.close()
        
    return output_path
