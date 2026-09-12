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
    ken_burns: bool = False,
    split_screen: bool = False,
) -> Path:
    """
    Compose the final video using a multi-layer NLE approach.
    """
    logger.info("Compositing timeline for %s", main_video_path.name)

    # Fast path: If only dynamic zoom on main speaker is needed (no B-roll)
    try:
        from services.video_engine import probe_duration
        dur = probe_duration(main_video_path)
        zoom_intervals = compute_zoom_intervals(dur, segments, clip_start_offset)
        
        if not broll_video_path or not broll_video_path.exists():
            return apply_dynamic_zoom_ffmpeg(
                video_path=main_video_path,
                output_path=output_path,
                zoom_intervals=zoom_intervals,
                target_width=target_width,
                target_height=target_height,
                zoom_factor=1.15,
            )
        else:
            # We have B-roll, but we still want punch-ins on the main video
            zoomed_main_path = main_video_path.with_name(f"{main_video_path.stem}_zoomed.mp4")
            apply_dynamic_zoom_ffmpeg(
                video_path=main_video_path,
                output_path=zoomed_main_path,
                zoom_intervals=zoom_intervals,
                target_width=target_width,
                target_height=target_height,
                zoom_factor=1.15,
            )
            main_video_path = zoomed_main_path
    except Exception as exc:
        logger.error("FFmpeg dynamic zoom failed: %s", exc)
        if not broll_video_path or not broll_video_path.exists():
            raise

    # Layer 0: Main Speaker (now zoomed)
    main_clip = VideoFileClip(str(main_video_path))
    duration = main_clip.duration
    
    layers = [main_clip]
    
    # Layer 1: B-Roll Video
    broll_clip = None
    if broll_video_path and broll_video_path.exists():
        logger.info("Adding B-Roll layer: %s (Ken Burns: %s, Split-Screen: %s)", broll_video_path.name, ken_burns, split_screen)
        broll_clip = VideoFileClip(str(broll_video_path))
        
        # Trim to duration first
        broll_clip = broll_clip.with_duration(broll_duration)

        # Split screen logic: limit height to top half if enabled
        b_target_h = target_height // 2 if split_screen else target_height
        
        # Apply Ken Burns if enabled
        if ken_burns:
            # We scale the clip slightly larger to allow zooming
            zoom_start = 1.0
            zoom_end = 1.15
            
            # Helper to calculate dynamic resize based on time
            def make_zoom(t):
                # t goes from 0 to broll_duration
                progress = t / max(broll_clip.duration, 0.1)
                current_zoom = zoom_start + (zoom_end - zoom_start) * progress
                return current_zoom
                
            # Apply dynamic zoom
            broll_clip = broll_clip.transform(
                lambda get_frame, t: cv2.resize(
                    get_frame(t), 
                    dsize=(0,0),
                    fx=make_zoom(t), 
                    fy=make_zoom(t), 
                    interpolation=cv2.INTER_LINEAR
                )
            )

        # Force resize to target width and height via MoviePy Resize
        # Since we might have zoomed, or it might be raw, we apply a hard crop/resize
        from moviepy.video.fx import Crop
        broll_clip = Resize(width=target_width).apply(broll_clip)
        
        # If it's too short vertically after width resize, we resize height instead and crop width
        if broll_clip.h < b_target_h:
            broll_clip = Resize(height=b_target_h).apply(broll_clip)
            
        broll_clip = Crop(
            x_center=broll_clip.w/2, 
            y_center=broll_clip.h/2, 
            width=target_width, 
            height=b_target_h
        ).apply(broll_clip)

        # Positioning
        if split_screen:
            broll_clip = broll_clip.with_start(broll_start).with_position(("center", "top"))
        else:
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
