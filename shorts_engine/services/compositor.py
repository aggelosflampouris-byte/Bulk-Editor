"""
services/compositor.py — Multi-layer generative compositor using MoviePy.

Responsibilities:
  1. Build a timeline from multiple layers (Main video, B-Roll, VFX overlay, Text).
  2. Implement Ken Burns effects for still images.
  3. Perform dynamic audio ducking (background music drops when speech is active).
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
from moviepy import (
    VideoFileClip, ImageClip, CompositeVideoClip, AudioFileClip, CompositeAudioClip
)
from moviepy.video.fx import FadeIn, FadeOut, Resize
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut, MultiplyVolume

logger = logging.getLogger(__name__)

def apply_ken_burns(clip, duration: float, zoom_factor: float = 1.15):
    """
    Apply a slow zoom-in effect to a clip (typically an image).
    """
    clip = clip.with_duration(duration)
    
    def crop_center(image, scale):
        h, w = image.shape[:2]
        new_w, new_h = int(w / scale), int(h / scale)
        x1, y1 = (w - new_w) // 2, (h - new_h) // 2
        cropped = image[y1:y1+new_h, x1:x1+new_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def image_transform(get_frame, t):
        frame = get_frame(t)
        scale = 1.0 + ((zoom_factor - 1.0) * (t / duration))
        return crop_center(frame, scale)

    return clip.transform(image_transform)

def compute_zoom_intervals(
    duration: float,
    segments: Optional[list] = None,
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
    import subprocess
    import shutil

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


def apply_dynamic_speech_zoom(clip, duration: float, segments: list, clip_start_offset: float = 0.0, zoom_factor: float = 1.15):
    """
    Apply a dynamic zoom-in and zoom-out effect to the main speaker based on speech segments.
    Alternates zoom on every spoken sentence/segment to simulate engaging jump cuts.
    """
    clip = clip.with_duration(duration)
    zoom_intervals = compute_zoom_intervals(duration, segments, clip_start_offset)

    def crop_center(image, scale):
        if scale == 1.0:
            return image
        h, w = image.shape[:2]
        new_w, new_h = int(w / scale), int(h / scale)
        x1, y1 = (w - new_w) // 2, (h - new_h) // 2
        cropped = image[y1:y1+new_h, x1:x1+new_w]
        import cv2
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def image_transform(get_frame, t):
        frame = get_frame(t)
        scale = 1.0
        for (z_start, z_end) in zoom_intervals:
            if z_start <= t <= z_end:
                scale = zoom_factor
                break
        return crop_center(frame, scale)

    return clip.transform(image_transform)


def compose_timeline(
    main_video_path: Path,
    output_path: Path,
    broll_image_path: Optional[Path] = None,
    broll_start: float = 0.0,
    broll_duration: float = 3.0,
    bg_music_path: Optional[Path] = None,
    text_overlay_path: Optional[Path] = None,
    target_width: int = 1080,
    target_height: int = 1920,
    segments: Optional[list] = None,
    clip_start_offset: float = 0.0,
) -> Path:
    """
    Compose the final video using a multi-layer NLE approach.
    """
    logger.info("Compositing timeline for %s", main_video_path.name)

    # Fast path: If only dynamic zoom on main speaker is needed (no B-roll, no custom overlays, no separate bg_music in MoviePy)
    if (
        (not broll_image_path or not broll_image_path.exists())
        and (not text_overlay_path or not text_overlay_path.exists())
        and (not bg_music_path or not bg_music_path.exists())
    ):
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
            logger.warning("Fast FFmpeg dynamic zoom failed (%s), falling back to MoviePy...", exc)

    # Layer 0: Main Speaker
    main_clip = VideoFileClip(str(main_video_path))
    duration = main_clip.duration
    
    if segments:
        logger.info("Applying dynamic speech zoom to main speaker...")
        main_clip = apply_dynamic_speech_zoom(main_clip, duration, segments, clip_start_offset, zoom_factor=1.15)
        
    layers = [main_clip]
    
    # Layer 1: B-Roll (with Ken Burns)
    broll_clip = None
    if broll_image_path and broll_image_path.exists():
        logger.info("Adding B-Roll layer: %s", broll_image_path.name)
        if broll_image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            broll_clip = ImageClip(str(broll_image_path))
            broll_clip = Resize(width=target_width).apply(broll_clip)
            broll_clip = apply_ken_burns(broll_clip, duration=broll_duration, zoom_factor=1.15)
        else:
            broll_clip = VideoFileClip(str(broll_image_path))
            broll_clip = Resize(width=target_width).apply(broll_clip)
            # Center crop height
            # Note: in MoviePy v2 cropping is a bit complex, but simple resizing works if ratio matches.
            # Assuming broll is already somewhat 16:9 or 9:16
            
        broll_clip = broll_clip.with_start(broll_start).with_position("center")
        
        # Transitions
        broll_clip = FadeIn(0.3).apply(broll_clip)
        broll_clip = FadeOut(0.3).apply(broll_clip)
        layers.append(broll_clip)
        
    # Layer 2: Text / Subtitles Overlay
    text_clip = None
    if text_overlay_path and text_overlay_path.exists():
        logger.info("Adding Text layer: %s", text_overlay_path.name)
        text_clip = VideoFileClip(str(text_overlay_path), has_mask=True)
        text_clip = text_clip.with_position("center")
        layers.append(text_clip)
        
    final_video = CompositeVideoClip(layers, size=(target_width, target_height))
    
    # Audio compositing
    audio_layers = []
    if main_clip.audio:
        audio_layers.append(main_clip.audio)
        
    if bg_music_path and bg_music_path.exists():
        logger.info("Adding Background Music layer: %s", bg_music_path.name)
        bg_audio = AudioFileClip(str(bg_music_path))
        
        if bg_audio.duration < duration:
            # Loop music (a bit complex in MoviePy without LoopAudio, so we'll just trim for now or use loop)
            from moviepy.audio.fx import AudioLoop
            bg_audio = AudioLoop(duration=duration).apply(bg_audio)
        else:
            bg_audio = bg_audio.with_duration(duration)
            
        bg_audio = MultiplyVolume(0.1).apply(bg_audio)
        bg_audio = AudioFadeIn(1.0).apply(bg_audio)
        bg_audio = AudioFadeOut(2.0).apply(bg_audio)
        
        audio_layers.append(bg_audio)
        
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
    if text_clip: text_clip.close()
        
    return output_path
