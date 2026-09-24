"""
services/hybrid_short_generator.py — Hybrid Short Production Engine.

Binds and mixes authentic video clips sourced from @DianismaNews
with AI-generated visual B-roll scenes and complementary Greek voiceover:
  - Part 1 (8–15s): Authentic Speaker Clip (authentic human voice, 9:16 vertical crop,
    masked old burned-in captions, dynamic karaoke subtitles).
  - Part 2 (15–20s): AI Breakdown & Call-to-Action (complementary script analyzing what the
    speaker said, synthetic Greek voiceover, 9:16 vertical Pexels B-roll, dynamic subtitles).
  - Seamlessly bound into a unified 25–35s high-retention Short.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types

try:
    from services.ai_short_generator import (
        _DEFAULT_VOICE,
        _assemble_scene_video,
        clean_json_markdown,
        synthesize_voiceover_with_segments,
    )
    from services.face_tracker import track_active_speaker
    from services.seo_generator import (
        SeoMetadata,
        _call_gemini_with_fallback,
        _validate_seo_dict,
    )
    from services.subtitle_masker import mask_burned_in_subtitles
    from services.timeline_utils import slice_segments
    from services.transcriber import (
        TranscriptionSegment,
        write_ass_file,
    )
    from services.video_engine import (
        burn_subtitles,
        crop_to_9_16,
        mix_background_music,
        probe_duration,
        probe_resolution,
    )
except ImportError:
    from shorts_engine.services.ai_short_generator import (
        _DEFAULT_VOICE,
        _assemble_scene_video,
        clean_json_markdown,
        synthesize_voiceover_with_segments,
    )
    from shorts_engine.services.face_tracker import track_active_speaker
    from shorts_engine.services.seo_generator import (
        SeoMetadata,
        _call_gemini_with_fallback,
        _validate_seo_dict,
    )
    from shorts_engine.services.subtitle_masker import mask_burned_in_subtitles
    from shorts_engine.services.timeline_utils import slice_segments
    from shorts_engine.services.transcriber import (
        TranscriptionSegment,
        write_ass_file,
    )
    from shorts_engine.services.video_engine import (
        burn_subtitles,
        crop_to_9_16,
        mix_background_music,
        probe_duration,
        probe_resolution,
    )

logger = logging.getLogger(__name__)

_HYBRID_SCRIPT_PROMPT = """\
You are an elite YouTube Shorts producer and investigative strategist for @DianismaNews.
An on-camera speaker delivers the following statement in Part 1 (first 10–15s) of a YouTube Short:
"{speaker_text}"

ORIGINAL VIDEO CONTEXT:
Title: {topic_title}
Context: {topic_context}

YOUR TASK:
Write a compelling Part 2 conclusion (12–16 seconds, 30–45 words total) in natural spoken Greek that:
1. Immediately connects with what the speaker said (e.g., "Αυτά τα στοιχεία επιβεβαιώνουν...", "Πίσω από αυτή τη δήλωση κρύβεται...").
2. Explains the underlying political or economic reality with hard facts or numbers.
3. Concludes with a strong call-to-action question ("Εσείς τι πιστεύετε; Γράψτε μας στα σχόλια!").
4. Outlines 2 sequential 9:16 visual scenes with English Pexels video search queries.
5. Generates high-CTR SEO metadata for the complete unified Short.

Return ONLY a valid JSON object matching this schema:
{{
  "title": "string",
  "hook": "string",
  "narration_script": "string",
  "scenes": [
    {{
      "scene_index": 1,
      "narration_chunk": "string",
      "visual_prompt": "string",
      "pexels_query": "string"
    }},
    {{
      "scene_index": 2,
      "narration_chunk": "string",
      "visual_prompt": "string",
      "pexels_query": "string"
    }}
  ],
  "seo": {{
    "title": "string",
    "description": "string",
    "tags": ["string"],
    "primary_keyword": "string"
  }}
}}
"""


@dataclass
class HybridScriptPackage:
    title: str
    hook: str
    narration_script: str
    scenes: list[dict[str, Any]]
    seo: SeoMetadata


def generate_hybrid_script(
    speaker_text: str,
    topic_title: str,
    topic_context: str,
    gemini_api_key: str,
) -> HybridScriptPackage:
    """
    Generate complementary Part 2 Greek narration script, visual scenes, and SEO metadata via Gemini.
    """
    client = genai.Client(api_key=gemini_api_key)
    prompt = _HYBRID_SCRIPT_PROMPT.format(
        speaker_text=speaker_text[:1200] if speaker_text else topic_title,
        topic_title=topic_title or "Επικαιρότητα",
        topic_context=topic_context[:1500] if topic_context else "Ελληνική πολιτική και κοινωνική επικαιρότητα.",
    )

    logger.info("Calling Gemini for Hybrid Part 2 script based on speaker's statement...")
    raw_json = _call_gemini_with_fallback(
        client=client,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.35,
            response_mime_type="application/json",
        ),
    )

    clean_text = clean_json_markdown(raw_json)
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Gemini Hybrid script JSON: %s", exc)
        data = {}

    title = str(data.get("title") or topic_title or "Επικαιρότητα").strip()
    hook = str(data.get("hook") or title).strip()
    narration = str(data.get("narration_script") or "Ποια είναι η γνώμη σας για αυτές τις εξελίξεις; Γράψτε μας στα σχόλια!").strip()
    raw_scenes = data.get("scenes") or []
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raw_scenes = [
            {
                "scene_index": 1,
                "narration_chunk": narration,
                "visual_prompt": "Greek parliament news and economic analysis background vertical 9:16",
                "pexels_query": "news parliament greece",
            }
        ]

    seo_dict = data.get("seo") or {}
    if not isinstance(seo_dict, dict) or not seo_dict.get("title"):
        seo = SeoMetadata.fallback(narration)
    else:
        try:
            seo = _validate_seo_dict(seo_dict)
        except (ValueError, KeyError):
            seo = SeoMetadata.fallback(narration)

    return HybridScriptPackage(
        title=title,
        hook=hook,
        narration_script=narration,
        scenes=raw_scenes,
        seo=seo,
    )


def build_hybrid_short(
    clip_video_path: Path,
    speaker_segments: list[TranscriptionSegment],
    topic_title: str,
    topic_context: str,
    settings: Any,
    tmp_dir: Path,
    output_dir: Path,
    report_cb: Any | None = None,
) -> tuple[Path, SeoMetadata]:
    """
    Build a hybrid Short binding the authentic clipped video with an AI-generated B-roll breakdown:
      - Part 1: Sliced human speaker video (9:16 vertical crop, subtitle masking, karaoke subtitles).
      - Part 2: AI-generated visual breakdown with Greek voiceover and dynamic subtitles.
      - Final: Seamlessly bound and concatenated with background music.
    """
    def _rpt(msg: str) -> None:
        logger.info("[Hybrid Engine] %s", msg)
        if report_cb:
            report_cb(msg)

    uid = uuid.uuid4().hex[:8]
    _rpt("Preparing Part 1: Authentic speaker footage...")

    # 1. Probe & Crop Part 1 to 9:16 vertical
    src_w, src_h = probe_resolution(clip_video_path)
    target_w = getattr(settings, "target_width", 1080)
    target_h = getattr(settings, "target_height", 1920)

    cropped_part1 = tmp_dir / f"hybrid_part1_cropped_{uid}.mp4"
    if src_w != target_w or src_h != target_h:
        tracking = track_active_speaker(clip_video_path, src_w, src_h, target_w, target_h)
        crop_x = getattr(tracking, "static_crop_x", max(0, (src_w - target_w) // 2))
        crop_to_9_16(clip_video_path, cropped_part1, target_width=target_w, target_height=target_h, crop_x_offset=crop_x)
    else:
        shutil.copy2(str(clip_video_path), str(cropped_part1))

    # Mask old subtitles on speaker footage
    masked_part1 = cropped_part1
    if getattr(settings, "mask_old_subtitles", True):
        _rpt("Masking old burned-in lower-third captions on speaker clip...")
        masked_dest = tmp_dir / f"hybrid_part1_masked_{uid}.mp4"
        masked_part1 = mask_burned_in_subtitles(cropped_part1, masked_dest)

    # Burn dynamic karaoke subtitles for Part 1
    speaker_dur = probe_duration(masked_part1)
    rebased_speaker_segs = slice_segments(speaker_segments, 0.0, speaker_dur) if speaker_segments else []
    subtitled_part1 = masked_part1

    if rebased_speaker_segs:
        _rpt("Burning fluid subtitles on authentic speaker clip...")
        ass_part1 = tmp_dir / f"hybrid_part1_{uid}.ass"
        write_ass_file(
            segments=rebased_speaker_segs,
            output_path=ass_part1,
            primary_keyword=None,
            subtitle_position=getattr(settings, "subtitle_position", "lower_third"),
            subtitle_mode=getattr(settings, "subtitle_mode", "dynamic"),
        )
        burned_dest1 = tmp_dir / f"hybrid_part1_subtitled_{uid}.mp4"
        burn_subtitles(input_path=masked_part1, ass_path=ass_part1, output_path=burned_dest1)
        subtitled_part1 = burned_dest1

    # 2. Part 2: Generate complementary AI voiceover + B-roll scenes
    _rpt("Generating Part 2: AI Voiceover & 9:16 B-roll breakdown...")
    speaker_text = " ".join(s.text for s in rebased_speaker_segs) if rebased_speaker_segs else topic_title
    pkg = generate_hybrid_script(
        speaker_text=speaker_text,
        topic_title=topic_title,
        topic_context=topic_context,
        gemini_api_key=settings.gemini_api_key,
    )

    voice_path = tmp_dir / f"hybrid_ai_voice_{uid}.mp3"
    voice_path, ai_segments = synthesize_voiceover_with_segments(
        text=pkg.narration_script,
        output_path=voice_path,
        voice=_DEFAULT_VOICE,
    )
    ai_dur = probe_duration(voice_path)

    _rpt("Sourcing 9:16 B-roll scenes for AI breakdown from Pexels...")
    ai_visual_bed = _assemble_scene_video(
        scenes=pkg.scenes,
        total_audio_duration=ai_dur,
        pexels_api_key=settings.pexels_api_key,
        tmp_dir=tmp_dir,
        target_width=target_w,
        target_height=target_h,
    )

    # Mux AI scenes with Greek voiceover
    ai_muxed = tmp_dir / f"hybrid_ai_muxed_{uid}.mp4"
    cmd_mux = [
        "ffmpeg", "-y",
        "-i", str(ai_visual_bed),
        "-i", str(voice_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(ai_muxed),
    ]
    res_mux = subprocess.run(cmd_mux, capture_output=True, check=False)
    if not ai_muxed.is_file():
        logger.warning("AI muxing failed (%s) — copying visual bed.", res_mux.stderr[:200] if res_mux.stderr else "")
        shutil.copy2(str(ai_visual_bed), str(ai_muxed))

    # Burn subtitles for Part 2
    subtitled_part2 = ai_muxed
    if ai_segments:
        _rpt("Burning fluid dynamic karaoke subtitles on AI breakdown...")
        ass_part2 = tmp_dir / f"hybrid_part2_{uid}.ass"
        write_ass_file(
            segments=ai_segments,
            output_path=ass_part2,
            primary_keyword=pkg.seo.primary_keyword,
            subtitle_position=getattr(settings, "subtitle_position", "lower_third"),
            subtitle_mode=getattr(settings, "subtitle_mode", "dynamic"),
        )
        burned_dest2 = tmp_dir / f"hybrid_part2_subtitled_{uid}.mp4"
        burn_subtitles(input_path=ai_muxed, ass_path=ass_part2, output_path=burned_dest2)
        subtitled_part2 = burned_dest2

    # 3. Concatenate Part 1 (Speaker) + Part 2 (AI Breakdown)
    _rpt("Binding authentic speaker footage with AI breakdown...")
    combined_path = tmp_dir / f"hybrid_combined_{uid}.mp4"

    # Use ffmpeg filter_complex concat with standardized audio/video specs to avoid PTS jumps
    cmd_concat = [
        "ffmpeg", "-y",
        "-i", str(subtitled_part1),
        "-i", str(subtitled_part2),
        "-filter_complex",
        (
            f"[0:v]scale={target_w}:{target_h},setsar=1,fps=30[v0];"
            f"[1:v]scale={target_w}:{target_h},setsar=1,fps=30[v1];"
            f"[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a0];"
            f"[1:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a1];"
            f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[vcat][acat]"
        ),
        "-map", "[vcat]",
        "-map", "[acat]",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-b:a", "192k",
        str(combined_path),
    ]

    res_concat = subprocess.run(cmd_concat, capture_output=True, check=False)
    if res_concat.returncode != 0 or not combined_path.is_file():
        logger.error("Hybrid concat failed: %s", res_concat.stderr[:300] if res_concat.stderr else "unknown error")
        # Fallback to Part 1 if concat failed
        combined_path = subtitled_part1

    current_path = combined_path

    # 4. Optional unified background music bed
    bg_music_path = settings.resolve_bg_music_path() if getattr(settings, "enable_bg_music", False) else None
    if bg_music_path is not None:
        _rpt("Layering background music bed across hybrid Short...")
        bg_dest = tmp_dir / f"hybrid_bgm_{uid}.mp4"
        try:
            current_path = mix_background_music(
                video_path=current_path,
                music_path=bg_music_path,
                output_path=bg_dest,
                volume=getattr(settings, "bg_music_volume", 0.10),
                ducking=getattr(settings, "bg_music_ducking", True),
            )
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning("BGM mixing skipped for hybrid Short: %s", exc)

    # 5. Output artifact
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output = output_dir / f"hybrid_short_{uid}.mp4"
    shutil.copy2(str(current_path), str(final_output))

    _rpt(f"Hybrid Short successfully generated: {final_output.name}")
    return final_output, pkg.seo
