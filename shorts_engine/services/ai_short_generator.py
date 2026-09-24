"""
services/ai_short_generator.py — Full AI Short Generation from Topic (Mode 2).

Transforms a channel topic or news headline into an original 9:16 Short:
  1. Gemini creates a high-CTR Greek narration script and scene-by-scene visual prompts.
  2. Edge-TTS synthesizes authoritative Greek voiceover (el-GR-NestorasNeural).
  3. Pexels Video API fetches 9:16 vertical B-roll clips matching the visual prompts.
  4. MoviePy / FFmpeg stitches the video scenes to match speech duration.
  5. Faster-Whisper transcribes speech with word timestamps for dynamic karaoke subtitles.
  6. Subtitles and background music are mixed into the final production 9:16 Short.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types

try:
    from services.broll_fetcher import download_clip, search_broll
    from services.seo_generator import (
        SeoMetadata,
        _call_gemini_with_fallback,
        _validate_seo_dict,
    )
    from services.transcriber import transcribe, write_ass_file
    from services.video_engine import (
        burn_subtitles,
        mix_background_music,
        probe_duration,
    )
except ImportError:
    from shorts_engine.services.broll_fetcher import (
        download_clip,
        search_broll,
    )
    from shorts_engine.services.seo_generator import (
        SeoMetadata,
        _call_gemini_with_fallback,
        _validate_seo_dict,
    )
    from shorts_engine.services.transcriber import transcribe, write_ass_file
    from shorts_engine.services.video_engine import (
        burn_subtitles,
        mix_background_music,
        probe_duration,
    )

logger = logging.getLogger(__name__)

_DEFAULT_VOICE = "el-GR-NestorasNeural"

_AI_SCRIPT_PROMPT = """\
You are an elite YouTube Shorts producer and investigative news strategist for @DianismaNews.
Your task: Create a viral 30–40 second Greek-language YouTube Short from this source topic and context.

SOURCE TOPIC:
Title: {topic_title}
Context / Background: {topic_context}

REQUIREMENTS:
1. "narration_script": A captivating, authoritative, natural spoken Greek script (50–75 words total, ~30–38 seconds).
   - Sentence 1: Hard-hitting viral hook (e.g., "Αυτό που συμβαίνει με τη ΔΕΗ δεν έχει προηγούμενο...")
   - Sentences 2–4: Core facts, numbers, and political/economic context.
   - Final sentence: Strong closing thought or call-to-action ("Εσείς τι πιστεύετε; Γράψτε μας στα σχόλια!").
2. "scenes": An array of 3 to 4 sequential visual scenes. For each scene:
   - "scene_index": integer (1, 2, 3...)
   - "narration_chunk": The exact sentence or portion of the script spoken during this scene.
   - "visual_prompt": A vivid visual description suitable for 9:16 vertical video stock.
   - "pexels_query": A clean 2–4 word English search query for stock video (e.g., "electric power grid", "money inflation bank", "athens greece parliament").
3. "seo":
   - "title": High-CTR Greek title with an emoji (max 65 chars).
   - "description": 2–3 sentence Greek YouTube description with relevant hashtags (#Shorts, #Ελλάδα, #Επικαιρότητα).
   - "tags": Array of 8–12 relevant tags in Greek and English.
   - "primary_keyword": The single most impactful Greek keyword in the script.

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
class AIShortPackage:
    title: str
    hook: str
    narration_script: str
    scenes: list[dict[str, Any]]
    seo: SeoMetadata


def clean_json_markdown(text: str) -> str:
    """Strip markdown code fence blocks from a JSON string."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def generate_script_and_scenes(
    topic_title: str,
    topic_context: str,
    gemini_api_key: str,
) -> AIShortPackage:
    """
    Generate viral Greek narration script, scene-by-scene visual prompts, and SEO metadata via Gemini.
    """
    client = genai.Client(api_key=gemini_api_key)
    prompt = _AI_SCRIPT_PROMPT.format(
        topic_title=topic_title or "Επικαιρότητα",
        topic_context=topic_context[:2500] if topic_context else "Ελληνική πολιτική και κοινωνική επικαιρότητα.",
    )

    logger.info("Calling Gemini to generate AI Short script and visual scenes for '%s'...", topic_title[:50])
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
        logger.error("Failed to parse Gemini AI Short script JSON: %s", exc)
        data = {}

    title = str(data.get("title") or topic_title or "Επικαιρότητα").strip()
    hook = str(data.get("hook") or title).strip()
    narration = str(data.get("narration_script") or topic_context[:300]).strip()
    raw_scenes = data.get("scenes") or []
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raw_scenes = [
            {
                "scene_index": 1,
                "narration_chunk": narration,
                "visual_prompt": "Greek news reporting breaking news background",
                "pexels_query": "news broadcasting studio",
            }
        ]

    raw_seo = data.get("seo") or {}
    try:
        seo = _validate_seo_dict(raw_seo)
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("SEO validation fallback: %s", exc)
        seo = SeoMetadata.fallback(title)

    return AIShortPackage(
        title=title,
        hook=hook,
        narration_script=narration,
        scenes=raw_scenes,
        seo=seo,
    )


async def _async_synthesize_voiceover(text: str, output_path: Path, voice: str = _DEFAULT_VOICE) -> None:
    import edge_tts
    comm = edge_tts.Communicate(text, voice)
    await comm.save(str(output_path))


def synthesize_voiceover(text: str, output_path: Path, voice: str = _DEFAULT_VOICE) -> Path:
    """
    Synthesize natural Greek voiceover audio using edge-tts.
    """
    logger.info("Synthesizing Greek voiceover with voice '%s'...", voice)
    asyncio.run(_async_synthesize_voiceover(text, output_path, voice))
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Edge-TTS synthesis produced empty file at {output_path}")
    return output_path


def _assemble_scene_video(
    scenes: list[dict[str, Any]],
    total_audio_duration: float,
    pexels_api_key: str,
    tmp_dir: Path,
    target_width: int = 1080,
    target_height: int = 1920,
) -> Path:
    """
    Fetch 9:16 Pexels video clips for each scene and concatenate them to match speech duration.
    """
    import subprocess

    scene_clips: list[Path] = []
    num_scenes = max(1, len(scenes))
    per_scene_dur = max(2.5, total_audio_duration / num_scenes)

    for i, scene in enumerate(scenes):
        query = str(scene.get("pexels_query") or "greece news").strip()
        dest_raw = tmp_dir / f"scene_{i+1}_raw.mp4"
        dest_scaled = tmp_dir / f"scene_{i+1}_9x16.mp4"

        downloaded_clip: Path | None = None
        if pexels_api_key:
            clip_meta = search_broll(query, pexels_api_key)
            if not clip_meta:
                # Fallback query
                clip_meta = search_broll("athens greece city", pexels_api_key)

            if clip_meta:
                try:
                    downloaded_clip = download_clip(clip_meta, dest_raw)
                except (RuntimeError, OSError) as exc:
                    logger.warning("Scene %d B-roll download failed: %s", i + 1, exc)
                    downloaded_clip = None

        if downloaded_clip and downloaded_clip.exists():
            # Scale & crop to exact 1080x1920, trim to per_scene_dur
            cmd = [
                "ffmpeg", "-y",
                "-stream_loop", "-1",
                "-i", str(downloaded_clip),
                "-t", f"{per_scene_dur:.2f}",
                "-vf", f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,crop={target_width}:{target_height}",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-an",
                str(dest_scaled),
            ]
        else:
            # Fallback synthetic animated gradient background
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=0x0a192f:s={target_width}x{target_height}:d={per_scene_dur:.2f}",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                str(dest_scaled),
            ]

        res = subprocess.run(cmd, capture_output=True, check=False)
        if res.returncode == 0 and dest_scaled.exists():
            scene_clips.append(dest_scaled)

    if not scene_clips:
        # Ultimate fallback single solid background
        fallback_path = tmp_dir / "fallback_bed.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x0f172a:s={target_width}x{target_height}:d={total_audio_duration:.2f}",
            "-c:v", "libx264",
            str(fallback_path),
        ]
        subprocess.run(cmd, capture_output=True, check=False)
        return fallback_path

    # Concatenate scene clips
    concat_list = tmp_dir / "concat_scenes.txt"
    with concat_list.open("w", encoding="utf-8") as f:
        for p in scene_clips:
            f.write(f"file '{p.resolve()}'\n")

    assembled_bed = tmp_dir / "assembled_scenes_bed.mp4"
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(assembled_bed),
    ]
    subprocess.run(cmd_concat, capture_output=True, check=False)
    return assembled_bed if assembled_bed.exists() else scene_clips[0]


def build_full_ai_short(
    topic_title: str,
    topic_context: str,
    settings: Any,
    tmp_dir: Path,
    output_dir: Path,
    report_cb: Any | None = None,
) -> tuple[Path, SeoMetadata]:
    """
    End-to-end generation of an original 9:16 Short from topic material.

    Returns:
        (final_video_path, seo_metadata)
    """
    def _rpt(msg: str) -> None:
        if report_cb:
            report_cb(msg)
        logger.info(msg)

    _rpt("Generating viral script and 9:16 visual scenes with Gemini...")
    pkg = generate_script_and_scenes(
        topic_title=topic_title,
        topic_context=topic_context,
        gemini_api_key=settings.gemini_api_key,
    )

    _rpt("Synthesizing natural Greek voiceover (Nestoras Neural)...")
    voice_path = tmp_dir / "ai_voiceover.mp3"
    synthesize_voiceover(pkg.narration_script, voice_path)

    speech_dur = probe_duration(voice_path)
    logger.info("Synthesized voiceover duration: %.2fs", speech_dur)

    _rpt("Sourcing 9:16 visual B-roll scenes from Pexels...")
    visual_bed = _assemble_scene_video(
        scenes=pkg.scenes,
        total_audio_duration=speech_dur,
        pexels_api_key=settings.pexels_api_key,
        tmp_dir=tmp_dir,
        target_width=settings.target_width,
        target_height=settings.target_height,
    )

    # Mux visual bed + synthesized speech
    _rpt("Muxing video scenes with voiceover...")
    import subprocess
    muxed_path = tmp_dir / "ai_short_muxed.mp4"
    cmd_mux = [
        "ffmpeg", "-y",
        "-i", str(visual_bed),
        "-i", str(voice_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(muxed_path),
    ]
    subprocess.run(cmd_mux, capture_output=True, check=False)

    # Whisper transcription on synthesized speech
    _rpt("Transcribing speech for dynamic fluid subtitles...")
    segments = transcribe(
        video_path=muxed_path,
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        beam_size=settings.whisper_beam_size,
        source_title=pkg.title,
    )

    # Dynamic subtitle generation
    _rpt("Burning fluid dynamic karaoke subtitles...")
    ass_path = tmp_dir / "ai_subtitles.ass"
    write_ass_file(
        segments=segments,
        output_path=ass_path,
        primary_keyword=pkg.seo.primary_keyword,
        subtitle_position=getattr(settings, "subtitle_position", "lower_third"),
        subtitle_mode=getattr(settings, "subtitle_mode", "dynamic"),
    )

    subtitled_path = tmp_dir / "ai_short_subtitled.mp4"
    burn_subtitles(
        video_path=muxed_path,
        ass_path=ass_path,
        output_path=subtitled_path,
    )

    current_path = subtitled_path

    # Background music
    if settings.enable_bg_music and getattr(settings, "bg_music_track", "none") != "none":
        _rpt("Layering background music bed...")
        bg_path = tmp_dir / "ai_short_bgm.mp4"
        try:
            current_path = mix_background_music(
                video_path=current_path,
                output_path=bg_path,
                track_preset=settings.bg_music_track,
                volume=0.12,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning("BGM mixing skipped: %s", exc)

    import uuid
    final_output = output_dir / f"ai_short_{uuid.uuid4().hex[:8]}.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(current_path), str(final_output))

    _rpt(f"AI Short generation complete: {final_output.name}")
    return final_output, pkg.seo
