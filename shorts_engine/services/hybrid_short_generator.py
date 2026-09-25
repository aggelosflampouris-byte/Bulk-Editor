"""
services/hybrid_short_generator.py — Hybrid Short Production Engine.

Binds and mixes authentic video clips sourced from @DianismaNews
with AI-generated visual B-roll scenes and complementary Greek voiceover
in a dynamic, high-retention BACK-AND-FORTH conversational structure:
  - Beat 1: AI Opening Hook (3–5s) — Teases the conflict and frames the speaker's statement.
  - Beat 2: Authentic Speaker Bite 1 (6–10s) — Human speaker delivers core statement.
  - Beat 3: AI Fact-Check & Deep Research Commentary (8–14s) — Cites verified numbers & data.
  - Beat 4: Authentic Speaker Bite 2 (5–8s, if multi-part) or AI Closing Outro (4–6s).
  - Beat 5: AI Closing Outro & Community CTA (4–6s) — Verdict & comment trigger.
  - Background: Unified royalty-free instrumental music bed with dynamic vocal ducking.
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
    from services.logic_guardrail import evaluate_logical_coherence
    from services.research_engine import (
        ResearchDossier,
        conduct_wide_and_deep_research,
    )
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
        apply_pacing_pattern_interrupts,
        burn_subtitles,
        crop_to_9_16,
        mix_background_music,
        probe_duration,
        probe_resolution,
        slice_video,
    )
except ImportError:
    from shorts_engine.services.ai_short_generator import (
        _DEFAULT_VOICE,
        _assemble_scene_video,
        clean_json_markdown,
        synthesize_voiceover_with_segments,
    )
    from shorts_engine.services.face_tracker import track_active_speaker
    from shorts_engine.services.logic_guardrail import evaluate_logical_coherence
    from shorts_engine.services.research_engine import (
        ResearchDossier,
        conduct_wide_and_deep_research,
    )
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
        apply_pacing_pattern_interrupts,
        burn_subtitles,
        crop_to_9_16,
        mix_background_music,
        probe_duration,
        probe_resolution,
        slice_video,
    )

logger = logging.getLogger(__name__)

_HYBRID_SCRIPT_PROMPT = """\
You are an elite YouTube Shorts producer and investigative strategist for @DianismaNews.
An on-camera human speaker delivers the following statement in a YouTube Short:
"{speaker_text}"

ORIGINAL VIDEO CONTEXT:
Title: {topic_title}
Context: {topic_context}

FACT-CHECKING & DEEP RESEARCH DOSSIER:
{research_context}

YOUR TASK:
Create a dynamic, journalistic BACK-AND-FORTH dialogue between the AI Journalist Voice and the Speaker Clip in natural, punchy Greek, strictly adhering to the 3-ACT VIRAL EXPLAINER SCRIPT BLUEPRINT:

ACT 1: PROVOCATIVE PREMISE (Opening Hook, 3–5s, 10–18 words in spoken Greek):
- A razor-sharp teaser framing the controversy or scandal, challenging the premise, and hyping the speaker's statement (e.g. "Αυτή η δήλωση στη Βουλή για τα οικονομικά άναψε φωτιές. Δείτε τι υποστήριξε ο...").

ACT 2: VERIFIED DATA REALITY & CONTRASTING FACTS (Commentary Breakdown, 8–14s, 25–40 words in spoken Greek):
- An investigative fact-checking breakdown that steps in directly after the speaker, citing concrete numbers, percentages, budget sums, or official records from the research dossier (e.g. "Όμως τα επίσημα στοιχεία δείχνουν κάτι εντελώς διαφορετικό: [συγκεκριμένοι αριθμοί/στοιχεία]...").

ACT 3: COMMENT-DRIVING POLARIZING QUESTION (Closing Outro, 4–6s, 10–18 words in spoken Greek):
- A polarizing closing verdict and community debate trigger designed to maximize comment volume (e.g. "Εσείς πιστεύετε τα λόγια ή τα επίσημα νούμερα; Γράψτε μας στα σχόλια και κάντε εγγραφή στο @DianismaNews!").

4. "scenes": 3 sequential 9:16 visual scenes with English Pexels video search queries matching each AI beat.
5. "seo": High-CTR metadata.

Return ONLY a valid JSON object matching this schema:
{{
  "title": "string",
  "hook": "string",
  "opening_hook": "string",
  "commentary_script": "string",
  "outro_script": "string",
  "narration_script": "string",
  "scenes": [
    {{
      "scene_index": 1,
      "beat": "hook",
      "narration_chunk": "string",
      "visual_prompt": "string",
      "pexels_query": "string"
    }},
    {{
      "scene_index": 2,
      "beat": "commentary",
      "narration_chunk": "string",
      "visual_prompt": "string",
      "pexels_query": "string"
    }},
    {{
      "scene_index": 3,
      "beat": "outro",
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
    opening_hook: str = ""
    commentary_script: str = ""
    outro_script: str = ""
    research_dossier: ResearchDossier | None = None


def generate_hybrid_script(
    speaker_text: str,
    topic_title: str,
    topic_context: str,
    gemini_api_key: str,
    research_dossier: ResearchDossier | None = None,
) -> HybridScriptPackage:
    """
    Generate back-and-forth Greek narration beats (Opening Hook, Fact-Check Commentary, Outro),
    visual scenes, and SEO metadata via Gemini using wide and deep research.
    """
    # 1. Conduct wide & deep research if dossier not supplied
    dossier = research_dossier
    if dossier is None:
        try:
            dossier = conduct_wide_and_deep_research(
                topic_title=topic_title,
                topic_context=f"{topic_context}\nSpeaker: {speaker_text[:300]}",
                gemini_api_key=gemini_api_key,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as res_exc:
            logger.warning("[Hybrid Engine] Research step skipped: %s", res_exc)
            dossier = None

    research_text = dossier.to_prompt_context() if dossier else "(Δεν υπάρχουν διαθέσιμα στοιχεία έρευνας)"

    client = genai.Client(api_key=gemini_api_key)
    prompt = _HYBRID_SCRIPT_PROMPT.format(
        speaker_text=speaker_text[:1200] if speaker_text else topic_title,
        topic_title=topic_title or "Επικαιρότητα",
        topic_context=topic_context[:1500] if topic_context else "Ελληνική πολιτική και κοινωνική επικαιρότητα.",
        research_context=research_text,
    )

    logger.info("Calling Gemini for Back-and-Forth Hybrid script grounded in research...")
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

    # Extract distinct back-and-forth beats
    opening_hook = str(data.get("opening_hook") or hook).strip()
    commentary = str(
        data.get("commentary_script")
        or data.get("narration_script")
        or "Αυτά τα νούμερα δείχνουν την πραγματική πίεση στην αγορά."
    ).strip()
    outro = str(data.get("outro_script") or "Εσείς τι πιστεύετε; Γράψτε μας τη γνώμη σας στα σχόλια!").strip()

    full_narration = str(data.get("narration_script") or f"{commentary} {outro}").strip()

    # Apply Logic & "Make Sense" Guardrail
    coherence = evaluate_logical_coherence(
        topic_title=topic_title,
        part1_text=speaker_text,
        part2_text=commentary,
        gemini_api_key=gemini_api_key,
    )
    if coherence.repaired_script:
        logger.info("[Hybrid Engine] Coherence guardrail refined commentary: %s", coherence.repaired_script)
        commentary = coherence.repaired_script
        full_narration = f"{commentary} {outro}"

    raw_scenes = data.get("scenes") or []
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raw_scenes = [
            {
                "scene_index": 1,
                "beat": "hook",
                "narration_chunk": opening_hook,
                "visual_prompt": "Greek parliament news and economic analysis background vertical 9:16",
                "pexels_query": "greece news politics",
            },
            {
                "scene_index": 2,
                "beat": "commentary",
                "narration_chunk": commentary,
                "visual_prompt": "Greek economy finance data market street",
                "pexels_query": "athens finance economy",
            },
            {
                "scene_index": 3,
                "beat": "outro",
                "narration_chunk": outro,
                "visual_prompt": "Greek citizens discussing in city",
                "pexels_query": "greece people street",
            },
        ]

    seo_dict = data.get("seo") or {}
    if not isinstance(seo_dict, dict) or not seo_dict.get("title"):
        seo = SeoMetadata.fallback(full_narration)
    else:
        try:
            seo = _validate_seo_dict(seo_dict)
        except (ValueError, KeyError):
            seo = SeoMetadata.fallback(full_narration)

    return HybridScriptPackage(
        title=title,
        hook=hook,
        narration_script=full_narration,
        scenes=raw_scenes,
        seo=seo,
        opening_hook=opening_hook,
        commentary_script=commentary,
        outro_script=outro,
        research_dossier=dossier,
    )


def _render_ai_subtitled_beat(
    text: str,
    scenes: list[dict[str, Any]],
    beat_name: str,
    uid: str,
    settings: Any,
    tmp_dir: Path,
    target_w: int,
    target_h: int,
    primary_kw: str | None = None,
) -> Path | None:
    """Render a standalone AI voiceover + B-roll + karaoke subtitles beat."""
    clean_text = (text or "").strip()
    if not clean_text:
        return None

    voice_path = tmp_dir / f"hybrid_ai_{beat_name}_{uid}.mp3"
    voice_path, beat_segments = synthesize_voiceover_with_segments(
        text=clean_text,
        output_path=voice_path,
        voice=_DEFAULT_VOICE,
    )
    beat_dur = probe_duration(voice_path)
    if beat_dur <= 0.2:
        return None

    beat_visual_bed = _assemble_scene_video(
        scenes=scenes if scenes else [{"pexels_query": "greece news politics"}],
        total_audio_duration=beat_dur,
        pexels_api_key=settings.pexels_api_key,
        tmp_dir=tmp_dir,
        target_width=target_w,
        target_height=target_h,
    )

    beat_muxed = tmp_dir / f"hybrid_ai_{beat_name}_muxed_{uid}.mp4"
    cmd_mux = [
        "ffmpeg", "-y",
        "-i", str(beat_visual_bed),
        "-i", str(voice_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(beat_muxed),
    ]
    subprocess.run(cmd_mux, capture_output=True, check=False)
    if not beat_muxed.is_file():
        shutil.copy2(str(beat_visual_bed), str(beat_muxed))

    subtitled_beat = beat_muxed
    if beat_segments:
        ass_beat = tmp_dir / f"hybrid_ai_{beat_name}_{uid}.ass"
        write_ass_file(
            segments=beat_segments,
            output_path=ass_beat,
            primary_keyword=primary_kw,
            subtitle_position=getattr(settings, "subtitle_position", "lower_third"),
            subtitle_mode=getattr(settings, "subtitle_mode", "dynamic"),
        )
        burned_beat = tmp_dir / f"hybrid_ai_{beat_name}_subtitled_{uid}.mp4"
        burn_subtitles(input_path=beat_muxed, ass_path=ass_beat, output_path=burned_beat)
        if burned_beat.is_file():
            subtitled_beat = burned_beat

    return subtitled_beat


def build_hybrid_short(
    clip_video_path: Path,
    speaker_segments: list[TranscriptionSegment],
    topic_title: str,
    topic_context: str,
    settings: Any,
    tmp_dir: Path,
    output_dir: Path,
    report_cb: Any | None = None,
    research_dossier: ResearchDossier | None = None,
) -> tuple[Path, SeoMetadata]:
    """
    Build a dynamic Back-and-Forth Hybrid Short binding authentic human speaker footage
    with AI-generated research breakdowns in an interleaved rhythm:
      - Beat 1: AI Opening Hook (3–5s) with B-Roll & Karaoke Captions.
      - Beat 2: Authentic Speaker Bite 1.
      - Beat 3: AI Fact-Check & Research Commentary (8–14s).
      - Beat 4: Authentic Speaker Bite 2 (if duration >= 14s) or AI Outro (4–6s).
      - Beat 5: AI Outro & Call to Action (if 2 speaker bites).
      - Final: Seamlessly concatenated and mixed with an enhanced royalty-free instrumental music bed.
    """
    def _rpt(msg: str) -> None:
        logger.info("[Hybrid Engine] %s", msg)
        if report_cb:
            report_cb(msg)

    uid = uuid.uuid4().hex[:8]
    _rpt("Preparing authentic speaker footage (9:16 vertical crop & subtitle masking)...")

    # 1. Probe & Crop Speaker footage to 9:16 vertical
    src_w, src_h = probe_resolution(clip_video_path)
    target_w = getattr(settings, "target_width", 1080)
    target_h = getattr(settings, "target_height", 1920)

    cropped_speaker = tmp_dir / f"hybrid_spk_cropped_{uid}.mp4"
    if src_w != target_w or src_h != target_h:
        tracking = track_active_speaker(clip_video_path, src_w, src_h, target_w, target_h)
        crop_x = getattr(tracking, "static_crop_x", max(0, (src_w - target_w) // 2))
        crop_x_expr = getattr(tracking, "crop_expression", None)
        crop_y = getattr(tracking, "static_crop_y", None)
        crop_y_expr = getattr(tracking, "crop_y_expression", None)
        crop_to_9_16(
            clip_video_path,
            cropped_speaker,
            target_width=target_w,
            target_height=target_h,
            crop_x_offset=crop_x,
            crop_x_expr=crop_x_expr,
            crop_y_offset=crop_y,
            crop_y_expr=crop_y_expr,
        )
    else:
        shutil.copy2(str(clip_video_path), str(cropped_speaker))

    # Mask old burned-in subtitles on speaker footage
    masked_speaker = cropped_speaker
    if getattr(settings, "mask_old_subtitles", True):
        _rpt("Masking old lower-third burned captions on speaker clip...")
        masked_dest = tmp_dir / f"hybrid_spk_masked_{uid}.mp4"
        masked_speaker = mask_burned_in_subtitles(cropped_speaker, masked_dest)

    total_spk_dur = probe_duration(masked_speaker)
    rebased_segs = slice_segments(speaker_segments, 0.0, total_spk_dur) if speaker_segments else []
    speaker_text = " ".join(s.text for s in rebased_segs) if rebased_segs else topic_title

    # 2. Generate Back-and-Forth Script with Wide & Deep Research
    _rpt("Conducting wide & deep research and generating back-and-forth script...")
    pkg = generate_hybrid_script(
        speaker_text=speaker_text,
        topic_title=topic_title,
        topic_context=topic_context,
        gemini_api_key=settings.gemini_api_key,
        research_dossier=research_dossier,
    )

    # Filter visual scenes by beat
    hook_scenes = [s for s in pkg.scenes if s.get("beat") == "hook"] or pkg.scenes[:1]
    commentary_scenes = [s for s in pkg.scenes if s.get("beat") in ("commentary", "commentary_1", "commentary_2")] or pkg.scenes[1:3] or pkg.scenes
    outro_scenes = [s for s in pkg.scenes if s.get("beat") == "outro"] or pkg.scenes[-1:]

    # 3. Determine Speaker Bite Splitting for Back-and-Forth Rhythm
    # If speaker footage is >= 14s and has multiple segments, split into 2 bites
    can_split_speaker = total_spk_dur >= 14.0 and len(rebased_segs) >= 2
    timeline_parts: list[Path] = []

    # Render Beat 1: AI Opening Hook
    _rpt("Rendering Beat 1: AI Opening Hook...")
    beat1_path = _render_ai_subtitled_beat(
        text=pkg.opening_hook or pkg.hook,
        scenes=hook_scenes,
        beat_name="beat1_hook",
        uid=uid,
        settings=settings,
        tmp_dir=tmp_dir,
        target_w=target_w,
        target_h=target_h,
        primary_kw=pkg.seo.primary_keyword,
    )
    if beat1_path and beat1_path.is_file():
        timeline_parts.append(beat1_path)

    if can_split_speaker:
        # Find split boundary near middle of speaker clip
        mid_target = total_spk_dur / 2.0
        best_split_time = mid_target
        min_diff = 999.0
        for seg in rebased_segs[:-1]:
            diff = abs(seg.end - mid_target)
            if diff < min_diff and 4.0 <= seg.end <= total_spk_dur - 4.0:
                min_diff = diff
                best_split_time = seg.end

        # Slice Bite 1
        b1_raw = tmp_dir / f"hybrid_spk_b1_{uid}.mp4"
        slice_video(masked_speaker, start_time=0.0, end_time=best_split_time, output_path=b1_raw)
        if best_split_time > 3.5:
            b1_paced = tmp_dir / f"hybrid_spk_b1_paced_{uid}.mp4"
            apply_pacing_pattern_interrupts(b1_raw, b1_paced, cut_interval=3.5, zoom_factor=1.12, target_width=target_w, target_height=target_h)
            b1_raw = b1_paced
        b1_segs = slice_segments(rebased_segs, 0.0, best_split_time)
        b1_sub = b1_raw
        if b1_segs:
            ass_b1 = tmp_dir / f"hybrid_spk_b1_{uid}.ass"
            write_ass_file(b1_segs, ass_b1, primary_keyword=None, subtitle_position=getattr(settings, "subtitle_position", "lower_third"), subtitle_mode="dynamic")
            b1_sub_dest = tmp_dir / f"hybrid_spk_b1_sub_{uid}.mp4"
            burn_subtitles(b1_raw, ass_b1, b1_sub_dest)
            if b1_sub_dest.is_file():
                b1_sub = b1_sub_dest
        timeline_parts.append(b1_sub)

        # Render Beat 3: AI Fact-Check Commentary
        _rpt("Rendering Beat 3: AI Deep Research Fact-Check...")
        beat3_path = _render_ai_subtitled_beat(
            text=pkg.commentary_script or pkg.narration_script,
            scenes=commentary_scenes,
            beat_name="beat3_commentary",
            uid=uid,
            settings=settings,
            tmp_dir=tmp_dir,
            target_w=target_w,
            target_h=target_h,
            primary_kw=pkg.seo.primary_keyword,
        )
        if beat3_path and beat3_path.is_file():
            timeline_parts.append(beat3_path)

        # Slice Bite 2
        b2_raw = tmp_dir / f"hybrid_spk_b2_{uid}.mp4"
        slice_video(masked_speaker, start_time=best_split_time, end_time=total_spk_dur, output_path=b2_raw)
        if (total_spk_dur - best_split_time) > 3.5:
            b2_paced = tmp_dir / f"hybrid_spk_b2_paced_{uid}.mp4"
            apply_pacing_pattern_interrupts(b2_raw, b2_paced, cut_interval=3.5, zoom_factor=1.12, target_width=target_w, target_height=target_h)
            b2_raw = b2_paced
        b2_segs_raw = slice_segments(rebased_segs, best_split_time, total_spk_dur)
        # Rebase timestamps starting at 0.0
        b2_segs = [
            TranscriptionSegment(
                start=max(0.0, round(s.start - best_split_time, 3)),
                end=max(0.05, round(s.end - best_split_time, 3)),
                text=s.text,
                words=[
                    (max(0.0, round(w[0] - best_split_time, 3)), max(0.05, round(w[1] - best_split_time, 3)), w[2])
                    for w in (s.words or [])
                ] if s.words else None,
            )
            for s in b2_segs_raw
        ]
        b2_sub = b2_raw
        if b2_segs:
            ass_b2 = tmp_dir / f"hybrid_spk_b2_{uid}.ass"
            write_ass_file(b2_segs, ass_b2, primary_keyword=None, subtitle_position=getattr(settings, "subtitle_position", "lower_third"), subtitle_mode="dynamic")
            b2_sub_dest = tmp_dir / f"hybrid_spk_b2_sub_{uid}.mp4"
            burn_subtitles(b2_raw, ass_b2, b2_sub_dest)
            if b2_sub_dest.is_file():
                b2_sub = b2_sub_dest
        timeline_parts.append(b2_sub)

        # Render Beat 5: AI Outro & CTA
        _rpt("Rendering Beat 5: AI Outro & Call to Action...")
        beat5_path = _render_ai_subtitled_beat(
            text=pkg.outro_script,
            scenes=outro_scenes,
            beat_name="beat5_outro",
            uid=uid,
            settings=settings,
            tmp_dir=tmp_dir,
            target_w=target_w,
            target_h=target_h,
            primary_kw=pkg.seo.primary_keyword,
        )
        if beat5_path and beat5_path.is_file():
            timeline_parts.append(beat5_path)

    else:
        # Single Speaker Bite structure: [Beat 1: Hook] -> [Beat 2: Speaker] -> [Beat 3: Commentary] -> [Beat 4: Outro]
        speaker_for_sub = masked_speaker
        if total_spk_dur > 3.5:
            spk_paced = tmp_dir / f"hybrid_spk_full_paced_{uid}.mp4"
            apply_pacing_pattern_interrupts(masked_speaker, spk_paced, cut_interval=3.5, zoom_factor=1.12, target_width=target_w, target_height=target_h)
            speaker_for_sub = spk_paced
        spk_sub = speaker_for_sub
        if rebased_segs:
            ass_spk = tmp_dir / f"hybrid_spk_full_{uid}.ass"
            write_ass_file(rebased_segs, ass_spk, primary_keyword=None, subtitle_position=getattr(settings, "subtitle_position", "lower_third"), subtitle_mode="dynamic")
            spk_sub_dest = tmp_dir / f"hybrid_spk_full_sub_{uid}.mp4"
            burn_subtitles(speaker_for_sub, ass_spk, spk_sub_dest)
            if spk_sub_dest.is_file():
                spk_sub = spk_sub_dest
        timeline_parts.append(spk_sub)

        # Render Beat 3: AI Commentary
        _rpt("Rendering Beat 3: AI Fact-Check Commentary...")
        beat3_path = _render_ai_subtitled_beat(
            text=pkg.commentary_script or pkg.narration_script,
            scenes=commentary_scenes,
            beat_name="beat3_commentary",
            uid=uid,
            settings=settings,
            tmp_dir=tmp_dir,
            target_w=target_w,
            target_h=target_h,
            primary_kw=pkg.seo.primary_keyword,
        )
        if beat3_path and beat3_path.is_file():
            timeline_parts.append(beat3_path)

        # Render Beat 4: AI Outro & CTA
        _rpt("Rendering Beat 4: AI Outro & Call to Action...")
        beat4_path = _render_ai_subtitled_beat(
            text=pkg.outro_script,
            scenes=outro_scenes,
            beat_name="beat4_outro",
            uid=uid,
            settings=settings,
            tmp_dir=tmp_dir,
            target_w=target_w,
            target_h=target_h,
            primary_kw=pkg.seo.primary_keyword,
        )
        if beat4_path and beat4_path.is_file():
            timeline_parts.append(beat4_path)

    # 4. Concat all timeline beats seamlessly
    _rpt(f"Binding {len(timeline_parts)} alternating beats into unified back-and-forth Short...")
    combined_path = tmp_dir / f"hybrid_combined_{uid}.mp4"

    cmd_concat_inputs: list[str] = ["ffmpeg", "-y"]
    for part in timeline_parts:
        cmd_concat_inputs.extend(["-i", str(part)])

    # Construct standardized scaling & audio formatting filter
    filter_lines: list[str] = []
    for i in range(len(timeline_parts)):
        filter_lines.append(f"[{i}:v]scale={target_w}:{target_h},setsar=1,fps=30[v{i}];")
        filter_lines.append(f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a{i}];")

    concat_labels = "".join(f"[v{i}][a{i}]" for i in range(len(timeline_parts)))
    filter_lines.append(f"{concat_labels}concat=n={len(timeline_parts)}:v=1:a=1[vcat][acat]")

    cmd_concat = (
        cmd_concat_inputs
        + [
            "-filter_complex",
            "".join(filter_lines),
            "-map", "[vcat]",
            "-map", "[acat]",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-c:a", "aac",
            "-b:a", "192k",
            str(combined_path),
        ]
    )

    res_concat = subprocess.run(cmd_concat, capture_output=True, check=False)
    if res_concat.returncode != 0 or not combined_path.is_file():
        logger.error("Hybrid back-and-forth concat failed: %s", res_concat.stderr[:300] if res_concat.stderr else "unknown error")
        combined_path = timeline_parts[0] if timeline_parts else masked_speaker

    current_path = combined_path

    # 5. Layer unified royalty-free instrumental background music bed with dynamic ducking
    bg_music_path = settings.resolve_bg_music_path() if getattr(settings, "enable_bg_music", True) else None
    if bg_music_path is not None:
        _rpt("Layering authentic instrumental background music bed across Short...")
        bg_dest = tmp_dir / f"hybrid_bgm_{uid}.mp4"
        try:
            current_path = mix_background_music(
                video_path=current_path,
                music_path=bg_music_path,
                output_path=bg_dest,
                volume=getattr(settings, "bg_music_volume", 0.22),
                ducking=getattr(settings, "bg_music_ducking", True),
            )
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning("BGM mixing skipped for hybrid Short: %s", exc)

    # 6. Save final output
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output = output_dir / f"hybrid_short_{uid}.mp4"
    shutil.copy2(str(current_path), str(final_output))

    _rpt(f"Back-and-Forth Hybrid Short successfully produced: {final_output.name}")
    return final_output, pkg.seo
