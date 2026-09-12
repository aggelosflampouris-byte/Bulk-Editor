"""
services/clip_selector.py — AI-driven clip selection via Gemini.

Responsibilities:
  1. Accept a full list of TranscriptionSegments for a source video.
  2. Format them as a timestamped text block and send to Gemini.
  3. Parse the returned JSON array into typed ClipCandidate objects,
     each carrying its own SeoMetadata and B-roll query.
  4. Provide a deterministic fallback when the AI call fails.

This module has zero FFmpeg or file-system dependencies.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from google import genai
from google.genai import types as genai_types
try:
    from services.seo_generator import (
        SeoMetadata,
        _call_gemini_with_fallback,  # reuse the model-fallback logic
        _validate_seo_dict,
        generate_seo,
    )
    from services.timeline_utils import format_transcript_with_timestamps
except ImportError:
    from shorts_engine.services.seo_generator import (
        SeoMetadata,
        _call_gemini_with_fallback,
        _validate_seo_dict,
        generate_seo,
    )
    from shorts_engine.services.timeline_utils import format_transcript_with_timestamps

if TYPE_CHECKING:
    try:
        from services.transcriber import TranscriptionSegment
    except ImportError:
        from shorts_engine.services.transcriber import TranscriptionSegment

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_MAX_OUTPUT_TOKENS: int = 8192

# ── Prompt ────────────────────────────────────────────────────────────────────

_CLIP_SELECTION_PROMPT = """\
You are a professional editorial editor and YouTube Shorts SEO strategist specialising \
in Greek-language video productions.

Below is a timestamped transcript from a longer source video. Your task is to \
identify the best moments to extract as standalone YouTube Shorts and craft authoritative \
SEO metadata that fits the image of the video.

SOURCE TITLE: {source_title}

CLIP STRUCTURE & HOOK REQUIREMENTS (CRITICAL):
- VIRALITY FIRST: Prioritize segments containing high emotion (laughter, surprise, anger), counter-intuitive statements, strong opinions, or highly relatable analogies over plain informational text.
- 0–3s ENGAGING HOOK: The clip MUST start immediately on a high-impact sentence, \
provocative question, or curiosity gap. \
NEVER start on filler words (e.g. "Εεε", "Λοιπόν", "Και να πούμε", "Ναι", "Όπως είπαμε"), \
throat-clearing, greetings, or dead silence. Cut straight into the core thought.
- 3–35s FAST-PACED BODY: Unbroken narrative thread with continuous value, tension, \
or analysis. Zero rambling or fluff.
- 35–50s PUNCHLINE / RESOLUTION: End cleanly on a conclusive takeaway, punchline, \
or clear resolution. Never cut mid-sentence or mid-thought.

CONSTRAINTS:
- You MUST select AT LEAST {min_clips} clips and at most {max_clips} clips (aim for {max_clips} clips if the video duration allows).
- Each clip's duration (end_time - start_time) MUST be between {min_dur} and \
{max_dur} seconds (inclusive). The ideal sweet spot is 35–50 seconds.
- Each clip must be self-contained: it must have a clear hook, development, \
and resolution/punchline. A viewer who has NOT seen the full video must be \
able to understand and appreciate it.
- Clips must NOT overlap. Spread them across different distinct moments and topics in the video.
- Order clips by estimated virality and engagement potential (best first).

TIMESTAMP & CUTTING RULES:
- "start_time" and "end_time" must be specified as total seconds (e.g. 285.0) or as "MM:SS" strings (e.g. "04:45").
- Align "start_time" to the natural speech pause immediately before the hook begins.
- Align "end_time" to the natural silence pause right after the punchline sentence finishes.
- Never write decimal minutes like 4.45 to mean 4m 45s (4 minutes 45 seconds is 285 seconds or "04:45").

OUTPUT FORMAT:
Respond ONLY with a valid JSON object — no markdown, no code fences, \
no explanation. The object must have exactly one key "clips" whose value \
is an array. Each array element must have exactly these keys:

{{
  "clips": [
    {{
      "start_time": <float total seconds or "MM:SS" string>,
      "end_time":   <float total seconds or "MM:SS" string>,
      "hook_summary": "<1–2 sentence English explanation of why this clip's hook \
and structure will maximize retention and CTR on YouTube Shorts>",
      "seo": {{
        "title": "<Greek title, max 60 chars. MUST BE AN ORIGINAL PHRASE that summarizes the core topic. DO NOT USE DIRECT QUOTES. 1-2 strategic emojis allowed>",
        "alt_titles": ["<Alternative title 1>", "<Alternative title 2>"],
        "primary_keyword": "<1-2 words Greek keyword that represents the core topic, exactly as it might appear in the transcript>",
        "description": "<Greek description, concise max 500 chars, structured with hook in first 2 lines, core value in the rest, CTA and 3-5 trending hashtags at the end, 1-2 strategic emojis allowed>",
        "pinned_comment": "<An engaging, controversial, or question-based Greek comment to pin at the top of the comments section to drive engagement>",
        "tags": ["<tag 1>", ..., "<tag 16>"]
      }},
      "broll_query": "<2–5 word English Pexels search query for stock footage \
matching what the speaker is talking about in THIS clip>"
    }}
  ]
}}

SEO & TITLE RULES (CRITICAL):
- Language: title and description inside "seo" must be in Greek.
- IMAGE & BRAND ALIGNMENT: The title must fit the visual identity of the source video (e.g., serious interview vs engaging vlog) but MUST prioritize high retention and Click-Through Rate (CTR).
- ENGAGING TONE:
  * Write in a highly engaging, relatable, or authoritative tone depending on the content.
  * You may use third-person ("Τι αποκαλύπτουν τα στοιχεία...") or curiosity-driven hooks ("Ο λόγος που...").
  * Avoid cheap clickbait, but ensure the title creates a strong curiosity gap.
- EMOJIS ALLOWED: You MAY use 1 or 2 highly relevant emojis (e.g., 🤯, 🔥, 📈, 🚨) to act as visual pattern interrupts and increase CTR. Do not overuse them.
- NO DIRECT QUOTES (CRITICAL): The title and alt_titles MUST be completely original, punchy phrases that act as a hook or summary. They MUST NEVER be sentences copied from the transcript. Max 60 characters.
- description must follow the 3-part structure (Hook -> Value -> CTA + hashtags like #Shorts #Ελλάδα).
- tags must be a list of 12 to 18 high-performing keywords and search phrases optimized for the YouTube search algorithm (mix of Greek search queries, entities/names mentioned, topic keywords, and TREND-JACKING broad category terms even if not explicitly mentioned). Do NOT include '#' symbol prefix in any tag.
- broll_query must describe a concrete visual scene, not an abstract concept.
- Do NOT output anything outside the JSON object.

TIMESTAMPED TRANSCRIPT:
{transcript}
"""


# ── Public Types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClipCandidate:
    """
    A single AI-selected clip from a source video.

    All time values are in seconds relative to the source video origin.
    """

    index: int             # 1-based position in the ranked list
    start_time: float      # seconds in the source video
    end_time: float        # seconds in the source video
    hook_summary: str      # why this clip will perform well (English)
    seo: SeoMetadata       # per-clip Greek SEO metadata
    broll_query: str       # English Pexels query for this clip

    @property
    def duration(self) -> float:
        """Clip duration in seconds."""
        return self.end_time - self.start_time

    @property
    def duration_display(self) -> str:
        """Human-readable duration string."""
        total = int(self.duration)
        return f"{total // 60}:{total % 60:02d}"

    @property
    def start_display(self) -> str:
        """Human-readable start time string."""
        return _seconds_to_display(self.start_time)

    @property
    def end_display(self) -> str:
        """Human-readable end time string."""
        return _seconds_to_display(self.end_time)


# ── Internal Helpers ───────────────────────────────────────────────────────────


def _seconds_to_display(seconds: float) -> str:
    """Format seconds as MM:SS."""
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def _parse_clips_json(raw_text: str) -> list[dict]:
    """
    Extract and parse the JSON clips array from a Gemini response.

    Strips markdown code fences if present, then parses the JSON object
    and extracts the "clips" array.

    Args:
        raw_text: Raw string from the Gemini API.

    Returns:
        List of raw clip dicts.

    Raises:
        ValueError: If JSON is missing or malformed, or "clips" key absent.
    """
    # Strip markdown code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fenced:
        raw_text = fenced.group(1)

    # Extract the outermost JSON object
    obj_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not obj_match:
        raise ValueError(f"No JSON object found in response: {raw_text[:200]!r}")

    try:
        data = json.loads(obj_match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error: {exc}\nRaw: {raw_text[:400]!r}") from exc

    if "clips" not in data or not isinstance(data["clips"], list):
        raise ValueError(f"Response missing 'clips' array. Keys found: {list(data.keys())}")

    return data["clips"]


def _parse_time_value(val: object) -> float:
    """
    Parse a timestamp representation into seconds.

    Supports float/int, 'MM:SS', 'HH:MM:SS', and 'MM:SS.mmm'.
    """
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val or "").strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
    return float(s)


def _normalize_clip_times(
    raw_start: object,
    raw_end: object,
    min_dur: float,
    max_dur: float,
) -> tuple[float, float]:
    """
    Convert raw start/end values into validated second timestamps.

    Detects if the LLM output timestamps as decimal minutes (e.g. 4.45 meaning
    4m 45s and 5.33 meaning 5m 33s) and converts them to total seconds (285s, 333s).
    """
    start = _parse_time_value(raw_start)
    end = _parse_time_value(raw_end)

    # Detect if decimal minutes were provided instead of seconds (e.g. 12.34 -> 12m 34s)
    # when raw difference is less than min_dur
    if end - start < min_dur:
        s_min = int(start)
        s_sec = round((start - s_min) * 100)
        e_min = int(end)
        e_sec = round((end - e_min) * 100)
        if s_sec < 60 and e_sec < 60:
            cand_s = s_min * 60.0 + s_sec
            cand_e = e_min * 60.0 + e_sec
            cand_dur = cand_e - cand_s
            if cand_e > cand_s and cand_dur >= (min_dur * 0.7):
                start, end = cand_s, cand_e

    return start, end


def _parse_single_clip(raw: dict, index: int, min_dur: float, max_dur: float) -> ClipCandidate | None:
    """
    Parse a single raw clip dict into a ClipCandidate.

    Applies soft validation (warns but doesn't raise on minor issues) to
    keep resilient pipeline behaviour. Returns None if the clip is
    fundamentally malformed (missing required keys, invalid timestamps).

    Args:
        raw:     Dict from the parsed JSON array.
        index:   1-based clip index for the ClipCandidate.
        min_dur: Minimum acceptable duration in seconds.
        max_dur: Maximum acceptable duration in seconds.

    Returns:
        A ClipCandidate instance, or None if the clip must be rejected.
    """
    try:
        start, end = _normalize_clip_times(
            raw.get("start_time"),
            raw.get("end_time"),
            min_dur=min_dur,
            max_dur=max_dur,
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Clip #%d: missing or invalid start_time/end_time — skipping. %s", index, exc)
        return None

    if end <= start:
        logger.warning("Clip #%d: end_time %.2f <= start_time %.2f — skipping.", index, end, start)
        return None

    duration = end - start
    # Soft-clamp instead of rejecting: keeps clips within [min, max] by trimming end
    if duration > max_dur:
        logger.warning("Clip #%d: duration %.1fs > max %.1fs — trimming end.", index, duration, max_dur)
        end = start + max_dur
    elif duration < min_dur:
        logger.warning("Clip #%d: duration %.1fs < min %.1fs — extending end.", index, duration, min_dur)
        end = start + min_dur

    hook_summary: str = str(raw.get("hook_summary") or "").strip() or "No hook summary provided."
    broll_query: str = str(raw.get("broll_query") or "").strip() or "people talking"

    # Parse the nested SEO object
    raw_seo = raw.get("seo") or {}
    if not isinstance(raw_seo, dict):
        raw_seo = {}

    try:
        seo = _validate_seo_dict(raw_seo)
    except ValueError as exc:
        logger.warning("Clip #%d: SEO parse failed (%s) — using fallback SEO.", index, exc)
        seo = SeoMetadata.fallback(hook_summary)

    return ClipCandidate(
        index=index,
        start_time=round(start, 3),
        end_time=round(end, 3),
        hook_summary=hook_summary,
        seo=seo,
        broll_query=broll_query,
    )


def _supplement_clips(
    existing_clips: list[ClipCandidate],
    segments: list[TranscriptionSegment],
    target_count: int,
    min_dur: float,
    max_dur: float,
    api_key: str | None = None,
    source_title: str = "",
) -> list[ClipCandidate]:
    """
    Generate supplementary non-overlapping clips to guarantee minimum clip count.

    Finds gaps in the timeline between existing clips and adds candidate windows
    until target_count is reached or no gaps remain.
    """
    if not segments or len(existing_clips) >= target_count:
        return existing_clips

    source_start = segments[0].start
    source_end = segments[-1].end
    source_duration = max(0.0, source_end - source_start)
    effective_min = min(min_dur, source_duration)
    clip_dur = min(max_dur, max(effective_min, (min_dur + max_dur) / 2.0))

    intervals = sorted(
        [(c.start_time, c.end_time) for c in existing_clips],
        key=lambda x: x[0],
    )

    gaps: list[tuple[float, float]] = []
    curr = source_start
    for s, e in intervals:
        if s - curr >= effective_min:
            gaps.append((curr, s))
        curr = max(curr, e)
    if source_end - curr >= effective_min:
        gaps.append((curr, source_end))

    result = list(existing_clips)
    gap_idx = 0
    while len(result) < target_count and gap_idx < len(gaps):
        gap_s, gap_e = gaps[gap_idx]
        gap_len = gap_e - gap_s
        if gap_len >= effective_min:
            cand_s = gap_s
            cand_e = min(cand_s + clip_dur, gap_e)
            if cand_e - cand_s >= effective_min:
                idx = len(result) + 1
                excerpt_segs = [s for s in segments if s.end >= cand_s and s.start <= cand_e]
                full_excerpt_text = " ".join(s.text.strip() for s in excerpt_segs)
                if api_key and full_excerpt_text:
                    import time
                    time.sleep(2)  # Avoid rate limiting
                    seo = generate_seo(full_excerpt_text, api_key, source_title=source_title)
                else:
                    seo = SeoMetadata.fallback(full_excerpt_text[:200] or f"Clip #{idx}")
                result.append(
                    ClipCandidate(
                        index=idx,
                        start_time=round(cand_s, 3),
                        end_time=round(cand_e, 3),
                        hook_summary="Supplementary candidate from video transcript.",
                        seo=seo,
                        broll_query="greek speaker talking",
                    )
                )
                gaps[gap_idx] = (cand_e, gap_e)
                continue
        gap_idx += 1

    # Secondary fallback: If non-overlapping gaps were not enough, slide windows across the duration
    if len(result) < target_count and source_duration >= effective_min:
        needed = target_count - len(result)
        max_start = max(source_start, source_end - clip_dur)
        stride = (max_start - source_start) / max(needed + 1, 1)
        for k in range(needed):
            cand_s = source_start + (k + 1) * stride
            cand_e = min(cand_s + clip_dur, source_end)
            if cand_e - cand_s < effective_min and source_duration >= effective_min:
                cand_s = max(source_start, cand_e - effective_min)
            idx = len(result) + 1
            excerpt_segs = [s for s in segments if s.end >= cand_s and s.start <= cand_e]
            full_excerpt_text = " ".join(s.text.strip() for s in excerpt_segs)
            if api_key and full_excerpt_text:
                import time
                time.sleep(2)  # Avoid rate limiting
                seo = generate_seo(full_excerpt_text, api_key, source_title=source_title)
            else:
                seo = SeoMetadata.fallback(full_excerpt_text[:200] or f"Clip #{idx}")
            result.append(
                ClipCandidate(
                    index=idx,
                    start_time=round(cand_s, 3),
                    end_time=round(cand_e, 3),
                    hook_summary="Supplementary candidate from video transcript.",
                    seo=seo,
                    broll_query="greek speaker talking",
                )
            )
            if len(result) >= target_count:
                break

    return [
        ClipCandidate(
            index=i + 1,
            start_time=c.start_time,
            end_time=c.end_time,
            hook_summary=c.hook_summary,
            seo=c.seo,
            broll_query=c.broll_query,
        )
        for i, c in enumerate(result)
    ]


def _build_fallback_clips(
    segments: list[TranscriptionSegment],
    max_clips: int,
    min_dur: float,
    max_dur: float,
    min_clips: int = 3,
    api_key: str | None = None,
    source_title: str = "",
) -> list[ClipCandidate]:
    """
    Generate evenly-distributed clips as a fallback when the Gemini call fails.

    Divides the transcript into evenly spaced windows of *target_dur* seconds,
    ensuring at least *min_clips* (minimum 3) are produced when duration permits.

    Args:
        segments: Full transcript segments.
        max_clips: Maximum number of clips to generate.
        min_dur:   Minimum clip duration (seconds).
        max_dur:   Maximum clip duration (seconds).
        min_clips: Minimum number of clips to attempt (at least 3).

    Returns:
        List of ClipCandidate objects.
    """
    if not segments:
        return []

    min_clips = max(3, min_clips)
    max_clips = max(min_clips, max_clips)

    source_end = segments[-1].end
    source_start = segments[0].start
    source_duration = max(0.0, source_end - source_start)
    effective_min = min(min_dur, source_duration)
    target_dur = min(max_dur, max(effective_min, (min_dur + max_dur) / 2.0))

    # Always attempt at least min_clips
    desired_clips = max(min_clips, min(max_clips, int(source_duration // target_dur) if target_dur > 0 else min_clips))
    desired_clips = max(desired_clips, min_clips)

    candidates: list[ClipCandidate] = []
    max_start = max(source_start, source_end - target_dur)
    if desired_clips > 1 and max_start > source_start:
        step = (max_start - source_start) / (desired_clips - 1)
    else:
        step = 0.0

    for i in range(desired_clips):
        clip_start = source_start + i * step
        clip_end = min(clip_start + target_dur, source_end)
        if clip_end - clip_start < effective_min and source_duration >= effective_min:
            clip_start = max(source_start, clip_end - effective_min)

        excerpt_segs = [s for s in segments if s.end >= clip_start and s.start <= clip_end]
        full_excerpt_text = " ".join(s.text.strip() for s in excerpt_segs)
        if api_key and full_excerpt_text:
            import time
            time.sleep(2)  # Avoid rate limiting
            seo = generate_seo(full_excerpt_text, api_key, source_title=source_title)
        else:
            seo = SeoMetadata.fallback(full_excerpt_text[:200] or f"Clip {i + 1}")
        candidates.append(
            ClipCandidate(
                index=i + 1,
                start_time=round(clip_start, 3),
                end_time=round(clip_end, 3),
                hook_summary="Fallback clip — evenly distributed across video.",
                seo=seo,
                broll_query="people talking",
            )
        )

    return candidates


# ── Public API ─────────────────────────────────────────────────────────────────


def select_clips(
    segments: list[TranscriptionSegment],
    gemini_api_key: str,
    max_clips: int = 10,
    min_clips: int = 3,
    min_dur: float = 35.0,
    max_dur: float = 50.0,
    source_title: str = "",
) -> list[ClipCandidate]:
    """
    Use Gemini to select the most viral-worthy clips from a transcript.

    Sends the full timestamped transcript to Gemini with a structured prompt
    requesting a ranked JSON array of clip candidates, each with its own SEO
    metadata and B-roll query. Applies boundary snapping and duration clamping
    before returning.

    Falls back to evenly-distributed clips if the API call fails or returns
    an unparseable response.

    Args:
        segments:        Full source transcript segments.
        gemini_api_key:  Google Gemini API key.
        max_clips:       Maximum number of clips to return (1–10).
        min_clips:       Minimum number of clips to return (default: 3).
        min_dur:         Minimum clip duration in seconds (default: 35.0).
        max_dur:         Maximum clip duration in seconds (default: 50.0).
        source_title:    Original video title, provided to Gemini for context.

    Returns:
        List of ClipCandidate objects, ordered by virality (best first),
        at most *max_clips* items.
    """
    min_clips = max(3, min(10, min_clips))
    max_clips = max(min_clips, min(10, max_clips))

    if not segments:
        return []

    if not gemini_api_key or not gemini_api_key.strip():
        logger.warning("Gemini API key absent — using fallback clip distribution.")
        return _build_fallback_clips(segments, max_clips, min_dur, max_dur, min_clips=min_clips)

    transcript_block = format_transcript_with_timestamps(segments)

    # Memory Optimization: Check cache first
    try:
        from services.cache_manager import load_cache_pickle, save_cache_pickle
    except ImportError:
        from shorts_engine.services.cache_manager import load_cache_pickle, save_cache_pickle

    import hashlib
    cache_hash = hashlib.md5(transcript_block.encode("utf-8")).hexdigest()
    cache_key = f"{source_title}_{cache_hash}_{max_clips}_{min_dur}_{max_dur}"
    cached_candidates = load_cache_pickle("select_clips", cache_key)
    if cached_candidates is not None:
        logger.info("Loaded %d clip candidates from cache.", len(cached_candidates))
        return cached_candidates

    prompt = _CLIP_SELECTION_PROMPT.format(
        source_title=source_title or "Unknown",
        min_clips=min_clips,
        max_clips=max_clips,
        min_dur=int(min_dur),
        max_dur=int(max_dur),
        transcript=transcript_block,
    )

    logger.info(
        "Calling Gemini for clip selection (min=%d, max=%d, dur=[%d-%d]s, transcript_len=%d chars)...",
        min_clips, max_clips, int(min_dur), int(max_dur), len(transcript_block),
    )

    try:
        client = genai.Client(api_key=gemini_api_key)
        raw_text = _call_gemini_with_fallback(
            client=client,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=_MAX_OUTPUT_TOKENS,
                temperature=0.3,  # Deterministic clip selection
                response_mime_type="application/json",
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
    except Exception as exc:
        logger.error("Gemini clip selection API call failed: %s — using fallback.", exc)
        return _build_fallback_clips(
            segments, max_clips, min_dur, max_dur, min_clips=min_clips,
            api_key=gemini_api_key, source_title=source_title
        )

    try:
        raw_clips = _parse_clips_json(raw_text)
    except ValueError as exc:
        logger.error("Clip JSON parse failed: %s — using fallback.", exc)
        return _build_fallback_clips(
            segments, max_clips, min_dur, max_dur, min_clips=min_clips,
            api_key=gemini_api_key, source_title=source_title
        )

    # Parse and validate each clip; skip malformed ones
    candidates: list[ClipCandidate] = []
    for i, raw_clip in enumerate(raw_clips[:max_clips]):
        clip = _parse_single_clip(raw_clip, index=i + 1, min_dur=min_dur, max_dur=max_dur)
        if clip is not None:
            candidates.append(clip)

    if not candidates:
        logger.warning("No valid clips parsed from Gemini response — using fallback.")
        return _build_fallback_clips(segments, max_clips, min_dur, max_dur, min_clips=min_clips)

    # Remove overlapping clips (keep the earlier one in the ranked order)
    deduplicated: list[ClipCandidate] = []
    for clip in candidates:
        if not any(
            clip.start_time < existing.end_time and clip.end_time > existing.start_time
            for existing in deduplicated
        ):
            deduplicated.append(clip)

    # Ensure minimum clip threshold (guaranteed at least 3 clips)
    source_duration = (segments[-1].end - segments[0].start) if segments else 0.0
    effective_min = min(min_dur, source_duration)
    target_min = max(3, min_clips) if source_duration >= effective_min else 1

    # FORCE guarantee of 3 clips minimum even if source duration is tiny
    if len(deduplicated) < target_min:
        logger.info(
            "Only %d clip(s) found after deduplication; generating supplementary candidates to satisfy min_clips=%d",
            len(deduplicated), target_min,
        )
        deduplicated = _supplement_clips(
            existing_clips=deduplicated,
            segments=segments,
            target_count=target_min,
            min_dur=min_dur,
            max_dur=max_dur,
            api_key=gemini_api_key,
            source_title=source_title,
        )

    # Secondary guarantee: if still below target_min, top up directly with fallback clips
    if len(deduplicated) < target_min:
        fallback_clips = _build_fallback_clips(
            segments, max_clips, min_dur, max_dur, min_clips=target_min
        )
        for fb in fallback_clips:
            if len(deduplicated) >= target_min:
                break
            deduplicated.append(
                ClipCandidate(
                    index=len(deduplicated) + 1,
                    start_time=fb.start_time,
                    end_time=fb.end_time,
                    hook_summary=fb.hook_summary,
                    seo=fb.seo,
                    broll_query=fb.broll_query,
                )
            )

    logger.info(
        "Clip selection complete: %d clips selected (satisfies min_clips=%d).",
        len(deduplicated), min_clips,
    )
    final_clips = deduplicated[:max_clips]
    try:
        save_cache_pickle("select_clips", cache_key, final_clips)
    except NameError:
        pass
    return final_clips
