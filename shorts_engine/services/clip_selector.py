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
from typing import TYPE_CHECKING, Optional

from google import genai
from google.genai import types as genai_types

from services.seo_generator import (
    SeoMetadata,
    _call_gemini_with_fallback,  # reuse the model-fallback logic
    _validate_seo_dict,
)
from services.timeline_utils import format_transcript_with_timestamps

if TYPE_CHECKING:
    from services.transcriber import TranscriptionSegment

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_MAX_OUTPUT_TOKENS: int = 8192

# ── Prompt ────────────────────────────────────────────────────────────────────

_CLIP_SELECTION_PROMPT = """\
You are a professional YouTube Shorts editor and SEO strategist specialising \
in Greek-language content.

Below is a timestamped transcript from a longer source video. Your task is to \
identify the best moments to extract as standalone YouTube Shorts.

SOURCE TITLE: {source_title}

CONSTRAINTS:
- Select AT MOST {max_clips} clips.
- Each clip's duration (end_time - start_time) MUST be between {min_dur} and \
{max_dur} seconds (inclusive).
- Each clip must be self-contained: it must have a clear hook, development, \
and resolution/punchline. A viewer who has NOT seen the full video must be \
able to understand and enjoy it.
- Clips must NOT overlap.
- Order clips by estimated virality potential (best first).

OUTPUT FORMAT:
Respond ONLY with a valid JSON object — no markdown, no code fences, \
no explanation. The object must have exactly one key "clips" whose value \
is an array. Each array element must have exactly these keys:

{{
  "clips": [
    {{
      "start_time": <float, seconds into the source video>,
      "end_time":   <float, seconds into the source video>,
      "hook_summary": "<1–2 sentence English explanation of why this clip \
will perform well on YouTube Shorts>",
      "seo": {{
        "title":       "<Greek title, max 60 chars, compelling & keyword-rich \
— NOT a direct transcript quote>",
        "description": "<Greek description, concise max 500 chars, hook in first 2 \
lines, value in the rest, CTA at the end>",
        "tags":        ["<English tag 1>", ..., "<English tag 10>"]
      }},
      "broll_query": "<2–5 word English Pexels search query for stock footage \
matching what the speaker is talking about in THIS clip>"
    }}
  ]
}}

RULES:
- title and description inside "seo" must be in Greek.
- tags must be plain English words or short phrases (no # prefix), \
exactly 10 items.
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


def _parse_single_clip(raw: dict, index: int, min_dur: float, max_dur: float) -> Optional[ClipCandidate]:
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
        start = float(raw["start_time"])
        end = float(raw["end_time"])
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


def _build_fallback_clips(
    segments: list["TranscriptionSegment"],
    max_clips: int,
    min_dur: float,
    max_dur: float,
) -> list[ClipCandidate]:
    """
    Generate evenly-distributed clips as a fallback when the Gemini call fails.

    Divides the transcript into *max_clips* evenly spaced windows of
    *target_dur* seconds, ensuring no clip extends beyond the last segment.

    Args:
        segments: Full transcript segments.
        max_clips: Maximum number of clips to generate.
        min_dur:   Minimum clip duration (seconds).
        max_dur:   Maximum clip duration (seconds).

    Returns:
        List of ClipCandidate objects.
    """
    if not segments:
        return []

    source_end = segments[-1].end
    source_start = segments[0].start
    target_dur = (min_dur + max_dur) / 2.0

    # Step size so clips are spread across the full video
    source_duration = source_end - source_start
    if source_duration <= target_dur:
        # Entire video fits in one clip
        clips_count = 1
        step = 0.0
    else:
        clips_count = min(max_clips, int(source_duration // target_dur))
        step = source_duration / clips_count

    candidates: list[ClipCandidate] = []
    for i in range(clips_count):
        clip_start = source_start + i * step
        clip_end = min(clip_start + target_dur, source_end)
        if clip_end - clip_start < min_dur:
            break

        seo = SeoMetadata.fallback(f"Clip {i + 1}")
        candidates.append(
            ClipCandidate(
                index=i + 1,
                start_time=round(clip_start, 3),
                end_time=round(clip_end, 3),
                hook_summary="Fallback clip — Gemini selection unavailable.",
                seo=seo,
                broll_query="people talking",
            )
        )

    return candidates


# ── Public API ─────────────────────────────────────────────────────────────────


def select_clips(
    segments: list["TranscriptionSegment"],
    gemini_api_key: str,
    max_clips: int = 10,
    min_dur: float = 20.0,
    max_dur: float = 40.0,
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
        min_dur:         Minimum clip duration in seconds.
        max_dur:         Maximum clip duration in seconds.
        source_title:    Original video title, provided to Gemini for context.

    Returns:
        List of ClipCandidate objects, ordered by virality (best first),
        at most *max_clips* items.
    """
    max_clips = max(1, min(10, max_clips))

    if not segments:
        logger.warning("select_clips: empty segments list — returning empty.")
        return []

    if not gemini_api_key or not gemini_api_key.strip():
        logger.warning("Gemini API key absent — using fallback clip distribution.")
        return _build_fallback_clips(segments, max_clips, min_dur, max_dur)

    transcript_block = format_transcript_with_timestamps(segments)
    prompt = _CLIP_SELECTION_PROMPT.format(
        source_title=source_title or "Unknown",
        max_clips=max_clips,
        min_dur=int(min_dur),
        max_dur=int(max_dur),
        transcript=transcript_block,
    )

    logger.info(
        "Calling Gemini for clip selection (max=%d, dur=[%d-%d]s, transcript_len=%d chars)...",
        max_clips, int(min_dur), int(max_dur), len(transcript_block),
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
        return _build_fallback_clips(segments, max_clips, min_dur, max_dur)

    try:
        raw_clips = _parse_clips_json(raw_text)
    except ValueError as exc:
        logger.error("Clip JSON parse failed: %s — using fallback.", exc)
        return _build_fallback_clips(segments, max_clips, min_dur, max_dur)

    # Parse and validate each clip; skip malformed ones
    candidates: list[ClipCandidate] = []
    for i, raw_clip in enumerate(raw_clips[:max_clips]):
        clip = _parse_single_clip(raw_clip, index=i + 1, min_dur=min_dur, max_dur=max_dur)
        if clip is not None:
            candidates.append(clip)

    if not candidates:
        logger.warning("No valid clips parsed from Gemini response — using fallback.")
        return _build_fallback_clips(segments, max_clips, min_dur, max_dur)

    # Remove overlapping clips (keep the earlier one in the ranked order)
    deduplicated: list[ClipCandidate] = []
    for clip in candidates:
        if not any(
            clip.start_time < existing.end_time and clip.end_time > existing.start_time
            for existing in deduplicated
        ):
            deduplicated.append(clip)

    logger.info(
        "Clip selection complete: %d/%d clips selected, %d after deduplication.",
        len(candidates), len(raw_clips), len(deduplicated),
    )
    return deduplicated[:max_clips]
