"""
services/seo_generator.py — YouTube Shorts SEO metadata via Gemini 2.5 Flash.

Responsibilities:
  1. Accept the full transcript text (Greek) for a processed Short.
  2. Construct a structured prompt requesting a JSON-formatted SEO package.
  3. Call the Gemini API using the official google-genai SDK.
  4. Parse and validate the JSON response.
  5. Return a typed SeoMetadata object.

This module has zero FFmpeg or transcription dependencies.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_GEMINI_MODEL = "gemini-2.5-flash"
_MAX_OUTPUT_TOKENS = 1024

# Strict prompt requesting JSON so we can parse deterministically
_PROMPT_TEMPLATE = """\
You are a YouTube Shorts SEO expert specialising in Greek-language content.

Given the following transcript from a Greek YouTube Short, generate optimised \
SEO metadata.

Respond ONLY with a valid JSON object — no markdown, no code fences, no \
explanation. The JSON must have exactly these keys:

{{
  "title": "<Greek title, max 60 characters, compelling & keyword-rich>",
  "description": "<Greek description, max 5000 characters, first 2 lines \
are the hook, rest expands value, ends with a CTA>",
  "tags": ["<English hashtag 1>", "<English hashtag 2>", ..., \
"<English hashtag 10>"]
}}

Rules:
- title and description must be in Greek.
- tags must be plain English words or short phrases (no # prefix).
- tags list must contain exactly 10 items.
- Do NOT include any text outside the JSON object.
- CRITICAL — The title must NOT be a direct quote or close paraphrase of the \
transcript. It must instead capture the TOPIC, HOOK, or VALUE PROPOSITION of \
the video (e.g. what the viewer will learn, why they should care, or what \
makes this clip interesting). Think of it as a clickable headline, not a \
transcript excerpt.

Transcript:
{transcript}
"""


# ── Public Types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SeoMetadata:
    """Immutable SEO metadata package for a single YouTube Short."""

    title: str
    description: str
    tags: tuple[str, ...]  # Immutable sequence, exactly 10 items

    @classmethod
    def fallback(cls, transcript_excerpt: str) -> "SeoMetadata":
        """
        Create a minimal fallback SeoMetadata when the API call fails.

        The fallback uses the first 60 characters of the transcript as the
        title and an empty description, ensuring the pipeline never crashes
        due to a missing SEO payload.
        """
        short_title = transcript_excerpt[:57] + "..." if len(transcript_excerpt) > 60 else transcript_excerpt
        return cls(
            title=short_title or "Greek Short",
            description=transcript_excerpt,
            tags=tuple([
                "shorts", "greek", "viral", "trending", "reels",
                "video", "fyp", "explore", "content", "greece",
            ]),
        )


# ── Internal Helpers ───────────────────────────────────────────────────────────

def _extract_json_from_response(text: str) -> dict:
    """
    Extract a JSON object from the model's response text.

    Gemini sometimes wraps JSON in markdown code fences despite instructions.
    This function strips fences and parses the bare JSON.

    Args:
        text: Raw response string from the Gemini API.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no valid JSON object is found.
    """
    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    # Try to extract the first JSON object in the string
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not obj_match:
        raise ValueError(f"No JSON object found in Gemini response: {text!r}")

    try:
        return json.loads(obj_match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse JSON from Gemini response: {exc}\nRaw text: {text!r}"
        ) from exc


def _validate_seo_dict(data: dict) -> SeoMetadata:
    """
    Validate the parsed SEO dict and coerce it into a SeoMetadata instance.

    Applies soft fixes (truncation, padding) rather than hard failures to
    maintain pipeline resilience.

    Args:
        data: Dict parsed from the Gemini JSON response.

    Returns:
        A validated SeoMetadata instance.

    Raises:
        ValueError: If required keys are entirely missing.
    """
    required_keys = {"title", "description", "tags"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Gemini response missing required keys: {missing}")

    title: str = str(data["title"])[:60]  # Hard-cap to YouTube's limit
    description: str = str(data["description"])[:5000]

    raw_tags = data["tags"]
    if not isinstance(raw_tags, list):
        raw_tags = []

    # Normalise: strip '#', strip whitespace, coerce to strings
    tags: list[str] = [str(t).lstrip("#").strip() for t in raw_tags]

    # Enforce exactly 10 tags: truncate if over, pad with generic terms if under
    _PADDING_TAGS = [
        "shorts", "greek", "viral", "trending", "reels",
        "video", "fyp", "explore", "content", "greece",
    ]
    if len(tags) > 10:
        tags = tags[:10]
    while len(tags) < 10:
        for pad in _PADDING_TAGS:
            if pad not in tags:
                tags.append(pad)
            if len(tags) == 10:
                break

    return SeoMetadata(title=title, description=description, tags=tuple(tags[:10]))


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_seo(transcript_text: str, api_key: str) -> SeoMetadata:
    """
    Generate YouTube Shorts SEO metadata from a Greek transcript using Gemini.

    If the API call fails for any reason (network, quota, parse error), a
    graceful fallback SeoMetadata is returned rather than propagating an
    exception, so the pipeline can still deliver a processed video.

    Args:
        transcript_text: Full Greek transcript text for the Short.
        api_key:         Google Gemini API key.

    Returns:
        A SeoMetadata instance (either from the API or a fallback).
    """
    if not api_key or not api_key.strip():
        logger.warning("Gemini API key is empty — returning fallback SEO metadata.")
        return SeoMetadata.fallback(transcript_text)

    if not transcript_text.strip():
        logger.warning("Transcript is empty — returning fallback SEO metadata.")
        return SeoMetadata.fallback("")

    prompt = _PROMPT_TEMPLATE.format(transcript=transcript_text)

    logger.info("Calling Gemini (%s) for SEO metadata...", _GEMINI_MODEL)
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=_MAX_OUTPUT_TOKENS,
                temperature=0.4,  # Lower temperature for deterministic JSON
                response_mime_type="application/json",
            ),
        )
        raw_text: str = response.text or ""
    except Exception as exc:
        logger.error("Gemini API call failed: %s — using fallback SEO.", exc)
        return SeoMetadata.fallback(transcript_text)

    try:
        data = _extract_json_from_response(raw_text)
        seo = _validate_seo_dict(data)
    except ValueError as exc:
        logger.error("SEO JSON parse/validate failed: %s — using fallback SEO.", exc)
        return SeoMetadata.fallback(transcript_text)

    logger.info("SEO metadata generated: title='%s', %d tags.", seo.title, len(seo.tags))
    return seo


def seo_to_dict(seo: SeoMetadata) -> dict[str, object]:
    """
    Serialise a SeoMetadata instance to a plain dict for JSON export.

    Args:
        seo: SeoMetadata instance.

    Returns:
        Dict with string keys: 'title', 'description', 'tags'.
    """
    return {
        "title": seo.title,
        "description": seo.description,
        "tags": list(seo.tags),
    }


# ── B-Roll Query Generation ────────────────────────────────────────────────────

# Prompt asking Gemini to produce a short, Pexels-optimised English query
# that describes a visually compelling scene matching the transcript topic.
_BROLL_QUERY_PROMPT = """\
You are a professional video editor choosing stock B-roll footage for a \
Greek YouTube Short about the topic spoken in the transcript below.

Your task: output ONE short English search query (2–5 words) for Pexels \
stock video that directly matches what the speaker is TALKING ABOUT.

STRICT RULES:
1. The query must describe a CONCRETE VISUAL SCENE, not an abstract concept.
2. The query must match the SPECIFIC TOPIC of the speech — not a generic \
   lifestyle or scenic shot.
3. NEVER output: beach, nature, sunset, feet, travel, vacation, relaxation, \
   or any scene unrelated to the speech topic.
4. If the speech is about PEOPLE AND RELATIONSHIPS → use: "people talking", \
   "friends reunion", "group conversation", "colleagues meeting"
5. If the speech is about BUSINESS / ECONOMY → use: "business meeting", \
   "stock market trading", "office team discussion"
6. If the speech is about TECHNOLOGY → use: "person using smartphone", \
   "software developer coding", "video call conference"
7. If the speech is about HEALTH → use: "doctor patient consultation", \
   "hospital corridor", "medical team"
8. Always output English only. No quotes. No punctuation. No explanation. \
   Output ONLY the search query on a single line.

Examples of GOOD queries: "two people video call", "business handshake office", \
"crowd protest street", "scientist laboratory experiment"
Examples of BAD queries: "distance", "communication", "life", "connection", \
"summer", "relaxation"

Transcript:
{transcript}
"""


def generate_broll_query(transcript_text: str, api_key: str) -> Optional[str]:
    """
    Use Gemini to derive a visually-meaningful English B-roll search query
    from a Greek transcript.

    Gemini understands the semantic content of the transcript and maps it
    to a concrete, Pexels-friendly visual scene description — far more
    accurate than the stopword-based word-picking fallback.

    Args:
        transcript_text: Full Greek transcript text.
        api_key:         Google Gemini API key.

    Returns:
        A 2-4 word English search query string, or None if the call fails
        (the pipeline will then fall back to extract_broll_query).
    """
    if not api_key or not api_key.strip():
        logger.debug("Gemini key absent — skipping AI broll query generation.")
        return None

    if not transcript_text.strip():
        return None

    prompt = _BROLL_QUERY_PROMPT.format(transcript=transcript_text[:2000])

    logger.info("Calling Gemini to generate B-roll search query...")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=32,   # Only a few words needed
                temperature=0.2,        # Low variance for consistency
            ),
        )
        raw: str = (response.text or "").strip()
    except Exception as exc:
        logger.warning("Gemini broll query call failed: %s — using fallback.", exc)
        return None

    # Sanitise: keep only the first line, strip quotes/punctuation
    query = raw.splitlines()[0].strip().strip('"\'.,')
    if not query:
        return None

    # Reject responses that are obviously wrong (too long, non-English markers)
    word_count = len(query.split())
    if word_count > 8:
        logger.warning(
            "Gemini broll query too long (%d words): '%s' — using fallback.",
            word_count, query,
        )
        return None

    logger.info("Gemini broll query: '%s'", query)
    return query

