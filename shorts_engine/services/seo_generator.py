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
import unicodedata
from dataclasses import dataclass

from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_GEMINI_MODELS = (
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-flash-latest",
)
_MAX_OUTPUT_TOKENS = 4096


def _call_gemini_with_fallback(
    client: genai.Client,
    contents: str,
    config: genai_types.GenerateContentConfig,
) -> str:
    """
    Attempt content generation across known Flash models in fallback order.
    Catches 404 (model deprecated) and 503 (high demand) to ensure resilience.
    """
    # Disable thinking tokens if unconfigured so output tokens aren't consumed by thought mode
    if getattr(config, "thinking_config", None) is None:
        try:
            config.thinking_config = genai_types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass

    last_err: Exception | None = None
    for model in _GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            logger.warning("Gemini model '%s' failed: %s — trying fallback...", model, exc)
            last_err = exc
    if last_err is not None:
        raise last_err
    raise RuntimeError("No Gemini models available.")

# Strict prompt requesting JSON so we can parse deterministically
_PROMPT_TEMPLATE = """\
You are an expert editorial strategist and YouTube Shorts SEO specialist for Greek-language video productions.

Given the following transcript and source context, generate an authoritative, high-CTR, \
highly engaging SEO package matching the prestige, subject matter, and visual image of the video.

SOURCE VIDEO CONTEXT / TITLE: {source_title}

Respond ONLY with a valid JSON object — no markdown, no code fences, no \
explanation. The JSON must have exactly these keys:

{{
  "title": "<Greek title, max 60 characters, engaging and curiosity-inducing, 1-2 strategic emojis allowed>",
  "description": "<Greek description, max 5000 characters, structured with hook in first 2 lines, core value in the rest, CTA and 3-5 trending hashtags at the end, 1-2 strategic emojis allowed>",
  "tags": ["<tag 1>", "<tag 2>", ..., "<tag 16>"]
}}

TITLE & STYLE RULES (CRITICAL):
- Language: All titles and descriptions MUST be in Greek.
- IMAGE & BRAND ALIGNMENT: The title must fit the visual identity of the source video (e.g., serious interview vs engaging vlog) but MUST prioritize high retention and Click-Through Rate (CTR).
- ENGAGING TONE:
  * Write in a highly engaging, relatable, or authoritative tone depending on the content.
  * You may use third-person ("Τι αποκαλύπτουν τα στοιχεία...") or curiosity-driven hooks ("Ο λόγος που...").
  * Avoid cheap clickbait, but ensure the title creates a strong curiosity gap.
- EMOJIS ALLOWED: You MAY use 1 or 2 highly relevant emojis (e.g., 🤯, 🔥, 📈, 🚨) to act as visual pattern interrupts and increase CTR. Do not overuse them.
- NOT A DIRECT QUOTE: The title must capture the core topic, thesis, or value proposition — not a flat excerpt or quote from the transcript. Max 60 characters.

DESCRIPTION & TAGS RULES:
- description must follow the 3-part structure:
  1. Lines 1–2: High-impact hook summarizing the core takeaway with primary search keywords.
  2. Lines 3–4: Analytical value expansion / key topics explored.
  3. Call to Action (CTA) + 3–5 relevant Greek hashtags (e.g. #Shorts #Ελλάδα).
- tags must be 12 to 18 high-performing keywords and search phrases (mix of Greek search queries, entity/speaker names, and topic categories). NO '#' prefix.
- Do NOT include any text outside the JSON object.

Transcript:
{transcript}
"""


# ── Public Types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SeoMetadata:
    """Immutable SEO metadata package for a single YouTube Short."""

    title: str
    description: str
    tags: tuple[str, ...]  # Immutable sequence of SEO tags

    @property
    def youtube_tags_display(self) -> str:
        """Comma-separated tag list formatted for direct copy-paste into YouTube Studio."""
        return ", ".join(self.tags)

    @property
    def safe_filename(self) -> str:
        """
        Filesystem-safe filename derived from the title (without extension).
        Removes characters prohibited across operating systems (< > : " / \\ | ? *)
        and normalises whitespace.
        """
        return sanitize_filename(self.title)

    @classmethod
    def fallback(cls, transcript_excerpt: str) -> SeoMetadata:
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


_EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff]|"  # SMP characters (emojis, transport, pictographs)
    r"[\u2600-\u27BF]|"          # Dingbats and miscellaneous symbols
    r"[\uFE00-\uFE0F]|"          # Variation selectors
    r"[\u200D]|"                 # Zero-width joiner
    r"[\u2300-\u23FF]|"          # Misc technical symbols
    r"[\u2B50\u2B55\u2934\u2935\u25AA-\u25FE]"  # Stars, circles, arrows, shapes
)


def strip_emojis(text: str) -> str:
    """
    Remove all emoji characters, symbols, and pictographs from text,
    normalising surrounding whitespace and punctuation.

    Preserves standard alphabets (Greek, Latin), numbers, accents,
    currency symbols, and punctuation marks.
    """
    if not text:
        return ""
    cleaned = _EMOJI_PATTERN.sub("", text)
    # Filter any remaining 'So' (Symbol, other) unicode characters, preserving currency/punctuation
    preserved = set("+=-%€$#@/\\:;.,!?()[]\"'«»")
    result: list[str] = []
    for ch in cleaned:
        cat = unicodedata.category(ch)
        if cat in ("So", "Cn") and ch not in preserved:
            continue
        result.append(ch)
    cleaned = "".join(result)
    cleaned = re.sub(r"\s+([!?:;.,])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def sanitize_filename(title: str, max_length: int = 100) -> str:
    """
    Sanitize a title into a safe, cross-platform filename stem without extension.

    Removes or converts characters prohibited on Windows, macOS, and Linux
    (< > : " / \\ | ? * and ASCII control characters) while preserving
    Greek and international unicode characters.

    Args:
        title: The source string (e.g. SEO title).
        max_length: Maximum character length for the resulting filename stem.

    Returns:
        Cleaned, filesystem-safe filename string.
    """
    if not title:
        return ""
        
    title = strip_emojis(title)

    # Replace colons, vertical pipes, and slashes with clean ' - ' separators
    cleaned = re.sub(r"[\s]*[/\\:|][\s]*", " - ", title)
    # Remove prohibited filename characters and control characters
    cleaned = re.sub(r'[<>"?*\x00-\x1f]', "", cleaned)
    # Collapse multiple spaces or dashes
    cleaned = re.sub(r"\s*-\s*-+\s*", " - ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Strip leading/trailing spaces, dots, and dashes
    cleaned = cleaned.strip(" .-\t\r\n")

    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(" .-\t\r\n")

    return cleaned


def get_download_filename(
    title: str | None = None,
    fallback_filename: str = "clip.mp4",
    extension: str = "mp4",
) -> str:
    """
    Compute a safe, user-friendly download filename.
    Uses title (e.g. SEO title) if non-empty and valid; otherwise falls back to fallback_filename.

    Args:
        title: The desired title string (e.g. SEO title).
        fallback_filename: Fallback filename if title is missing or invalid.
        extension: Desired file extension (e.g. 'mp4' or 'json').

    Returns:
        Safe filename string including extension.
    """
    from pathlib import Path

    ext = extension.lstrip(".")
    if title:
        safe = sanitize_filename(title)
        if safe:
            return f"{safe}.{ext}"
    if fallback_filename:
        p = Path(fallback_filename)
        if p.suffix.lstrip(".").lower() == ext.lower():
            return p.name
        return f"{p.stem}.{ext}"
    return f"clip.{ext}"


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

    raw_title = str(data["title"])
    clean_title = raw_title.strip()
    if not clean_title:
        clean_title = "Ελληνικό Βίντεο"
    title: str = clean_title[:60]  # Hard-cap to YouTube's limit

    clean_description = str(data["description"]).strip()
    description: str = clean_description[:5000]

    raw_tags = data["tags"]
    if not isinstance(raw_tags, list):
        raw_tags = []

    # Normalise: strip emojis, strip '#', strip whitespace, remove empty, deduplicate preserving order
    tags: list[str] = []
    for t in raw_tags:
        cleaned = strip_emojis(str(t).lstrip("#").strip().strip(","))
        if cleaned and cleaned not in tags:
            tags.append(cleaned)

    # Pad with essential category tags if fewer than 10 tags provided
    _PADDING_TAGS = [
        "shorts", "greek", "viral", "trending", "reels",
        "video", "fyp", "explore", "content", "greece",
    ]
    if len(tags) > 20:
        tags = tags[:20]
    while len(tags) < 10:
        for pad in _PADDING_TAGS:
            if pad not in tags:
                tags.append(pad)
            if len(tags) >= 10:
                break

    return SeoMetadata(title=title, description=description, tags=tuple(tags))


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_seo(
    transcript_text: str,
    api_key: str,
    source_title: str = "",
) -> SeoMetadata:
    """
    Generate YouTube Shorts SEO metadata from a Greek transcript using Gemini.

    If the API call fails for any reason (network, quota, parse error), a
    graceful fallback SeoMetadata is returned rather than propagating an
    exception, so the pipeline can still deliver a processed video.

    Args:
        transcript_text: Full Greek transcript text for the Short.
        api_key:         Google Gemini API key.
        source_title:    Optional source video title or topic context.

    Returns:
        A SeoMetadata instance (either from the API or a fallback).
    """
    if not api_key or not api_key.strip():
        logger.warning("Gemini API key is empty — returning fallback SEO metadata.")
        return SeoMetadata.fallback(transcript_text)

    if not transcript_text.strip():
        logger.warning("Transcript is empty — returning fallback SEO metadata.")
        return SeoMetadata.fallback("")

    prompt = _PROMPT_TEMPLATE.format(
        transcript=transcript_text,
        source_title=source_title or "Unknown",
    )

    logger.info("Calling Gemini for SEO metadata...")
    try:
        client = genai.Client(api_key=api_key)
        raw_text = _call_gemini_with_fallback(
            client=client,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=_MAX_OUTPUT_TOKENS,
                temperature=0.4,  # Lower temperature for deterministic JSON
                response_mime_type="application/json",
            ),
        )
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


def generate_broll_query(transcript_text: str, api_key: str) -> str | None:
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
        raw = _call_gemini_with_fallback(
            client=client,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=1024,  # Adequate headroom for thinking tokens + query
                temperature=0.2,         # Low variance for consistency
            ),
        ).strip()
    except Exception as exc:
        logger.warning("Gemini broll query call failed: %s — using fallback.", exc)
        return None

    # Sanitise: keep only the first non-empty line, strip quotes/punctuation
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None
    query = lines[0].strip('"\'.,')
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


# ── Greek Transcript Correction ────────────────────────────────────────────────

_CORRECTION_PROMPT = """\
You are a professional Greek language editor and proofreader.

The following lines are raw speech-to-text transcript segments from Greek \
audio transcribed by Whisper. They often contain speech-to-text phonetic \
errors, wrong word boundaries, missing/wrong diacritics (τόνοι), or grammar mistakes.

Common Whisper Greek errors to ALWAYS correct:
- Passive verb endings misheard as separate words (e.g. "Χαίρο με" / "χαίρο με" -> "Χαίρομαι" / "χαίρομαι", "σκέφτο με" -> "σκέφτομαι")
- Verb forms: "είσαστε" -> "είστε", "βλέπωμε" -> "βλέπουμε"
- "ό,τι" vs "ότι", "πως" vs "πώς", "που" vs "πού"
- Missing accent marks (τόνοι) and spelling errors

Correct EACH line so it reads as accurate, natural, grammatically correct Greek. Follow these rules:
1. Output the EXACT SAME number of lines as input — one corrected line per input line.
2. Do NOT merge or split lines.
3. Do NOT add unnecessary punctuation; keep subtitles clean and natural.
4. Do NOT change the speaker's intended meaning.
5. If a line is already correct, output it unchanged.
6. Output ONLY the corrected lines, nothing else.

Lines to correct:
{lines}
"""


def align_words_with_corrected_text(
    original_words: list[tuple[float, float, str]],
    corrected_text: str,
    seg_start: float,
    seg_end: float,
) -> list[tuple[float, float, str]]:
    """
    Synchronise word-level timestamps with Gemini-corrected Greek text.

    If the word count matches the original Whisper segment, the original
    precise timestamps are preserved and only the word text is updated.
    If the word count changed (e.g. 'Χαίρο' + 'με' merged into 'Χαίρομαι',
    or words split/joined), the segment duration is proportionally distributed
    across the new words weighted by character length.
    """
    new_words = corrected_text.strip().split()
    if not new_words:
        return original_words

    if original_words and len(original_words) == len(new_words):
        return [(w[0], w[1], nw) for w, nw in zip(original_words, new_words)]

    total_chars = max(1, sum(len(w) for w in new_words))
    total_duration = max(0.1, seg_end - seg_start)

    aligned: list[tuple[float, float, str]] = []
    current_time = seg_start
    for i, nw in enumerate(new_words):
        if i == len(new_words) - 1:
            w_end = seg_end
        else:
            w_dur = (len(nw) / total_chars) * total_duration
            w_end = current_time + w_dur
        aligned.append((round(current_time, 3), round(w_end, 3), nw))
        current_time = w_end

    return aligned


def correct_transcript_greek(
    segments: list,
    api_key: str,
) -> list:
    """
    Use Gemini to fix Whisper transcription errors in Greek segments.

    Sends the text of each segment to Gemini for grammar/spelling correction,
    then maps the corrected text back to the original segments while preserving
    all word-level timing data (only the .text attribute is updated).

    Args:
        segments: List of TranscriptionSegment objects from transcribe().
        api_key:  Google Gemini API key.

    Returns:
        List of corrected TranscriptionSegment objects (or original on failure).
    """
    # Lazy import to avoid circular dependency between seo_generator ↔ transcriber
    from services.transcriber import TranscriptionSegment

    if not api_key or not api_key.strip():
        logger.debug("Gemini key absent — skipping transcript correction.")
        return segments

    if not segments:
        return segments

    # Build numbered input lines (one per segment)
    input_lines = [seg.text for seg in segments]

    # Limit prompt size — for very long transcripts batch in 60-line windows
    if len(input_lines) > 60:
        logger.info(
            "Transcript has %d segments; correcting in one batch (first 60).",
            len(input_lines),
        )
        input_lines_batch = input_lines[:60]
        rest = segments[60:]
    else:
        input_lines_batch = input_lines
        rest = []

    prompt = _CORRECTION_PROMPT.format(lines="\n".join(input_lines_batch))

    logger.info("Calling Gemini to correct %d transcript segments...", len(input_lines_batch))
    try:
        client = genai.Client(api_key=api_key)
        raw = _call_gemini_with_fallback(
            client=client,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=2048,
                temperature=0.1,  # Very low — stay close to original
            ),
        ).strip()
    except Exception as exc:
        logger.warning("Gemini transcript correction failed: %s — using original.", exc)
        return segments

    corrected_lines = [line.strip() for line in raw.splitlines() if line.strip()]

    # Validate: must have same count as input batch
    if len(corrected_lines) != len(input_lines_batch):
        logger.warning(
            "Gemini returned %d corrected lines for %d segments — using original.",
            len(corrected_lines), len(input_lines_batch),
        )
        return segments

    # Rebuild segments with corrected text and synchronised word timings
    corrected_segments: list[TranscriptionSegment] = []
    for seg, new_text in zip(segments[:len(input_lines_batch)], corrected_lines):
        aligned_words = align_words_with_corrected_text(
            original_words=seg.words,
            corrected_text=new_text,
            seg_start=seg.start,
            seg_end=seg.end,
        )
        corrected_segments.append(
            TranscriptionSegment(
                start=seg.start,
                end=seg.end,
                text=new_text,
                words=aligned_words,
            )
        )

    return corrected_segments + rest
