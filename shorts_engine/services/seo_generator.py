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

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from google.genai.errors import APIError

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_GEMINI_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
)
_MAX_OUTPUT_TOKENS = 4096


def _call_gemini_with_fallback(
    client: genai.Client,
    contents: str,
    config: genai_types.GenerateContentConfig,
) -> str:
    """
    Attempt content generation across known Flash models in fallback order.
    Implements exponential backoff on transient errors (503 UNAVAILABLE, timeouts)
    and immediate failover on rate limits (429) or missing models (404).
    """
    last_err: Exception | None = None
    for model in _GEMINI_MODELS:
        attempts_for_model = 2
        for attempt in range(attempts_for_model):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                return response.text or ""
            except (APIError, OSError, ValueError, RuntimeError) as exc:
                last_err = exc
                err_str = str(exc)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
                    logger.warning("Gemini Rate Limit on %s — immediately failing over to next model.", model)
                    break
                if "404" in err_str or "NOT_FOUND" in err_str:
                    logger.debug("Gemini model '%s' not found — skipping to next.", model)
                    break
                if ("503" in err_str or "UNAVAILABLE" in err_str) and attempt < attempts_for_model - 1:
                    backoff = 2.0 * (attempt + 1)
                    logger.warning("Gemini model '%s' temporarily unavailable (503). Retrying in %.1fs...", model, backoff)
                    import time
                    time.sleep(backoff)
                    continue
                logger.warning("Gemini model '%s' failed on attempt %d: %s — trying fallback...", model, attempt + 1, exc)
                break
    if last_err is not None:
        raise last_err
    raise RuntimeError("No Gemini models available.")

# Strict prompt requesting JSON so we can parse deterministically
_PROMPT_TEMPLATE = """\
You are an expert editorial strategist and YouTube Shorts SEO specialist for Greek-language video productions.

Given the following transcript and source context, generate an authoritative, high-CTR, \
highly engaging SEO package matching the prestige, subject matter, and visual image of the video.

{brand_voice_injection}

SOURCE VIDEO CONTEXT / TITLE: {source_title}

Respond ONLY with a valid JSON object — no markdown, no code fences, no \
explanation. The JSON must have exactly these keys:

{{
  "title": "<Greek title, max 60 chars. MUST BE AN ORIGINAL PHRASE that summarizes the core topic. DO NOT USE DIRECT QUOTES. 1-2 strategic emojis allowed>",
  "curiosity_title": "<Greek title focusing purely on the curiosity gap/mystery>",
  "authority_title": "<Greek title focusing on authority, facts, or ultimate solutions>",
  "contrarian_title": "<Greek title focusing on a controversial, edgy, or 'why you are wrong' angle>",
  "primary_keyword": "<1-2 words Greek keyword that represents the core topic, exactly as it might appear in the transcript>",
  "description": "<Greek description, max 5000 characters, structured with hook in first 2 lines, core value in the rest, CTA and 3-5 trending hashtags at the end, 1-2 strategic emojis allowed>",
  "pinned_comment": "<An Engagement Trap Greek comment to pin at the top of the comments section. It MUST ask a polarizing question, state a strong opinion, or use a fill-in-the-blank prompt designed specifically to farm comments and boost the YouTube algorithm>",
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
- NO DIRECT QUOTES (CRITICAL): The titles MUST be completely original, punchy phrases that act as a hook or summary. They MUST NEVER be sentences copied from the transcript. Max 60 characters.

DESCRIPTION & TAGS RULES:
- description must follow the 3-part structure:
  1. Lines 1–2: High-impact hook summarizing the core takeaway with primary search keywords.
  2. Lines 3–4: Analytical value expansion / key topics explored.
  3. Call to Action (CTA) + 3–5 relevant Greek hashtags (e.g. #Shorts #Ελλάδα).
- tags must be 12 to 18 high-performing keywords and search phrases (mix of Greek search queries, entity/speaker names, topic categories, and TREND-JACKING keywords related to the broader category even if not explicitly mentioned). NO '#' prefix.
- Do NOT include any text outside the JSON object.

Transcript:
{transcript}
"""

_CRITIC_PROMPT = """\
You are an elite YouTube Shorts SEO Critic. Your job is to review a generated title and description \
for a Greek YouTube Short and determine if it has high viral potential, strong Click-Through Rate (CTR) potential, \
and a compelling curiosity gap.

Score the Title's CTR potential from 1 to 10.
If the score is below 7, provide specific instructions on how to rewrite the Title and Description to make it more viral.
If the score is 7 or higher, you can just say "Score: X. Looks great."

Review the following:
Title: {title}
Description: {description}

JSON Output Schema:
{{
  "score": <int>,
  "feedback": "<string explanation or rewrite instructions>"
}}
Respond ONLY with valid JSON.
"""



# ── Public Types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SeoMetadata:
    """Immutable SEO metadata package for a single YouTube Short."""

    title: str
    description: str
    tags: tuple[str, ...]  # Immutable sequence of SEO tags
    primary_keyword: str
    pinned_comment: str
    curiosity_title: str
    authority_title: str
    contrarian_title: str

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
            tags=(
                "shorts", "greek", "viral", "trending", "reels",
                "video", "fyp", "explore", "content", "greece",
            ),
            primary_keyword="shorts",
            pinned_comment="Ποια είναι η δική σας άποψη; Γράψτε στα σχόλια! 👇",
            curiosity_title="Greek Short (Curiosity)",
            authority_title="Greek Short (Authority)",
            contrarian_title="Greek Short (Contrarian)",
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

    # Parse new advanced SEO fields with safe fallbacks
    primary_keyword = str(data.get("primary_keyword", "")).strip()
    pinned_comment = str(data.get("pinned_comment", "")).strip()
    curiosity_title = str(data.get("curiosity_title", "")).strip()[:60]
    authority_title = str(data.get("authority_title", "")).strip()[:60]
    contrarian_title = str(data.get("contrarian_title", "")).strip()[:60]

    # Pad with essential category tags if fewer than 10 tags provided
    _PADDING_TAGS = [
        "shorts", "greek", "viral", "trending", "reels",
        "video", "fyp", "explore", "content", "greece",
    ]
    if len(tags) > 20:
        tags = tags[:20]
    elif len(tags) < 10:
        for p in _PADDING_TAGS:
            if p not in tags:
                tags.append(p)
            if len(tags) >= 12:
                break

    return SeoMetadata(
        title=title,
        description=description,
        tags=tuple(tags),
        primary_keyword=primary_keyword,
        pinned_comment=pinned_comment,
        curiosity_title=curiosity_title,
        authority_title=authority_title,
        contrarian_title=contrarian_title,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_seo(
    transcript_text: str,
    api_key: str,
    source_title: str = "",
    brand_voice: str = "",
) -> SeoMetadata:
    """
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

    brand_voice_injection = ""
    if brand_voice:
        brand_voice_injection = f"CRITICAL BRAND VOICE RULES:\n- Ensure the tone matches this persona perfectly: '{brand_voice}'"

    prompt = _PROMPT_TEMPLATE.format(
        transcript=transcript_text,
        source_title=source_title or "Unknown",
        brand_voice_injection=brand_voice_injection,
    )

    # Memory Optimization: Check cache first
    try:
        from services.cache_manager import load_cache_pickle, save_cache_pickle
    except ImportError:
        from shorts_engine.services.cache_manager import (
            load_cache_pickle,
            save_cache_pickle,
        )

    # Hash the full prompt so we never store multi-KB strings as dict/file keys.
    cache_key = hashlib.md5(f"{prompt}_{source_title}".encode()).hexdigest()
    cached_seo = load_cache_pickle("seo_metadata", cache_key)
    if cached_seo is not None:
        return cached_seo

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
    except (APIError, genai_errors.APIError, RuntimeError, ValueError, OSError) as exc:
        logger.error("Gemini API call failed: %s — using fallback SEO.", exc)
        return SeoMetadata.fallback(transcript_text)

    try:
        data = _extract_json_from_response(raw_text)
        seo = _validate_seo_dict(data)
    except ValueError as exc:
        logger.error("SEO JSON parse/validate failed: %s — using fallback SEO.", exc)
        return SeoMetadata.fallback(transcript_text)

    # ── Virality Critic Loop ──
    critic_prompt = _CRITIC_PROMPT.format(title=seo.title, description=seo.description)
    try:
        critic_raw = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=critic_prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            )
        ).text.strip()
        critic_data = _extract_json_from_response(critic_raw)
        score = critic_data.get("score", 10)
        feedback = critic_data.get("feedback", "")
        
        if score < 7:
            logger.warning("Critic scored SEO %d/10. Feedback: %s. Regenerating...", score, feedback)
            # Regenerate once with feedback
            regen_prompt = prompt + f"\n\nCRITIC FEEDBACK FROM PREVIOUS ATTEMPT (Address this!):\n{feedback}"
            regen_raw = _call_gemini_with_fallback(
                client=client,
                contents=regen_prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=_MAX_OUTPUT_TOKENS,
                    temperature=0.7,  # Higher temp for creative rewrite
                    response_mime_type="application/json",
                ),
            )
            regen_data = _extract_json_from_response(regen_raw)
            seo = _validate_seo_dict(regen_data)
        else:
            logger.info("Critic approved SEO (Score: %d/10).", score)
    except (APIError, genai_errors.APIError, RuntimeError, ValueError, KeyError, OSError) as exc:
        logger.warning("Critic loop failed or skipped: %s", exc)

    logger.info("SEO metadata generated: title='%s', %d tags.", seo.title, len(seo.tags))
    save_cache_pickle("seo_metadata", cache_key, seo)
    return seo


def seo_to_dict(seo: SeoMetadata) -> dict[str, object]:
    """
    Serialise a SeoMetadata instance to a plain dict for JSON export.

    Args:
        seo: SeoMetadata instance.

    Returns:
        Dict with all SeoMetadata fields serialised to JSON-safe types.
    """
    return {
        "title": seo.title,
        "description": seo.description,
        "tags": list(seo.tags),
        "primary_keyword": seo.primary_keyword,
        "pinned_comment": seo.pinned_comment,
        "curiosity_title": seo.curiosity_title,
        "authority_title": seo.authority_title,
        "contrarian_title": seo.contrarian_title,
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
    except (APIError, genai_errors.APIError, RuntimeError, ValueError, OSError) as exc:
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
audio transcribed by Whisper. They often contain severe speech-to-text phonetic \
errors, wrong word boundaries, missing/wrong diacritics (τόνοι), or grammar mistakes.

Common Whisper Greek errors to ALWAYS correct:
- Phonetic misinterpretations: Fix words that sound similar but make no sense in the context of the sentence.
- Repetitions / Hallucinations: Remove unnatural repeating phrases or stuttering if they are clearly AI glitches.
- Passive verb endings misheard as separate words (e.g. "Χαίρο με" / "χαίρο με" -> "Χαίρομαι" / "χαίρομαι", "σκέφτο με" -> "σκέφτομαι")
- Verb forms: "είσαστε" -> "είστε", "βλέπωμε" -> "βλέπουμε"
- "ό,τι" vs "ότι", "πως" vs "πώς", "που" vs "πού"
- Missing accent marks (τόνοι) and spelling errors
- Fix Capitalization at the start of sentences and proper nouns.
- Contextual Acronyms, Public Organizations & Sense Checking (CRITICAL):
  Whisper frequently misinterprets Greek public institutions, news agencies, and economic acronyms phonetically or breaks them into separate gibberish words. You must detect the intended meaning from the context of the sentence:
  * "με με", "μμε", "μεμε", "μ μ ε", "τα με με" -> "ΜΜΕ"
  * "α δε", "ααδε", "α αδέ", "αάδε", "α δε ε", "α δ ε" -> "ΑΑΔΕ"
  * "δε η", "δεη", "δ ε η" -> "ΔΕΗ"
  * "δεδδηε", "δδε", "δ ε δ δ η ε" -> "ΔΕΔΔΗΕ"
  * "εφκα", "ε φ κ α" -> "ΕΦΚΑ"
  * "οασα", "ο α σ α" -> "ΟΑΣΑ"
  * "γεεθα", "γ ε ε θ α" -> "ΓΕΕΘΑ"
  * "ασεπ", "α σ ε π" -> "ΑΣΕΠ"
  * "οπεκα", "ο π ε κ α" -> "ΟΠΕΚΑ"
  * "φπα", "φ π α" -> "ΦΠΑ"
  * "ελστατ", "ελ στατ" -> "ΕΛΣΤΑΤ"
  * "εε", "ε ε" (when referring to European Union institutions or rules) -> "ΕΕ"
  * Always format Greek institutional and organizational acronyms in ALL-CAPS (e.g., "ΜΜΕ", "ΑΑΔΕ", "ΔΕΗ", "ΕΦΚΑ", "ΔΕΔΔΗΕ").
  * LOGIC CHECK: If a sentence discusses tax audits, revenue, news media, energy bills, or pensions, ensure the institutional acronym is logically applied (e.g., "έλεγχος από την ΑΑΔΕ", "ρεπορτάζ στα ΜΜΕ", "τιμολόγια της ΔΕΗ").

Correct EACH line so it reads as accurate, natural, grammatically correct Greek. Follow these rules:
1. Output the EXACT SAME number of lines as input — one corrected line per input line.
2. PRESERVE THE LINE INDEX: Each output line MUST start with its exact index tag matching the input (e.g. [0] ..., [1] ...).
3. Do NOT merge or split lines.
4. Do NOT add unnecessary punctuation; keep subtitles clean and natural.
5. Do NOT change the speaker's intended meaning, but aggressively fix nonsense and speech-to-text AI glitches.
6. LOGICAL GUARDRAIL (CRITICAL): Ensure the sentence actually makes sense. If the literal words form a confusing or disjointed sentence, rewrite them slightly to form a coherent, logical statement that fits the context.
7. If a line is already correct, output it with its tag unchanged.
8. Output ONLY the tagged lines, nothing else.

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
    import difflib

    new_words = corrected_text.strip().split()
    if not new_words:
        return original_words or []

    # Guard: segments without word-level timestamps (words=None) must fall back
    # to a proportional distribution across the corrected words.
    if not original_words:
        seg_dur = seg_end - seg_start
        per_word = seg_dur / len(new_words) if new_words else seg_dur
        return [
            (round(seg_start + i * per_word, 3), round(seg_start + (i + 1) * per_word, 3), w)
            for i, w in enumerate(new_words)
        ]

    if len(original_words) == len(new_words):
        return [(w[0], w[1], nw) for w, nw in zip(original_words, new_words)]

    # Use difflib to map the new words to the original words by string similarity
    orig_texts = [w[2].lower() for w in original_words]
    new_texts = [nw.lower() for nw in new_words]

    sm = difflib.SequenceMatcher(None, orig_texts, new_texts)
    aligned: list[tuple[float, float, str]] = []
    
    # We will compute an average word duration from the original words as a fallback
    avg_duration = max(0.1, (seg_end - seg_start) / len(original_words))

    # To track the last known valid time
    current_time = seg_start


    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal' or op == 'replace':
            # Map the new words to the old words proportionally within this block
            block_orig_words = original_words[i1:i2]
            block_new_words = new_words[j1:j2]
            
            if not block_orig_words or not block_new_words:
                continue
                
            block_start = block_orig_words[0][0]
            block_end = block_orig_words[-1][1]
            block_duration = block_end - block_start
            
            # Update current time
            current_time = block_start
            
            # Distribute time among new words in this block
            for w_idx, nw in enumerate(block_new_words):
                w_dur = block_duration / len(block_new_words)
                w_end = current_time + w_dur
                aligned.append((round(current_time, 3), round(w_end, 3), nw))
                current_time = w_end

        elif op == 'insert':
            # We have new words inserted without replacing any old words.
            # Give each new word the avg_duration starting from current_time.
            for nw in new_words[j1:j2]:
                w_end = current_time + avg_duration
                aligned.append((round(current_time, 3), round(w_end, 3), nw))
                current_time = w_end
                
        # op == 'delete' means old words were deleted, so we just skip them and don't add to aligned.

    # Ensure the final word ends correctly within segment limits if possible
    # We don't strictly enforce seg_end here since words might legitimately overrun,
    # but we could clamp it if needed. For now, trust the relative mapping.
    
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
    Processes transcripts in sequential batches of 50 to cover 100% of the video.

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

    batch_size = 50
    corrected_segments: list[TranscriptionSegment] = []

    try:
        client = genai.Client(api_key=api_key)
    except (genai_errors.APIError, RuntimeError, ValueError, OSError) as exc:
        logger.warning("Failed to initialize Gemini client for transcript correction: %s", exc)
        return segments

    total_batches = (len(segments) + batch_size - 1) // batch_size
    logger.info("Correcting %d transcript segments in %d batch(es)...", len(segments), total_batches)

    for b_idx in range(total_batches):
        batch = segments[b_idx * batch_size : (b_idx + 1) * batch_size]
        input_lines = [f"[{i}] {seg.text}" for i, seg in enumerate(batch)]
        prompt = _CORRECTION_PROMPT.format(lines="\n".join(input_lines))

        try:
            raw = _call_gemini_with_fallback(
                client=client,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=2048,
                    temperature=0.1,  # Very low — stay close to original
                ),
            ).strip()
        except (genai_errors.APIError, RuntimeError, ValueError, OSError) as exc:
            logger.warning("Gemini transcript correction failed on batch %d: %s — using original.", b_idx + 1, exc)
            corrected_segments.extend(batch)
            continue

        # Strip markdown fences if present
        raw_cleaned = raw.strip()
        if raw_cleaned.startswith("```"):
            raw_cleaned = "\n".join(raw_cleaned.split("\n")[1:])
        if raw_cleaned.endswith("```"):
            raw_cleaned = "\n".join(raw_cleaned.split("\n")[:-1])

        raw_lines = [line.strip() for line in raw_cleaned.splitlines() if line.strip()]

        # Parse indexed lines
        corrected_map: dict[int, str] = {}
        for line in raw_lines:
            match = re.match(r"^\[(\d+)\]\s*(.*)$", line)
            if match:
                idx = int(match.group(1))
                text = match.group(2).strip()
                if text:
                    corrected_map[idx] = text

        # Fallback if model omitted bracket tags but returned same line count
        if not corrected_map and len(raw_lines) == len(batch):
            for i, line in enumerate(raw_lines):
                corrected_map[i] = line

        # Rebuild segments with corrected text and synchronised word timings
        for i, seg in enumerate(batch):
            new_text = corrected_map.get(i, seg.text)
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

    return corrected_segments
