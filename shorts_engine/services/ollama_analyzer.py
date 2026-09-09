"""
services/ollama_analyzer.py — Content analysis, SEO tags, chapters, and clip highlights via local Ollama.

Responsibilities:
  1. Communicate with local Ollama daemon (Qwen 2.5, Mistral, Llama 3).
  2. Prompt-engineer structured JSON extraction:
     - Target SEO Tags (comma-separated, keyword-focused).
     - Video Chapters (with precise start/end timestamps based on topic changes).
     - High-retention hooks for vertical clips/Shorts.
  3. Validate and parse output into typed dataclasses.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import ollama

try:
    from services.transcriber import TranscriptionSegment
    from services.youtube_transcript_fetcher import format_transcript_for_llm
except ImportError:
    from shorts_engine.services.transcriber import TranscriptionSegment
    from shorts_engine.services.youtube_transcript_fetcher import format_transcript_for_llm

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b")


class OllamaServiceError(Exception):
    """Raised when communication with Ollama daemon fails."""


class OllamaParseError(OllamaServiceError):
    """Raised when the LLM output cannot be parsed into expected JSON structure."""


@dataclass(frozen=True)
class VideoChapter:
    """A chapter marker within the video timeline."""
    title: str
    start_time: float
    end_time: float
    summary: str


@dataclass(frozen=True)
class HighRetentionHook:
    """A candidate vertical Short / viral highlight clip."""
    title: str
    start_time: float
    end_time: float
    hook_summary: str
    virality_reason: str

    @property
    def duration(self) -> float:
        return round(self.end_time - self.start_time, 2)


@dataclass(frozen=True)
class OllamaAnalysisResult:
    """Complete structured output from Ollama content analysis."""
    target_seo_tags: list[str]
    video_chapters: list[VideoChapter]
    high_retention_hooks: list[HighRetentionHook]
    model_name: str
    raw_payload: dict[str, Any] = field(default_factory=dict)


def _parse_timestamp(val: Any) -> float:
    """Convert float, int, or MM:SS/HH:MM:SS string to seconds."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        parts = val.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        try:
            return float(val)
        except ValueError:
            return 0.0
    return 0.0


def _clean_json_text(text: str) -> str:
    """Strip markdown code fence blocks if present."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def build_analysis_prompt(transcript_text: str) -> str:
    """Construct a rigorous prompt for structured JSON extraction."""
    return f"""You are an expert video content strategist, viral editor, and YouTube SEO specialist.
Analyze the following timestamped video transcript and output a single valid JSON object.

The JSON MUST have the following structure:
{{
  "target_seo_tags": ["tag 1", "tag 2", "tag 3", "tag 4", "tag 5"],
  "video_chapters": [
    {{
      "title": "Short chapter title",
      "start_time": 0.0,
      "end_time": 60.0,
      "summary": "Key discussion points in this chapter."
    }}
  ],
  "high_retention_hooks": [
    {{
      "title": "Catchy Short Title",
      "start_time": 15.0,
      "end_time": 55.0,
      "hook_summary": "Core point or emotional takeaway of the clip.",
      "virality_reason": "Why this 30-60s moment creates immediate retention and curiosity."
    }}
  ]
}}

Requirements:
1. "target_seo_tags": 10-15 keyword-focused tags relevant to the topic (Greek and English terms if Greek transcript).
2. "video_chapters": Logical chronological breakdown of the full video based on topic transitions.
3. "high_retention_hooks": At least 3 standalone 30-60 second vertical clips with high viral retention.
4. Output ONLY valid JSON without any additional text or commentary.

TRANSCRIPT:
{transcript_text}
"""


def analyze_transcript_with_ollama(
    transcript_data: Union[str, Sequence[TranscriptionSegment]],
    model: str = DEFAULT_OLLAMA_MODEL,
    host: Optional[str] = None,
) -> OllamaAnalysisResult:
    """
    Analyze a transcript using an Ollama LLM (Qwen 2.5, Mistral, Llama 3).

    Extracts:
      - Target SEO Tags (comma-separated, keyword-focused).
      - Video Chapters (with precise start/end timestamps based on topic changes).
      - High-retention hooks for vertical clips/Shorts.

    Args:
        transcript_data: Raw transcript string or sequence of TranscriptionSegments.
        model:           Ollama model tag (e.g. 'qwen2.5:32b', 'qwen2.5', 'mistral').
        host:            Ollama server URL (defaults to http://localhost:11434).

    Returns:
        Structured OllamaAnalysisResult dataclass.

    Raises:
        OllamaServiceError: On daemon connection or execution failure.
        OllamaParseError:   If JSON parsing or schema validation fails.
    """
    if isinstance(transcript_data, str):
        transcript_text = transcript_data
    else:
        transcript_text = format_transcript_for_llm(transcript_data)

    if not transcript_text.strip():
        raise ValueError("Transcript text cannot be empty.")

    resolved_host = host or DEFAULT_OLLAMA_HOST
    logger.info("Connecting to Ollama host '%s' using model '%s'...", resolved_host, model)

    client = ollama.Client(host=resolved_host)
    prompt = build_analysis_prompt(transcript_text)

    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional video SEO and viral clips analyzer. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": 0.3},
        )
        content = response.message.content or ""
    except Exception as exc:
        logger.error("Ollama API call failed on host '%s' (model: '%s'): %s", resolved_host, model, exc)
        raise OllamaServiceError(f"Ollama request failed: {exc}") from exc

    cleaned_json = _clean_json_text(content)
    try:
        data = json.loads(cleaned_json)
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode JSON from Ollama: %s\nRaw content: %s", exc, content)
        raise OllamaParseError(f"Invalid JSON returned by Ollama: {exc}") from exc

    # Parse SEO tags
    raw_tags = data.get("target_seo_tags", [])
    if isinstance(raw_tags, str):
        seo_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    elif isinstance(raw_tags, list):
        seo_tags = [str(t).strip() for t in raw_tags if str(t).strip()]
    else:
        seo_tags = []

    # Parse Video Chapters
    chapters: list[VideoChapter] = []
    for chap in data.get("video_chapters", []):
        if not isinstance(chap, dict):
            continue
        chapters.append(
            VideoChapter(
                title=str(chap.get("title", "Chapter")).strip(),
                start_time=_parse_timestamp(chap.get("start_time", 0.0)),
                end_time=_parse_timestamp(chap.get("end_time", 0.0)),
                summary=str(chap.get("summary", "")).strip(),
            )
        )

    # Parse High Retention Hooks (minimum 3 clips guaranteed if available)
    hooks: list[HighRetentionHook] = []
    for hook in data.get("high_retention_hooks", []):
        if not isinstance(hook, dict):
            continue
        st = _parse_timestamp(hook.get("start_time", 0.0))
        et = _parse_timestamp(hook.get("end_time", 0.0))
        hooks.append(
            HighRetentionHook(
                title=str(hook.get("title", "Highlight Hook")).strip(),
                start_time=st,
                end_time=et,
                hook_summary=str(hook.get("hook_summary", "")).strip(),
                virality_reason=str(hook.get("virality_reason", "")).strip(),
            )
        )

    return OllamaAnalysisResult(
        target_seo_tags=seo_tags,
        video_chapters=chapters,
        high_retention_hooks=hooks,
        model_name=model,
        raw_payload=data,
    )
