"""
services/highlight_scorer.py — Content analysis & viral 30–60s hook detection using Qwen 2.5.

Responsibilities:
  1. Read Greek timestamped speech segments from faster-whisper-large-v3.
  2. Score retention and identify viral 30–60s hooks using Qwen 2.5-32B/72B
     (via OpenAI-compatible endpoint like Ollama/vLLM/OpenRouter) or Gemini fallback.
  3. Extract the high-scoring cut segment with millisecond precision using FFmpeg.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

try:
    from services.transcriber import TranscriptionSegment
    from services.video_engine import FFmpegError
except ImportError:
    from shorts_engine.services.transcriber import TranscriptionSegment
    from shorts_engine.services.video_engine import FFmpegError

logger = logging.getLogger(__name__)

_DEFAULT_REQUEST_TIMEOUT = 45


@dataclass(frozen=True)
class HighlightCut:
    """Descriptor for a scored high-retention viral cut segment."""

    start_time: float
    end_time: float
    hook_text: str
    virality_score: float  # 0.0 to 10.0
    reasoning: str


_QWEN_PROMPT_TEMPLATE = """\
You are an expert short-form video producer specializing in Greek viral YouTube Shorts and TikToks.

Analyze the following Greek speech transcript with millisecond timestamps. Identify the single MOST VIRAL, \
ENGAGING, and HIGH-RETENTION continuous segment between {min_duration:.0f} and {max_duration:.0f} seconds long.

The segment MUST:
1. Start with an immediate, gripping hook or compelling question/statement.
2. Form a complete, coherent thought or insight.
3. Keep viewer attention high throughout.

Respond ONLY with a valid JSON object matching this schema:
{{
  "start_time": <float: start in seconds>,
  "end_time": <float: end in seconds>,
  "hook_text": "<string: Greek opening hook phrase>",
  "virality_score": <float: 1.0 to 10.0>,
  "reasoning": "<string: brief explanation in English why this hook retains viewers>"
}}

Timestamped transcript:
{transcript_lines}
"""


def _format_segments_for_scoring(segments: list[TranscriptionSegment]) -> str:
    lines: list[str] = []
    for s in segments:
        lines.append(f"[{s.start:.2f}s - {s.end:.2f}s] {s.text.strip()}")
    return "\n".join(lines)


def _call_qwen_api(
    prompt: str,
    api_base: str,
    api_key: str,
    model: str,
) -> Optional[str]:
    """Call an OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter) for Qwen 2.5."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a professional video editor. Output only JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=_DEFAULT_REQUEST_TIMEOUT)
        if not resp.ok:
            logger.warning("Qwen API returned status %d: %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("Qwen API call failed: %s", exc)
        return None


def score_highlight(
    segments: list[TranscriptionSegment],
    total_duration: float,
    api_base: str = "http://localhost:11434/v1",
    api_key: str = "",
    model: str = "qwen2.5:32b",
    gemini_api_key: str = "",
    min_duration: float = 30.0,
    max_duration: float = 60.0,
) -> HighlightCut:
    """
    Find the optimal viral 30–60s segment in a longer video.

    If video duration is already <= max_duration, returns the full video range.
    Otherwise, queries Qwen 2.5 (or Gemini fallback) to score and select the best cut.
    """
    if total_duration <= max_duration:
        hook = segments[0].text if segments else "Greek Short"
        return HighlightCut(
            start_time=0.0,
            end_time=total_duration,
            hook_text=hook,
            virality_score=8.5,
            reasoning="Clip is already within Shorts duration; keeping full video.",
        )

    formatted = _format_segments_for_scoring(segments)
    prompt = _QWEN_PROMPT_TEMPLATE.format(
        min_duration=min_duration,
        max_duration=max_duration,
        transcript_lines=formatted[:12000],  # keep context window safe
    )

    # 1. Attempt Qwen 2.5
    raw_json = _call_qwen_api(prompt, api_base, api_key, model)

    # 2. Fallback to Gemini if Qwen is unreachable
    if not raw_json and gemini_api_key:
        try:
            try:
                from services.seo_generator import _call_gemini_with_fallback
            except ImportError:
                from shorts_engine.services.seo_generator import _call_gemini_with_fallback
            from google import genai
            from google.genai import types as genai_types
            client = genai.Client(api_key=gemini_api_key)
            raw_json = _call_gemini_with_fallback(
                client=client,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=1024,
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
        except Exception as exc:
            logger.warning("Gemini highlight scoring fallback failed: %s", exc)

    # 3. Parse and validate JSON response
    if raw_json:
        try:
            cleaned = raw_json.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            data = json.loads(cleaned)

            start = float(data.get("start_time", 0.0))
            end = float(data.get("end_time", min(total_duration, start + max_duration)))
            score = float(data.get("virality_score", 7.0))
            hook = str(data.get("hook_text", ""))
            reason = str(data.get("reasoning", "Engaging hook and clear value proposition."))

            # Clamp boundaries
            start = max(0.0, min(start, total_duration - min_duration))
            end = min(total_duration, max(start + min_duration, end))

            logger.info("Highlight scored: [%.2fs - %.2fs] score=%.1f", start, end, score)
            return HighlightCut(
                start_time=start,
                end_time=end,
                hook_text=hook,
                virality_score=score,
                reasoning=reason,
            )
        except Exception as exc:
            logger.warning("Failed to parse highlight cut JSON: %s", exc)

    # Default heuristic fallback: take first 45 seconds
    default_end = min(total_duration, 45.0)
    hook = segments[0].text if segments else "Greek Short"
    return HighlightCut(
        start_time=0.0,
        end_time=default_end,
        hook_text=hook,
        virality_score=6.0,
        reasoning="Heuristic fallback to opening 45 seconds.",
    )


def extract_clip_segment(
    input_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
) -> Path:
    """Extract a sub-clip from *input_path* between *start_time* and *end_time*."""
    duration = max(1.0, end_time - start_time)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-i", str(input_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise FFmpegError(cmd, res.returncode, res.stderr)
    return output_path
