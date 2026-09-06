"""
services/transcriber.py — Greek speech transcription via faster-whisper.

Responsibilities:
  1. Transcribe a video file using the faster-whisper WhisperModel (CPU, int8).
  2. Return typed TranscriptionSegment objects.
  3. Render an ASS subtitle file with Greek-safe font styling.

This module has zero FFmpeg or HTTP dependencies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from config import ASS_HEADER_TEMPLATE, ASS_HIGHLIGHT_STYLE_LINE, ASS_STYLE_LINE

logger = logging.getLogger(__name__)


# ── Public Types ───────────────────────────────────────────────────────────────

class TranscriptionSegment:
    """
    Immutable typed container for a single whisper transcript segment.
    Uses __slots__ for memory efficiency when processing many segments.
    """

    __slots__ = ("start", "end", "text", "words")

    def __init__(
        self,
        start: float,
        end: float,
        text: str,
        words: Optional[list[tuple[float, float, str]]] = None,
    ) -> None:
        self.start: float = start
        self.end: float = end
        self.text: str = text.strip()
        # Each entry: (word_start, word_end, word_text)
        self.words: Optional[list[tuple[float, float, str]]] = words

    def __repr__(self) -> str:
        return f"TranscriptionSegment(start={self.start:.2f}, end={self.end:.2f}, text={self.text!r})"


# ── Core API ───────────────────────────────────────────────────────────────────

def transcribe(
    video_path: Path,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> list[TranscriptionSegment]:
    """
    Transcribe Greek speech from *video_path* using faster-whisper.

    The language is hard-forced to Greek ("el") to avoid the overhead of
    language detection on short clips and to maximise accuracy.

    Args:
        video_path:    Absolute path to the input video file.
        model_size:    Whisper model variant (tiny/base/small/medium/large-v3).
        device:        Compute device — always "cpu" for this deployment.
        compute_type:  Quantisation level — "int8" is optimal for CPU.

    Returns:
        Ordered list of TranscriptionSegment objects.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError:      If the WhisperModel fails to load or transcribe.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info("Loading WhisperModel (size=%s, device=%s)", model_size, device)
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load WhisperModel '{model_size}': {exc}"
        ) from exc

    logger.info("Transcribing '%s' (language=el)...", video_path.name)
    try:
        raw_segments, _info = model.transcribe(
            str(video_path),
            language="el",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Transcription failed for '{video_path.name}': {exc}"
        ) from exc

    segments: list[TranscriptionSegment] = []
    for seg in raw_segments:
        if not seg.text.strip():
            continue
        # Extract word-level timing when available
        word_data: Optional[list[tuple[float, float, str]]] = None
        if seg.words:
            word_data = [
                (w.start, w.end, w.word)
                for w in seg.words
                if w.word.strip()
            ]
        segments.append(
            TranscriptionSegment(seg.start, seg.end, seg.text, word_data)
        )

    logger.info("Transcription complete — %d segments extracted.", len(segments))
    return segments


def full_transcript_text(segments: list[TranscriptionSegment]) -> str:
    """
    Concatenate all segment texts into a single prose string.

    Used as input to the SEO generator and B-roll keyword extraction.
    """
    return " ".join(seg.text for seg in segments)


# ── ASS Subtitle Generation ────────────────────────────────────────────────────

def _seconds_to_ass_time(seconds: float) -> str:
    """
    Convert a float timestamp (seconds) to ASS time format: H:MM:SS.cc

    ASS centiseconds are two digits (hundredths of a second).
    """
    hours = int(seconds // 3600)
    remainder = seconds % 3600
    minutes = int(remainder // 60)
    secs = remainder % 60
    centiseconds = int(round((secs % 1) * 100))
    whole_secs = int(secs)
    return f"{hours}:{minutes:02d}:{whole_secs:02d}.{centiseconds:02d}"


def _escape_ass_text(text: str) -> str:
    """
    Escape characters that have special meaning in ASS dialogue lines.

    The ASS spec treats '{' as the start of an override tag block, and
    '\\n' / '\\N' as soft/hard line breaks. We escape bare braces to prevent
    accidental tag injection from transcript text.
    """
    # Replace curly braces with their ASS literal equivalents
    text = text.replace("{", r"\{").replace("}", r"\}")
    return text


def _build_karaoke_text(words: list[tuple[float, float, str]], seg_start: float) -> str:
    """
    Build an ASS karaoke text string with per-word {\\k} timing tags.

    Each word is prefixed with {\\rHighlight\\k<centiseconds>} so it renders
    in yellow while being spoken and returns to white afterwards.  Between
    words, {\\rDefault} resets the style to white.

    The {\\k} tag duration is the time the highlighted word is displayed
    (in centiseconds, i.e. hundredths of a second).

    Args:
        words:     List of (word_start, word_end, word_text) tuples.
        seg_start: Segment start time in seconds (used to compute relative offsets).

    Returns:
        ASS dialogue Text field string with inline karaoke override tags.
    """
    parts: list[str] = []
    for w_start, w_end, w_text in words:
        duration_cs = max(1, int(round((w_end - w_start) * 100)))
        safe = _escape_ass_text(w_text)
        # Switch to Highlight style for the active word, then reset to Default
        parts.append(f"{{\\rHighlight\\k{duration_cs}}}{safe}{{\\rDefault}}")
    return " ".join(parts)


def segments_to_ass(
    segments: list[TranscriptionSegment],
    style_line: Optional[str] = None,
    highlight_style_line: Optional[str] = None,
) -> str:
    """
    Render a full ASS subtitle file string from a list of segments.

    When word-level timing is available on a segment, each word is wrapped
    in karaoke {\\k} tags so the active word highlights yellow.
    Segments without word data fall back to plain white text.

    Args:
        segments:             Ordered list of TranscriptionSegment objects.
        style_line:           Override the default ASS style line.
        highlight_style_line: Override the highlight ASS style line.

    Returns:
        Complete ASS file content as a string, ready to be written to disk.
    """
    effective_style = style_line if style_line is not None else ASS_STYLE_LINE
    effective_highlight = (
        highlight_style_line
        if highlight_style_line is not None
        else ASS_HIGHLIGHT_STYLE_LINE
    )

    dialogue_lines: list[str] = []
    for seg in segments:
        start = _seconds_to_ass_time(seg.start)
        end = _seconds_to_ass_time(seg.end)

        if seg.words:
            # Word-level karaoke highlight
            text_field = _build_karaoke_text(seg.words, seg.start)
        else:
            # Fallback: plain text
            text_field = _escape_ass_text(seg.text)

        dialogue_lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text_field}"
        )

    return ASS_HEADER_TEMPLATE.format(
        style_line=effective_style,
        highlight_style_line=effective_highlight,
        dialogue_lines="\n".join(dialogue_lines),
    )


def write_ass_file(
    segments: list[TranscriptionSegment],
    output_path: Path,
    style_line: Optional[str] = None,
    highlight_style_line: Optional[str] = None,
) -> Path:
    """
    Generate and write an ASS subtitle file for the given segments.

    Args:
        segments:             Ordered list of TranscriptionSegment objects.
        output_path:          Destination path for the .ass file.
        style_line:           Optional style override (see segments_to_ass).
        highlight_style_line: Optional highlight style override.

    Returns:
        The resolved, written output_path.

    Raises:
        OSError: If the file cannot be written.
    """
    ass_content = segments_to_ass(segments, style_line, highlight_style_line)

    # ASS files must be UTF-8 encoded to preserve Greek glyphs
    output_path.write_text(ass_content, encoding="utf-8")
    logger.info("ASS subtitle file written to '%s'.", output_path)
    return output_path
