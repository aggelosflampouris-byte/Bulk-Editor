"""
services/timeline_utils.py — Timestamp manipulation utilities.

Responsibilities:
  1. Format a list of TranscriptionSegments into a timestamped text block
     for use as Gemini input.
  2. Snap AI-suggested clip boundaries to the nearest natural speech pause
     (silence gap ≥ 300ms between segments) while respecting duration limits.
  3. Slice a full segment list to a [start, end] window and re-base all
     timestamps to t=0 so subtitle files start at the beginning of the clip.

This module has zero FFmpeg, HTTP, or AI dependencies.
All functions are pure (no side effects) to allow easy unit testing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

try:
    from services.transcriber import TranscriptionSegment
except ImportError:
    from shorts_engine.services.transcriber import TranscriptionSegment

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Minimum silence gap between segments/words to qualify as a natural pause (seconds).
_MIN_SILENCE_GAP: float = 0.25

# Maximum distance (seconds) we are willing to move a boundary when snapping.
_SNAP_RADIUS: float = 3.0

# Pre-roll padding (seconds) before speech start to avoid cutting initial plosive/consonant.
_PRE_ROLL: float = 0.04

# Post-roll padding (seconds) after speech end to preserve natural vocal decay.
_POST_ROLL: float = 0.08

# Sentence terminating punctuation marks (Greek & Latin: period, exclamation, question mark ';', ellipsis)
_SENTENCE_TERMINATORS: tuple[str, ...] = (".", "!", "?", ";", "…", "...", "·")

# Clause separating punctuation marks (comma, colon, dash)
_CLAUSE_SEPARATORS: tuple[str, ...] = (",", ":", "-")


# ── Public API ─────────────────────────────────────────────────────────────────


def format_transcript_with_timestamps(
    segments: list["TranscriptionSegment"],
) -> str:
    """
    Format a segment list into a timestamped block suitable for Gemini input.

    Each line has the form:
        [MM:SS.s – MM:SS.s] segment text here

    Args:
        segments: Ordered list of TranscriptionSegment objects.

    Returns:
        Multi-line string with one segment per line, prefixed by timestamps.
    """
    lines: list[str] = []
    for seg in segments:
        start_str = _seconds_to_display(seg.start)
        end_str = _seconds_to_display(seg.end)
        lines.append(f"[{start_str} – {end_str}] {seg.text.strip()}")
    return "\n".join(lines)


def snap_to_silence(
    start_time: float,
    end_time: float,
    segments: list["TranscriptionSegment"],
    min_dur: float = 20.0,
    max_dur: float = 40.0,
) -> tuple[float, float]:
    """
    Adjust AI-suggested clip boundaries to natural speech pauses, word boundaries,
    and sentence punctuation.

    Key Rules:
      1. Word Boundary Protection: Never cut mid-word.
      2. Start Snapping: Snap to the beginning of speech (sentence/phrase start)
         with a 40ms pre-roll buffer to protect initial plosives.
      3. End Snapping: Snap after speech ends (sentence/phrase end) with an 80ms
         post-roll buffer to preserve natural vocal decay.
      4. Duration Clamping: Keep clip duration within [min_dur, max_dur].

    Args:
        start_time: AI-suggested start in seconds.
        end_time:   AI-suggested end in seconds.
        segments:   Full source transcript segments.
        min_dur:    Minimum acceptable clip duration (seconds).
        max_dur:    Maximum acceptable clip duration (seconds).

    Returns:
        (snapped_start, snapped_end) tuple.
    """
    if not segments:
        return start_time, end_time

    # Flatten all words or segment intervals
    all_words: list[tuple[float, float, str]] = []
    for seg in segments:
        if seg.words:
            all_words.extend(seg.words)
        else:
            all_words.append((seg.start, seg.end, seg.text))

    if not all_words:
        return start_time, end_time

    all_words.sort(key=lambda w: w[0])
    total_words = len(all_words)

    # ── Candidate Starts ───────────────────────────────────────────────────────
    # A clip start must begin at a word start with pre-roll, never cutting the previous word.
    candidate_starts: list[tuple[float, float]] = []  # (timestamp, quality_bonus)
    for i in range(total_words):
        w_start, w_end, w_text = all_words[i]
        prev_end = all_words[i - 1][1] if i > 0 else 0.0

        if prev_end < w_start:
            # Silence gap before speech
            t_cand = max(0.0, max(prev_end + 0.01, w_start - _PRE_ROLL))
        else:
            t_cand = max(0.0, w_start)

        # Quality scoring
        bonus = 0.0
        gap_before = w_start - prev_end if i > 0 else 1.0
        prev_text = all_words[i - 1][2].strip() if i > 0 else ""

        is_sentence_start = (
            i == 0
            or any(prev_text.endswith(p) for p in _SENTENCE_TERMINATORS)
            or gap_before >= _MIN_SILENCE_GAP
        )
        is_clause_start = (
            i > 0 and any(prev_text.endswith(p) for p in _CLAUSE_SEPARATORS)
        )

        if is_sentence_start:
            bonus += 0.8
        elif is_clause_start:
            bonus += 0.4

        if gap_before >= 0.15:
            bonus += 0.2

        candidate_starts.append((round(t_cand, 3), bonus))

    # ── Candidate Ends ─────────────────────────────────────────────────────────
    # A clip end must finish at a word end with post-roll, never cutting the next word.
    candidate_ends: list[tuple[float, float]] = []  # (timestamp, quality_bonus)
    for i in range(total_words):
        w_start, w_end, w_text = all_words[i]
        next_start = all_words[i + 1][0] if i + 1 < total_words else None

        if next_start is not None and next_start > w_end:
            t_cand = min(next_start - 0.01, w_end + _POST_ROLL)
        else:
            t_cand = w_end + _POST_ROLL

        bonus = 0.0
        gap_after = (next_start - w_end) if next_start is not None else 1.0
        cur_text = w_text.strip()

        is_sentence_end = (
            i == total_words - 1
            or any(cur_text.endswith(p) for p in _SENTENCE_TERMINATORS)
            or gap_after >= _MIN_SILENCE_GAP
        )
        is_clause_end = any(cur_text.endswith(p) for p in _CLAUSE_SEPARATORS)

        if is_sentence_end:
            bonus += 0.8
        elif is_clause_end:
            bonus += 0.4

        if gap_after >= 0.15:
            bonus += 0.2

        candidate_ends.append((round(t_cand, 3), bonus))

    # ── Find Best Start ────────────────────────────────────────────────────────
    best_start: float = start_time
    best_start_score: float = float("inf")
    for t_cand, bonus in candidate_starts:
        dist = abs(t_cand - start_time)
        if dist <= _SNAP_RADIUS:
            score = dist - bonus
            if score < best_start_score:
                best_start_score = score
                best_start = t_cand

    # ── Find Best End ──────────────────────────────────────────────────────────
    best_end: float = end_time
    best_end_score: float = float("inf")
    for t_cand, bonus in candidate_ends:
        dist = abs(t_cand - end_time)
        if dist <= _SNAP_RADIUS:
            score = dist - bonus
            if score < best_end_score:
                best_end_score = score
                best_end = t_cand

    snapped_start = best_start
    snapped_end = best_end

    # ── Word Boundary Protection ───────────────────────────────────────────────
    # Ensure neither cut point lands inside a word, or misses pre/post-roll
    for i, (w_s, w_e, _) in enumerate(all_words):
        if w_s <= snapped_start < w_e:
            prev_e = all_words[i - 1][1] if i > 0 else 0.0
            if prev_e < w_s:
                snapped_start = max(0.0, max(prev_e + 0.01, round(w_s - _PRE_ROLL, 3)))
            else:
                snapped_start = max(0.0, w_s)
            break

    for i, (w_s, w_e, _) in enumerate(all_words):
        if w_s < snapped_end <= w_e:
            next_s = all_words[i + 1][0] if i + 1 < total_words else float("inf")
            if next_s > w_e:
                snapped_end = min(next_s - 0.01, round(w_e + _POST_ROLL, 3))
            else:
                snapped_end = w_e
            break

    # ── Duration Clamping ──────────────────────────────────────────────────────
    duration = snapped_end - snapped_start

    if duration > max_dur:
        # Try to find a candidate end in [snapped_start + min_dur, snapped_start + max_dur]
        valid_ends = [
            (t, bonus)
            for t, bonus in candidate_ends
            if snapped_start + min_dur <= t <= snapped_start + max_dur
        ]
        if valid_ends:
            # Pick the one closest to snapped_start + max_dur with bonus
            best_e = min(
                valid_ends,
                key=lambda x: abs(x[0] - (snapped_start + max_dur)) - x[1],
            )
            snapped_end = best_e[0]
        else:
            snapped_end = round(snapped_start + max_dur, 3)

    elif duration < min_dur:
        valid_ends = [
            (t, bonus)
            for t, bonus in candidate_ends
            if snapped_start + min_dur <= t <= snapped_start + max_dur
        ]
        if valid_ends:
            best_e = min(
                valid_ends,
                key=lambda x: abs(x[0] - (snapped_start + min_dur)) - x[1],
            )
            snapped_end = best_e[0]
        else:
            snapped_end = round(snapped_start + min_dur, 3)

    # Re-check word collision on end boundary if clamped
    for i, (w_s, w_e, _) in enumerate(all_words):
        if w_s < snapped_end <= w_e:
            next_s = all_words[i + 1][0] if i + 1 < total_words else float("inf")
            if next_s > w_e:
                snapped_end = min(next_s - 0.01, round(w_e + _POST_ROLL, 3))
            else:
                snapped_end = w_e
            break

    logger.debug(
        "Boundary snap: [%.2f, %.2f] → [%.2f, %.2f] (dur=%.1fs)",
        start_time, end_time, snapped_start, snapped_end,
        snapped_end - snapped_start,
    )
    return snapped_start, snapped_end


def slice_segments(
    segments: list["TranscriptionSegment"],
    start_time: float,
    end_time: float,
) -> list["TranscriptionSegment"]:
    """
    Filter segments to those within [start_time, end_time] and re-base timestamps.

    A segment is included if it overlaps with the [start_time, end_time] window.
    All timestamps in the returned segments are shifted by -start_time so that
    the first segment starts near t=0, producing a valid stand-alone subtitle file.

    Args:
        segments:   Full source transcript segments.
        start_time: Clip start time in the source video (seconds).
        end_time:   Clip end time in the source video (seconds).

    Returns:
        A new list of TranscriptionSegment objects with re-based timestamps.
        Returns an empty list if no segments fall in the window.
    """
    # Import here to avoid circular imports; transcriber imports config
    try:
        from services.transcriber import TranscriptionSegment
    except ImportError:
        from shorts_engine.services.transcriber import TranscriptionSegment

    offset = start_time
    clipped: list[TranscriptionSegment] = []

    for seg in segments:
        # Include segment if it overlaps the window (not purely before or after)
        if seg.end <= start_time or seg.start >= end_time:
            continue

        # Re-base segment timestamps
        new_start = max(0.0, round(seg.start - offset, 3))
        new_end = max(new_start + 0.05, round(seg.end - offset, 3))

        # Re-base word-level timestamps if present
        new_words = None
        if seg.words:
            new_words = []
            for w_start, w_end, w_text in seg.words:
                # Only include words that fall within the clip window
                if w_end <= start_time or w_start >= end_time:
                    continue
                nws = max(0.0, round(w_start - offset, 3))
                nwe = max(nws + 0.01, round(w_end - offset, 3))
                new_words.append((nws, nwe, w_text))
            if not new_words:
                new_words = None

        clipped.append(
            TranscriptionSegment(
                start=new_start,
                end=new_end,
                text=seg.text,
                words=new_words,
            )
        )

    logger.debug(
        "Sliced %d segments from [%.2f, %.2f] → %d segments",
        len(segments), start_time, end_time, len(clipped),
    )
    return clipped


# ── Private Helpers ────────────────────────────────────────────────────────────


def _seconds_to_display(seconds: float) -> str:
    """
    Format a float number of seconds into MM:SS.s display format.

    Examples:
        0.0   → "00:00.0"
        90.5  → "01:30.5"
        3610.0 → "60:10.0"
    """
    total_whole = int(seconds)
    tenths = int(round((seconds % 1) * 10))
    minutes = total_whole // 60
    secs = total_whole % 60
    return f"{minutes:02d}:{secs:02d}.{tenths}"
