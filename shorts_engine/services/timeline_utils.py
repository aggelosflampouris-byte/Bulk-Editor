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

if TYPE_CHECKING:
    from services.transcriber import TranscriptionSegment

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Minimum silence gap between segments to qualify as a snap point (seconds).
_MIN_SILENCE_GAP: float = 0.25

# Maximum distance (seconds) we are willing to move a boundary when snapping.
_SNAP_RADIUS: float = 3.0


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
    Adjust AI-suggested clip boundaries to the nearest natural speech pause.

    A "snap point" is a gap between consecutive segments of at least
    _MIN_SILENCE_GAP seconds. For each boundary (start and end), we search
    within ±_SNAP_RADIUS seconds for the closest gap and snap to it.

    If no gap is found within the radius, the original boundary is kept.
    After snapping, the resulting duration is clamped to [min_dur, max_dur]
    by nudging the end point.

    Args:
        start_time: AI-suggested start in seconds.
        end_time:   AI-suggested end in seconds.
        segments:   Full source transcript segments (used to find silence gaps).
        min_dur:    Minimum acceptable clip duration (seconds).
        max_dur:    Maximum acceptable clip duration (seconds).

    Returns:
        (snapped_start, snapped_end) tuple.
    """
    if not segments:
        return start_time, end_time

    # Build list of silence gap midpoints between consecutive segments
    gap_points: list[float] = []
    for i in range(len(segments) - 1):
        gap_start = segments[i].end
        gap_end = segments[i + 1].start
        if gap_end - gap_start >= _MIN_SILENCE_GAP:
            # Snap to the end of the previous segment (clean cut after speech ends)
            gap_points.append(gap_start)

    def _nearest_gap(target: float) -> float:
        """Find the gap point closest to *target* within _SNAP_RADIUS."""
        best: float = target
        best_dist: float = _SNAP_RADIUS
        for gp in gap_points:
            dist = abs(gp - target)
            if dist < best_dist:
                best_dist = dist
                best = gp
        return best

    snapped_start = _nearest_gap(start_time)
    snapped_end = _nearest_gap(end_time)

    # Ensure start < end with minimum separation
    if snapped_end <= snapped_start + min_dur:
        snapped_end = snapped_start + min_dur

    # Clamp duration to [min_dur, max_dur]
    duration = snapped_end - snapped_start
    if duration > max_dur:
        snapped_end = snapped_start + max_dur
    elif duration < min_dur:
        snapped_end = snapped_start + min_dur

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
    from services.transcriber import TranscriptionSegment

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
