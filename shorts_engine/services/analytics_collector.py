"""
services/analytics_collector.py — Analytics Data Aggregator.

Collects all available data from the project memory, traffic scheduler,
and channel analyzer and composes a structured analytics context object
that is injected into the AI Analyst chatbot as ground-truth knowledge.

Design:
  - Zero external network calls at collection time — all data is local.
  - Serializable to a Markdown summary for prompt injection.
  - Strict typing throughout for safe downstream consumption.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VideoPerformanceRecord:
    """Aggregated stats for a single produced video project."""
    project_id: str
    title: str
    niche: str
    caption_style: str
    hook_text: str
    virality_score: float
    user_rating: int
    duration_seconds: float
    status: str
    created_at: str
    tags: list[str] = field(default_factory=list)
    user_notes: str = ""


@dataclass
class AnalyticsContext:
    """Complete analytics snapshot passed to the AI Analyst chatbot."""
    collected_at: str
    total_projects: int
    approved_projects: int
    avg_virality_score: float
    avg_user_rating: float
    top_niches: list[tuple[str, int]]             # [(niche, count)]
    top_caption_styles: list[tuple[str, int]]     # [(style, count)]
    recent_videos: list[VideoPerformanceRecord]   # Last 20 by created_at
    high_rated_videos: list[VideoPerformanceRecord]  # rating >= 4
    low_rated_videos: list[VideoPerformanceRecord]   # rating <= 2
    scheduled_upload_slots: list[str]             # Human-readable slot strings
    niche_breakdown: dict[str, dict[str, Any]]    # niche → {avg_virality, avg_rating, count}

    def to_markdown_summary(self) -> str:
        """Render the analytics context as a structured Markdown summary for prompt injection."""
        lines: list[str] = [
            f"# Greek Shorts Engine — Analytics Snapshot",
            f"**Collected at:** {self.collected_at}",
            "",
            "## Key Performance Indicators",
            f"- **Total projects produced:** {self.total_projects}",
            f"- **Approved / Published:** {self.approved_projects}",
            f"- **Average virality score:** {self.avg_virality_score:.2f} / 10",
            f"- **Average user rating:** {self.avg_user_rating:.2f} / 5",
            "",
        ]

        if self.top_niches:
            lines.append("## Top Niches by Volume")
            for niche, count in self.top_niches[:8]:
                breakdown = self.niche_breakdown.get(niche, {})
                avg_v = breakdown.get("avg_virality", 0.0)
                avg_r = breakdown.get("avg_rating", 0.0)
                lines.append(
                    f"- **{niche}** — {count} videos | avg virality {avg_v:.1f} | avg rating {avg_r:.1f}"
                )
            lines.append("")

        if self.top_caption_styles:
            lines.append("## Caption Style Usage")
            for style, count in self.top_caption_styles[:6]:
                lines.append(f"- `{style}`: {count} videos")
            lines.append("")

        if self.high_rated_videos:
            lines.append("## Best-Performing Videos (rating ≥ 4)")
            for v in self.high_rated_videos[:10]:
                lines.append(
                    f"- ⭐ {v.title[:70]} | niche={v.niche} | "
                    f"virality={v.virality_score:.1f} | rating={v.user_rating}/5 | "
                    f"hook: \"{v.hook_text[:60]}\""
                )
                if v.user_notes:
                    lines.append(f"  - Notes: {v.user_notes[:120]}")
            lines.append("")

        if self.low_rated_videos:
            lines.append("## Under-Performing Videos (rating ≤ 2)")
            for v in self.low_rated_videos[:5]:
                lines.append(
                    f"- ❌ {v.title[:70]} | niche={v.niche} | "
                    f"virality={v.virality_score:.1f} | rating={v.user_rating}/5"
                )
                if v.user_notes:
                    lines.append(f"  - Notes: {v.user_notes[:120]}")
            lines.append("")

        if self.recent_videos:
            lines.append("## 10 Most Recent Videos")
            for v in self.recent_videos[:10]:
                lines.append(
                    f"- [{v.created_at[:10]}] {v.title[:60]} | niche={v.niche} | "
                    f"style={v.caption_style} | virality={v.virality_score:.1f}"
                )
            lines.append("")

        if self.scheduled_upload_slots:
            lines.append("## Next Optimal Greek Upload Slots")
            for slot in self.scheduled_upload_slots[:5]:
                lines.append(f"- {slot}")
            lines.append("")

        lines.append(
            "---\n"
            "*This data is the complete ground truth for the Greek Shorts Engine running "
            "for channel @DianismaNews. Use it to give evidence-based recommendations.*"
        )
        return "\n".join(lines)


def collect_analytics(max_projects: int = 200) -> AnalyticsContext:
    """
    Aggregate all available local analytics data into a single AnalyticsContext.

    Reads from:
      - project_memory (memory_index.jsonl)
      - traffic_scheduler (next upload slots)

    Returns:
        AnalyticsContext with structured performance statistics.
    """
    try:
        from services.project_memory import list_project_records
    except ImportError:
        from shorts_engine.services.project_memory import list_project_records

    try:
        from services.traffic_scheduler import get_greek_high_traffic_slots
    except ImportError:
        from shorts_engine.services.traffic_scheduler import get_greek_high_traffic_slots

    raw_records = list_project_records(limit=max_projects)

    performance_records: list[VideoPerformanceRecord] = [
        VideoPerformanceRecord(
            project_id=r.project_id,
            title=r.source_title,
            niche=r.niche,
            caption_style=r.caption_style,
            hook_text=r.hook_text,
            virality_score=r.virality_score,
            user_rating=r.user_rating,
            duration_seconds=r.duration_seconds,
            status=r.status,
            created_at=r.created_at,
            tags=r.tags,
            user_notes=r.user_notes,
        )
        for r in raw_records
    ]

    total = len(performance_records)
    approved = sum(1 for v in performance_records if v.status == "approved")
    avg_virality = (
        sum(v.virality_score for v in performance_records) / total if total else 0.0
    )
    avg_rating = (
        sum(v.user_rating for v in performance_records) / total if total else 0.0
    )

    # Niche breakdown
    niche_counts: dict[str, int] = {}
    niche_virality: dict[str, list[float]] = {}
    niche_ratings: dict[str, list[int]] = {}
    for v in performance_records:
        niche_counts[v.niche] = niche_counts.get(v.niche, 0) + 1
        niche_virality.setdefault(v.niche, []).append(v.virality_score)
        niche_ratings.setdefault(v.niche, []).append(v.user_rating)

    top_niches = sorted(niche_counts.items(), key=lambda x: x[1], reverse=True)
    niche_breakdown: dict[str, dict[str, Any]] = {
        niche: {
            "count": niche_counts[niche],
            "avg_virality": round(
                sum(niche_virality[niche]) / len(niche_virality[niche]), 2
            ),
            "avg_rating": round(
                sum(niche_ratings[niche]) / len(niche_ratings[niche]), 2
            ),
        }
        for niche in niche_counts
    }

    # Caption style breakdown
    style_counts: dict[str, int] = {}
    for v in performance_records:
        style_counts[v.caption_style] = style_counts.get(v.caption_style, 0) + 1
    top_caption_styles = sorted(style_counts.items(), key=lambda x: x[1], reverse=True)

    # Sorted segments
    recent = sorted(performance_records, key=lambda v: v.created_at, reverse=True)
    high_rated = sorted(
        [v for v in performance_records if v.user_rating >= 4],
        key=lambda v: (v.user_rating, v.virality_score),
        reverse=True,
    )
    low_rated = sorted(
        [v for v in performance_records if v.user_rating <= 2],
        key=lambda v: v.user_rating,
    )

    # Upload slots
    try:
        slots = get_greek_high_traffic_slots(count=7)
        slot_strings = [s.display_str for s in slots]
    except Exception as exc:
        logger.warning("Could not fetch upload slots: %s", exc)
        slot_strings = []

    return AnalyticsContext(
        collected_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        total_projects=total,
        approved_projects=approved,
        avg_virality_score=round(avg_virality, 2),
        avg_user_rating=round(avg_rating, 2),
        top_niches=top_niches,
        top_caption_styles=top_caption_styles,
        recent_videos=recent[:20],
        high_rated_videos=high_rated,
        low_rated_videos=low_rated,
        scheduled_upload_slots=slot_strings,
        niche_breakdown=niche_breakdown,
    )
