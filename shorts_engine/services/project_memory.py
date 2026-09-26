"""
services/project_memory.py — Project Memory Dataset & Continuous Learning System.

Stores and queries past processed video projects in a structured local dataset:
  - shorts_engine/projects/memory_index.jsonl
  - shorts_engine/projects/{project_id}.json

Enables the AI to "remember" past successful projects, what hook styles,
video types, caption templates, and clip durations worked best for specific
occasions (such as Car Vlogs, Tech Reviews, Politics, etc.), and injects this
few-shot contextual memory into Gemini prompts for future videos.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Projects storage directory under shorts_engine/projects
_ROOT_DIR = Path(__file__).parent.parent.resolve()
PROJECTS_DIR = _ROOT_DIR / "projects"
MEMORY_INDEX_FILE = PROJECTS_DIR / "memory_index.jsonl"


@dataclass
class ProjectRecord:
    """
    Structured record representing a completed or approved video project in memory.
    """

    project_id: str
    created_at: str
    source_title: str
    source_url: str
    duration_seconds: float
    niche: str                          # e.g. "automotive", "politics", "technology"
    video_type: str                     # e.g. "car_vlog", "mechanic_diy", "commentary"
    caption_style: str                  # e.g. "CAR_PULSE_INDUSTRIAL", "HORMOZI_PUNCH"
    detected_scenes: list[str] = field(default_factory=list)
    hook_summary: str = ""
    hook_text: str = ""
    virality_score: float = 8.5
    tags: list[str] = field(default_factory=list)
    status: str = "approved"            # "approved" | "scheduled" | "draft" | "rejected"
    user_rating: int = 5                # 1 to 5 scale
    user_notes: str = ""
    output_path: str = ""


def _initialize_memory_storage() -> None:
    """Ensure projects directory exists and seed initial baseline memory if empty."""
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_INDEX_FILE.exists() or MEMORY_INDEX_FILE.stat().st_size == 0:
        _seed_baseline_memory_dataset()


def _seed_baseline_memory_dataset() -> None:
    """
    Seed initial memory records into memory_index.jsonl to provide rich
    few-shot contextual learning right away (including car vlog & tech).
    """
    baseline_records = [
        ProjectRecord(
            project_id="proj_car_vlog_engine_repair",
            created_at=datetime.now(timezone.utc).isoformat(),
            source_title="Αυτό το επικίνδυνο λάθος στον κινητήρα μπορεί να σου κοστίσει ακριβά",
            source_url="https://m.youtube.com/watch?v=ae2MXs-m4I2",
            duration_seconds=42.0,
            niche="automotive",
            video_type="mechanic_diy_car_vlog",
            caption_style="CAR_PULSE_INDUSTRIAL",
            detected_scenes=["engine_bay_inspection", "cylinder_head_gasket", "preventive_maintenance"],
            hook_summary="Reveals dangerous engine maintenance mistake with open hood diagnostic",
            hook_text="Αν ακούσεις αυτόν τον θόρυβο στον κινητήρα, σβήσε το αμέσως!",
            virality_score=9.4,
            tags=["κινητήρας", "αυτοκίνητο", "service", "φλάντζα", "μηχανικός", "car_vlog"],
            status="approved",
            user_rating=5,
            user_notes="High retention when opening directly on the engine bay and highlighting the diagnostic warning with CAR_PULSE_INDUSTRIAL captions.",
        ),
        ProjectRecord(
            project_id="proj_politics_budget_breakdown",
            created_at=datetime.now(timezone.utc).isoformat(),
            source_title="Η αλήθεια για τα κρυφά κόστη του νέου προϋπολογισμού",
            source_url="https://youtube.com/watch?v=pol_budget_2026",
            duration_seconds=48.0,
            niche="politics",
            video_type="investigative_commentary",
            caption_style="DOCUMENTARY_CLEAN",
            detected_scenes=["speaker_authoritative", "economic_graph_overlay", "parliament_broll"],
            hook_summary="Direct exposé on hidden public charges with document backing",
            hook_text="Αυτό που δεν σου είπαν για τις αυξήσεις στα τιμολόγια.",
            virality_score=9.1,
            tags=["οικονομία", "προϋπολογισμός", "πολιτική", "ακρίβεια", "φόροι"],
            status="approved",
            user_rating=5,
            user_notes="Clear, understated documentary captions ensure high credibility on investigative pieces.",
        ),
        ProjectRecord(
            project_id="proj_tech_ai_breakthrough",
            created_at=datetime.now(timezone.utc).isoformat(),
            source_title="Το νέο AI μοντέλο που αλλάζει τα πάντα στο coding",
            source_url="https://youtube.com/watch?v=tech_ai_2026",
            duration_seconds=38.0,
            niche="technology",
            video_type="tech_review_demo",
            caption_style="NEON_CYBER",
            detected_scenes=["screen_demo", "speaker_reaction", "speed_benchmark"],
            hook_summary="Side-by-side benchmark proving 10x coding productivity gain",
            hook_text="Μην ξαναγράψεις κώδικα με τον παλιό τρόπο.",
            virality_score=9.3,
            tags=["AI", "τεχνολογία", "software", "προγραμματισμός", "future"],
            status="approved",
            user_rating=5,
            user_notes="Neon cyber captions and snappy cut pacing worked exceptionally well for tech audience.",
        ),
    ]

    with MEMORY_INDEX_FILE.open("w", encoding="utf-8") as fh:
        for rec in baseline_records:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            # Write individual project file as well
            p_file = PROJECTS_DIR / f"{rec.project_id}.json"
            p_file.write_text(json.dumps(asdict(rec), ensure_ascii=False, indent=2), encoding="utf-8")


def save_project_record(record: ProjectRecord) -> Path:
    """
    Save a project record to both the individual JSON file and the append-only JSONL index.
    """
    _initialize_memory_storage()
    p_file = PROJECTS_DIR / f"{record.project_id}.json"
    p_file.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")

    # Append to memory_index.jsonl
    with MEMORY_INDEX_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    logger.info("Project record saved to memory: %s ('%s')", record.project_id, record.source_title)
    return p_file


def list_project_records(limit: int = 50) -> list[ProjectRecord]:
    """
    List past projects recorded in memory in reverse chronological order.
    """
    _initialize_memory_storage()
    if not MEMORY_INDEX_FILE.exists():
        return []

    records: list[ProjectRecord] = []
    seen_ids: set[str] = set()

    with MEMORY_INDEX_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                rec = ProjectRecord(**data)
                if rec.project_id not in seen_ids:
                    seen_ids.add(rec.project_id)
                    records.append(rec)
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.debug("Skipping malformed memory record: %s", exc)

    records.sort(key=lambda r: r.created_at, reverse=True)
    return records[:limit]


def find_relevant_past_projects(
    title: str,
    niche: str,
    text_sample: str = "",
    top_k: int = 3,
) -> list[ProjectRecord]:
    """
    Retrieve past projects from the memory dataset most relevant to the current video.
    Uses token overlap, niche match, and semantic tag intersection.
    """
    all_records = list_project_records(limit=100)
    if not all_records:
        return []

    query_tokens = set(f"{title} {niche} {text_sample}".lower().split())

    scored: list[tuple[float, ProjectRecord]] = []
    for rec in all_records:
        score = 0.0
        # Exact niche match bonus
        if rec.niche.lower() == niche.lower():
            score += 5.0

        # Title & hook tokens overlap
        rec_tokens = set(f"{rec.source_title} {rec.hook_text} {' '.join(rec.tags)} {rec.user_notes}".lower().split())
        overlap = query_tokens & rec_tokens
        score += len(overlap) * 1.5

        # Approval and high rating multiplier
        if rec.status == "approved":
            score += 2.0
        score += (rec.user_rating - 3) * 0.5

        if score > 0:
            scored.append((score, rec))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [rec for _, rec in scored[:top_k]]


def get_memory_learning_context(
    title: str,
    niche: str,
    text_sample: str = "",
) -> str:
    """
    Format a concise, high-value few-shot memory prompt based on past projects
    to inject into Gemini for clip selection, hook design, and SEO.
    """
    matches = find_relevant_past_projects(title=title, niche=niche, text_sample=text_sample, top_k=2)
    if not matches:
        return ""

    lines = ["\n[AI MEMORY FROM PAST SUCCESSFUL PROJECTS IN THIS DOMAIN]:"]
    for idx, m in enumerate(matches, start=1):
        lines.append(
            f"- Project '{m.source_title}' ({m.niche.upper()}): "
            f"Best hook style: '{m.hook_text}'. Caption template: '{m.caption_style}'. "
            f"Key takeaway: {m.user_notes or m.hook_summary}"
        )
    lines.append(
        "Apply these proven structural patterns: emulate the high-performing hook structure and "
        "ensure strong focal continuity tailored to this genre."
    )
    return "\n".join(lines)


def update_project_status(
    project_id: str,
    status: str,
    user_rating: int | None = None,
    user_notes: str | None = None,
) -> bool:
    """
    Update status, rating, or feedback notes for a specific project.
    """
    _initialize_memory_storage()
    p_file = PROJECTS_DIR / f"{project_id}.json"
    if not p_file.exists():
        return False

    try:
        data = json.loads(p_file.read_text(encoding="utf-8"))
        data["status"] = status
        if user_rating is not None:
            data["user_rating"] = user_rating
        if user_notes is not None:
            data["user_notes"] = user_notes

        p_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        # Update index
        records = list_project_records(limit=200)
        with MEMORY_INDEX_FILE.open("w", encoding="utf-8") as fh:
            for r in records:
                if r.project_id == project_id:
                    r.status = status
                    if user_rating is not None:
                        r.user_rating = user_rating
                    if user_notes is not None:
                        r.user_notes = user_notes
                fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        return True
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to update project %s: %s", project_id, exc)
        return False
