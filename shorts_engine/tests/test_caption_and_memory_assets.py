"""
tests/test_caption_and_memory_assets.py — Tests for Caption Templates, Asset Library, Project Memory, and Automotive Niche.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from shorts_engine.services.caption_styles import (
    CAPTION_TEMPLATES,
    get_caption_style,
    list_caption_styles,
    recommend_caption_style,
)
from shorts_engine.services.asset_library import (
    ASSETS_DIR,
    MUSIC_DIR,
    SFX_DIR,
    find_assets_by_tags,
    initialize_asset_library,
    list_assets,
)
from shorts_engine.services.project_memory import (
    ProjectRecord,
    find_relevant_past_projects,
    get_memory_learning_context,
    list_project_records,
    save_project_record,
)
from shorts_engine.services.niche_templates import (
    auto_detect_niche,
    get_template,
    template_options,
)


# ── Caption Templates Tests ───────────────────────────────────────────────────

def test_caption_templates_registered():
    styles = list_caption_styles()
    assert len(styles) >= 7
    ids = [s.id for s in styles]
    assert "CAR_PULSE_INDUSTRIAL" in ids
    assert "HORMOZI_PUNCH" in ids
    assert "VIRAL_TIKTOK_BOUNCE" in ids
    assert "NEON_CYBER" in ids
    assert "DOCUMENTARY_CLEAN" in ids


def test_caption_style_ass_generation():
    car_style = get_caption_style("CAR_PULSE_INDUSTRIAL")
    base_line, hl_line = car_style.build_ass_styles(margin_v=480)
    assert "Style: Default" in base_line
    assert "Style: HighlightBox" in hl_line
    assert "480" in base_line
    assert car_style.primary_color in base_line


def test_recommend_caption_style_occasions():
    # Automotive / Car vlog
    rec_auto = recommend_caption_style(
        niche="automotive",
        topic_or_title="Αυτό το επικίνδυνο λάθος στον κινητήρα μπορεί να σου κοστίσει",
    )
    assert rec_auto.id == "CAR_PULSE_INDUSTRIAL"

    # Tech / AI
    rec_tech = recommend_caption_style(
        niche="technology",
        topic_or_title="Το νέο AI μοντέλο που αλλάζει τα πάντα",
    )
    assert rec_tech.id == "NEON_CYBER"

    # Politics / News
    rec_news = recommend_caption_style(
        niche="politics",
        topic_or_title="Η αλήθεια για τα κρυφά κόστη του προϋπολογισμού",
    )
    assert rec_news.id == "DOCUMENTARY_CLEAN"


# ── Asset Library Tests ───────────────────────────────────────────────────────

def test_asset_library_initialization_and_sfx_seeding():
    initialize_asset_library()
    assert SFX_DIR.is_dir()
    assert MUSIC_DIR.is_dir()

    sfx_items = list_assets("sfx")
    assert len(sfx_items) >= 4
    sfx_names = [s.name.lower() for s in sfx_items]
    assert any("whoosh" in n for n in sfx_names)
    assert any("ding" in n or "bell" in n for n in sfx_names)


def test_find_assets_by_tags():
    results = find_assets_by_tags(["whoosh"], category="sfx")
    assert len(results) > 0
    assert any("whoosh" in r.name.lower() for r in results)


# ── Project Memory Dataset Tests ──────────────────────────────────────────────

def test_project_memory_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    test_projects_dir = tmp_path / "projects"
    test_index_file = test_projects_dir / "memory_index.jsonl"
    monkeypatch.setattr("shorts_engine.services.project_memory.PROJECTS_DIR", test_projects_dir)
    monkeypatch.setattr("shorts_engine.services.project_memory.MEMORY_INDEX_FILE", test_index_file)

    initial_recs = list_project_records()
    assert len(initial_recs) >= 1  # Baseline seeds exist

    # Save a new project record
    rec = ProjectRecord(
        project_id="test_proj_car_diagnostics",
        created_at="2026-09-26T15:30:00Z",
        source_title="Πώς να ελέγξεις τα λάδια του αυτοκινήτου μόνος σου",
        source_url="https://youtube.com/watch?v=car_test_123",
        duration_seconds=44.0,
        niche="automotive",
        video_type="car_vlog",
        caption_style="CAR_PULSE_INDUSTRIAL",
        detected_scenes=["hood_open", "dipstick_check"],
        hook_summary="Quick DIY engine oil level test",
        hook_text="Μην κάνεις ποτέ αυτό το λάθος όταν αλλάζεις λάδια!",
        virality_score=9.2,
        tags=["αυτοκίνητο", "λάδια", "service"],
        status="approved",
        user_rating=5,
        user_notes="High retention when showing the dipstick immediately.",
    )
    save_project_record(rec)

    recs_after = list_project_records()
    saved = next((r for r in recs_after if r.project_id == "test_proj_car_diagnostics"), None)
    assert saved is not None
    assert saved.niche == "automotive"
    assert saved.caption_style == "CAR_PULSE_INDUSTRIAL"


def test_find_relevant_past_projects_and_prompt_context():
    matches = find_relevant_past_projects(
        title="Έλεγχος κινητήρα και φλάντζας",
        niche="automotive",
        text_sample="κινητήρας service αυτοκινήτου",
        top_k=2,
    )
    assert len(matches) > 0
    assert matches[0].niche == "automotive"

    prompt_context = get_memory_learning_context(
        title="Έλεγχος κινητήρα",
        niche="automotive",
    )
    assert "[AI MEMORY FROM PAST SUCCESSFUL PROJECTS IN THIS DOMAIN]" in prompt_context
    assert "CAR_PULSE_INDUSTRIAL" in prompt_context


# ── Niche Templates & Auto-Detection Tests ────────────────────────────────────

def test_automotive_niche_template():
    tpl = get_template("automotive")
    assert tpl.name == "automotive"
    assert tpl.caption_style == "CAR_PULSE_INDUSTRIAL"
    assert tpl.whisper_context_hint == "automotive"
    assert "κινητήρας" in tpl.brand_voice or "mechanic" in tpl.brand_voice.lower()

    opts = dict(template_options())
    assert "automotive" in opts


def test_auto_detect_niche():
    # Car Vlog / Automotive
    niche_car = auto_detect_niche(
        title="Αυτό το επικίνδυνο λάθος στον κινητήρα μπορεί να σου κοστίσει ακριβά",
        description="Ανακαλύψτε τι κρύβεται κάτω από το καπό και τη φλάντζα του μοτέρ.",
    )
    assert niche_car == "automotive"

    # Politics
    niche_pol = auto_detect_niche(
        title="Οι αποφάσεις της κυβέρνησης στη βουλή για τους νέους φόρους",
        description="Πολιτικές εξελίξεις και προϋπολογισμός.",
    )
    assert niche_pol == "politics"

    # Tech
    niche_tech = auto_detect_niche(
        title="Το νέο AI chatbot που γράφει software κώδικα σε δευτερόλεπτα",
        description="Τεχνολογική επανάσταση στην πληροφορική.",
    )
    assert niche_tech == "technology"
