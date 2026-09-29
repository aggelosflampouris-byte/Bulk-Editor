"""
tests/test_seo_preflight.py — Unit tests for SEO preflight generation and pipeline overrides.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shorts_engine.services.autopilot import _apply_seo_override
from shorts_engine.services.seo_generator import SeoMetadata
from shorts_engine.services.seo_preflight import (
    PreflightSeoPackage,
    generate_ai_seo,
    get_template_defaults,
)


def test_get_template_defaults_politics():
    pkg = get_template_defaults("politics", topic="Εκλογές και Βουλή")
    assert pkg.niche == "politics"
    assert "Εκλογές και Βουλή" in pkg.title
    assert "Εκλογές και Βουλή" in pkg.description
    assert len(pkg.tags) >= 10
    assert pkg.source == "template"
    assert "Shorts" in pkg.tags_display


def test_get_template_defaults_automotive():
    pkg = get_template_defaults("automotive", topic="BMW M3 Test Drive")
    assert pkg.niche == "automotive"
    assert "BMW M3 Test Drive" in pkg.description
    assert any("car" in t.lower() or "αυτοκίνητο" in t.lower() for t in pkg.tags)


def test_get_template_defaults_fallback_custom():
    pkg = get_template_defaults("unknown_niche", topic="Viral Story")
    assert pkg.niche == "custom"
    assert "Viral Story" in pkg.description


def test_preflight_package_to_seo_metadata():
    pkg = PreflightSeoPackage(
        title="🔥 Μυστικό Σκάνδαλο",
        curiosity_title="🎯 Αυτό που δεν είδατε",
        authority_title="📣 Επίσημη Δήλωση",
        contrarian_title="⚡ Το λάθος όλων",
        description="Αναλυτικό video...",
        tags=["shorts", "news", "greece"],
        pinned_comment="Σχολιάστε παρακάτω!",
        primary_keyword="Σκάνδαλο",
        niche="politics",
        source="ai",
    )
    seo_meta = pkg.to_seo_metadata()
    assert isinstance(seo_meta, SeoMetadata)
    assert seo_meta.title == "🔥 Μυστικό Σκάνδαλο"
    assert seo_meta.curiosity_title == "🎯 Αυτό που δεν είδατε"
    assert seo_meta.tags == ("shorts", "news", "greece")
    assert seo_meta.pinned_comment == "Σχολιάστε παρακάτω!"
    assert seo_meta.primary_keyword == "Σκάνδαλο"


def test_apply_seo_override_none():
    base_seo = SeoMetadata(
        title="Original Title",
        description="Original Desc",
        tags=("tag1", "tag2"),
        primary_keyword="kw",
        pinned_comment="pinned",
        curiosity_title="cur",
        authority_title="auth",
        contrarian_title="contra",
    )
    res = _apply_seo_override(base_seo, None)
    assert res == base_seo


def test_apply_seo_override_with_custom_values():
    base_seo = SeoMetadata(
        title="Original Title",
        description="Original Desc",
        tags=("tag1", "tag2"),
        primary_keyword="kw",
        pinned_comment="pinned",
        curiosity_title="cur",
        authority_title="auth",
        contrarian_title="contra",
    )
    override_pkg = PreflightSeoPackage(
        title="Custom Overridden Title",
        description="Custom Desc",
        tags=["custom1", "custom2", "custom3"],
        pinned_comment="Custom Pin",
        primary_keyword="CustomKW",
        curiosity_title="CustomCuriosity",
    )
    res = _apply_seo_override(base_seo, override_pkg)
    assert res.title == "Custom Overridden Title"
    assert res.description == "Custom Desc"
    assert res.tags == ("custom1", "custom2", "custom3")
    assert res.pinned_comment == "Custom Pin"
    assert res.primary_keyword == "CustomKW"
    assert res.curiosity_title == "CustomCuriosity"
    assert res.authority_title == "auth"  # Preserves fallback when override is empty


def test_generate_ai_seo_mocked():
    mock_json = """{
        "title": "🔥 Τι κρύβουν για τον προϋπολογισμό",
        "curiosity_title": "🎯 Το μυστικό του προϋπολογισμού",
        "authority_title": "📣 Ανάλυση Οικονομολόγου",
        "contrarian_title": "⚡ Γιατί όλοι κάνουν λάθος",
        "primary_keyword": "προϋπολογισμός",
        "description": "Πλήρης ανάλυση για τον προϋπολογισμό.",
        "pinned_comment": "Ποια είναι η γνώμη σας;",
        "tags": ["οικονομία", "προϋπολογισμός", "shorts", "greece", "dianisma"]
    }"""

    with patch("shorts_engine.services.seo_preflight._call_gemini_with_fallback", return_value=mock_json):
        pkg = generate_ai_seo(
            gemini_api_key="fake-key",
            niche="economics",
            source_title="Προϋπολογισμός 2026",
            source_url="https://youtube.com/watch?v=123",
        )
        assert pkg.source == "ai"
        assert pkg.title == "🔥 Τι κρύβουν για τον προϋπολογισμό"
        assert pkg.curiosity_title == "🎯 Το μυστικό του προϋπολογισμού"
        assert len(pkg.tags) >= 8
        assert pkg.primary_keyword == "προϋπολογισμός"
