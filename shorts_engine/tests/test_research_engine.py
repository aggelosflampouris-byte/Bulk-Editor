"""
tests/test_research_engine.py — Unit tests for the Wide & Deep Research Engine.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from shorts_engine.services.research_engine import (
    ResearchDossier,
    conduct_wide_and_deep_research,
    fetch_google_news_rss,
)


def test_research_dossier_formatting() -> None:
    dossier = ResearchDossier(
        topic="Ελληνική Οικονομία 2026",
        summary="Συνοπτική εικόνα για τον πληθωρισμό και την ανάπτυξη.",
        key_statistics=["8.2% πληθωρισμός τροφίμων", "450 εκατ. ευρώ ενισχύσεις"],
        counter_arguments=["Αντιδράσεις για το κόστος ενέργειας"],
        official_statements=["Δήλωση Υπουργού για μέτρα στήριξης"],
        recent_headlines=["Πληθωρισμός: Νέα μέτρα ανακοίνωσε η κυβέρνηση"],
    )

    ctx = dossier.to_prompt_context()
    assert "Ελληνική Οικονομία 2026" in ctx
    assert "8.2% πληθωρισμός" in ctx
    assert "450 εκατ. ευρώ" in ctx
    assert "Αντίλογος" in ctx
    assert "Επίσημες Δηλώσεις" in ctx
    assert "Πρόσφατοι Τίτλοι Ειδήσεων" in ctx


def test_fetch_google_news_rss_mocked() -> None:
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Google News</title>
        <item>
          <title>Σημαντική εξέλιξη στην οικονομία - NewsIT</title>
          <pubDate>Fri, 25 Sep 2026 08:00:00 GMT</pubDate>
          <source url="https://www.newsit.gr">NewsIT</source>
        </item>
        <item>
          <title>Αντιδράσεις για τα νέα μέτρα - The TOC</title>
          <pubDate>Fri, 25 Sep 2026 09:30:00 GMT</pubDate>
          <source url="https://www.thetoc.gr">The TOC</source>
        </item>
      </channel>
    </rss>
    """

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = sample_xml

    with patch("httpx.Client.get", return_value=mock_resp):
        items = fetch_google_news_rss("οικονομία", max_items=2)
        assert len(items) == 2
        assert "Σημαντική εξέλιξη" in items[0]["title"]
        assert items[0]["source"] == "NewsIT"
        assert "Αντιδράσεις" in items[1]["title"]


def test_conduct_wide_and_deep_research_without_key() -> None:
    # When no Gemini key is provided, gracefully returns RSS-backed dossier
    with patch("shorts_engine.services.research_engine.fetch_google_news_rss") as mock_rss:
        mock_rss.return_value = [{"title": "Είδηση 1", "pub_date": "", "source": ""}]
        dossier = conduct_wide_and_deep_research(
            topic_title="Πολιτικές εξελίξεις",
            topic_context="Συζήτηση στη Βουλή",
            gemini_api_key=None,
        )
        assert isinstance(dossier, ResearchDossier)
        assert dossier.topic == "Πολιτικές εξελίξεις"
        assert len(dossier.recent_headlines) == 1


def test_conduct_wide_and_deep_research_with_gemini() -> None:
    mock_payload = {
        "summary": "Έντονη πολιτική αντιπαράθεση για το νομοσχέδιο.",
        "key_statistics": ["320 εκατ. ευρώ εξοικονόμηση", "15% αύξηση εισφορών"],
        "official_statements": ["Η κυβέρνηση δεσμεύεται για δημοσιονομική σταθερότητα"],
        "counter_arguments": ["Η αντιπολίτευση καταγγέλλει επιβάρυνση"],
        "timeline_events": ["Ψηφοφορία την προσεχή Τετάρτη"],
    }

    with (
        patch("shorts_engine.services.research_engine.fetch_google_news_rss", return_value=[]),
        patch("shorts_engine.services.research_engine.genai.Client"),
        patch("shorts_engine.services.research_engine._call_gemini_with_fallback", return_value=json.dumps(mock_payload)),
    ):
        dossier = conduct_wide_and_deep_research(
            topic_title="Νομοσχέδιο Βουλή",
            topic_context="Δήλωση Υπουργού",
            gemini_api_key="FAKE_KEY",
        )

        assert isinstance(dossier, ResearchDossier)
        assert len(dossier.key_statistics) == 2
        assert "320 εκατ. ευρώ" in dossier.key_statistics[0]
        assert len(dossier.counter_arguments) == 1
        assert "Ψηφοφορία" in dossier.timeline_events[0]
