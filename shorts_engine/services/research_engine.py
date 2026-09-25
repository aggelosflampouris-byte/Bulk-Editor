"""
services/research_engine.py — Wide & Deep Fact-Checking & Context Research Engine.

Performs investigative research for @DianismaNews YouTube Shorts by:
  1. Wide Retrieval: Pulling fresh real-time Greek news coverage, headlines,
     and timeline developments via Google News RSS (zero API quota bottlenecks).
  2. Deep Synthesis: Using Gemini with fallback models to extract concrete verified statistics,
     financial amounts, legislation, official quotes, and opposing viewpoints.
  3. Structured Dossier: Providing structured research context for script generators
     to anchor commentary in hard facts rather than generic filler.
"""

from __future__ import annotations

import html
import json
import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import httpx
from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError

try:
    from services.seo_generator import _call_gemini_with_fallback
except ImportError:
    from shorts_engine.services.seo_generator import _call_gemini_with_fallback

logger = logging.getLogger(__name__)


@dataclass
class ResearchDossier:
    topic: str
    summary: str
    key_statistics: list[str] = field(default_factory=list)
    official_statements: list[str] = field(default_factory=list)
    counter_arguments: list[str] = field(default_factory=list)
    timeline_events: list[str] = field(default_factory=list)
    recent_headlines: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Render a dense factual research brief formatted for LLM script prompts."""
        lines: list[str] = ["=== ΕΡΕΥΝΑ & ΕΠΑΛΗΘΕΥΜΕΝΑ ΔΕΔΟΜΕΝΑ (@DianismaNews) ==="]
        lines.append(f"Θέμα: {self.topic}")
        if self.summary:
            lines.append(f"Σύνοψη Γεγονότων: {self.summary}")

        if self.key_statistics:
            lines.append("Συγκεκριμένα Στοιχεία & Αριθμοί:")
            for s in self.key_statistics:
                lines.append(f"  • {s}")

        if self.counter_arguments:
            lines.append("Αντίλογος & Αντιδράσεις:")
            for c in self.counter_arguments:
                lines.append(f"  • {c}")

        if self.official_statements:
            lines.append("Επίσημες Δηλώσεις / Θέσεις:")
            for st in self.official_statements:
                lines.append(f"  • {st}")

        if self.recent_headlines:
            lines.append("Πρόσφατοι Τίτλοι Ειδήσεων:")
            for h in self.recent_headlines[:4]:
                lines.append(f"  • {h}")

        return "\n".join(lines)


def fetch_google_news_rss(query: str, max_items: int = 6) -> list[dict[str, str]]:
    """
    Fetch real-time Greek news items from Google News RSS.
    Requires zero external API keys and provides fresh breaking coverage.
    """
    cleaned_q = query.strip()
    if not cleaned_q:
        return []

    encoded = urllib.parse.quote(cleaned_q)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=el&gl=GR&ceid=GR:el"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        with httpx.Client(timeout=6.0, follow_redirects=True, headers=headers) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                logger.warning("Google News RSS returned status %d for '%s'", resp.status_code, cleaned_q)
                return []

            root = ET.fromstring(resp.text)
            items: list[dict[str, str]] = []
            for item_elem in root.findall(".//item")[:max_items]:
                title_elem = item_elem.find("title")
                pub_elem = item_elem.find("pubDate")
                source_elem = item_elem.find("source")

                raw_title = title_elem.text if title_elem is not None and title_elem.text else ""
                clean_title = html.unescape(raw_title).strip()
                pub_date = pub_elem.text.strip() if pub_elem is not None and pub_elem.text else ""
                source_name = source_elem.text.strip() if source_elem is not None and source_elem.text else ""

                if clean_title:
                    items.append({
                        "title": clean_title,
                        "pub_date": pub_date,
                        "source": source_name,
                    })
            return items
    except (httpx.HTTPError, ET.ParseError, OSError, ValueError, RuntimeError) as exc:
        logger.warning("Google News RSS retrieval failed for '%s': %s", cleaned_q, exc)
        return []


def clean_json_markdown(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


_DEEP_RESEARCH_SYNTHESIS_PROMPT = """\
You are the Chief Investigative Researcher for @DianismaNews (Greek political & socioeconomic YouTube channel).
Your objective is to provide a DEEP, FACTUAL, DATA-BACKED research brief on the following topic and speaker statement.

TOPIC / TITLE:
{topic_title}

SPEAKER STATEMENT / CONTEXT:
{topic_context}

LIVE MEDIA HEADLINES & RECENT REPORTING:
{recent_news}

YOUR TASK:
Synthesize this into an authoritative investigative research dossier in Greek.
Focus on:
1. Hard numbers, percentages, budget amounts, dates, or official survey stats.
2. The core debate: What did the speaker argue VS what do opposing experts, official documents, or the public say?
3. Uncover the underlying systemic conflict or "elephant in the room".

Return ONLY a valid JSON object matching this schema:
{{
  "summary": "2-3 sentences explaining what happened and why it matters in Greek",
  "key_statistics": [
    "Concrete statistic or financial figure with number and context",
    "Another concrete fact or date"
  ],
  "official_statements": [
    "Core claim or official argument"
  ],
  "counter_arguments": [
    "Opposing view, critique, or contradictory evidence"
  ],
  "timeline_events": [
    "Recent key development or background date"
  ]
}}
"""


def conduct_wide_and_deep_research(
    topic_title: str,
    topic_context: str = "",
    gemini_api_key: str | None = None,
) -> ResearchDossier:
    """
    Perform wide retrieval across Greek news and deep Gemini synthesis to assemble
    a verified fact-checking research dossier.
    """
    cleaned_title = topic_title.strip()
    logger.info("[Research Engine] Initiating wide & deep research for '%s'...", cleaned_title[:60])

    # 1. Wide Retrieval: Pull multiple search queries from the title
    # Query 1: Full title keywords
    words = re.findall(r"\b[\wΆ-ώ]{4,}\b", cleaned_title, re.UNICODE)
    primary_query = " ".join(words[:4]) if words else cleaned_title
    
    # Query 2: Core entities / topic
    news_items = fetch_google_news_rss(primary_query, max_items=5)
    if len(news_items) < 3 and len(words) >= 2:
        alt_query = " ".join(words[:2])
        news_items.extend(fetch_google_news_rss(alt_query, max_items=4))

    # Deduplicate headlines
    seen_titles: set[str] = set()
    unique_headlines: list[str] = []
    for it in news_items:
        t = it.get("title", "")
        if t and t not in seen_titles:
            seen_titles.add(t)
            unique_headlines.append(t)

    headlines_block = "\n".join(f"- {h}" for h in unique_headlines[:6]) if unique_headlines else "(Δεν βρέθηκαν πρόσφατοι εξωτερικοί τίτλοι)"

    # If no Gemini API key provided, produce a rule-based dossier
    if not gemini_api_key:
        logger.info("[Research Engine] No Gemini key provided — returning RSS headline dossier.")
        return ResearchDossier(
            topic=cleaned_title,
            summary=f"Επικαιρότητα και εξελίξεις γύρω από το θέμα '{cleaned_title}'.",
            key_statistics=[],
            official_statements=[topic_context[:200]] if topic_context else [],
            counter_arguments=[],
            timeline_events=[],
            recent_headlines=unique_headlines[:5],
        )

    # 2. Deep Synthesis via Gemini
    try:
        client = genai.Client(api_key=gemini_api_key)
        prompt = _DEEP_RESEARCH_SYNTHESIS_PROMPT.format(
            topic_title=cleaned_title,
            topic_context=topic_context[:2000] if topic_context else cleaned_title,
            recent_news=headlines_block,
        )

        config = genai_types.GenerateContentConfig(
            temperature=0.25,
            response_mime_type="application/json",
        )

        raw_json = _call_gemini_with_fallback(client=client, contents=prompt, config=config)
        clean_text = clean_json_markdown(raw_json)
        data: dict[str, Any] = json.loads(clean_text)

        dossier = ResearchDossier(
            topic=cleaned_title,
            summary=str(data.get("summary") or f"Ανάλυση για {cleaned_title}."),
            key_statistics=[str(s) for s in data.get("key_statistics", []) if str(s).strip()],
            official_statements=[str(s) for s in data.get("official_statements", []) if str(s).strip()],
            counter_arguments=[str(s) for s in data.get("counter_arguments", []) if str(s).strip()],
            timeline_events=[str(s) for s in data.get("timeline_events", []) if str(s).strip()],
            recent_headlines=unique_headlines[:5],
        )
        logger.info(
            "[Research Engine] Deep research completed: %d stats, %d counterpoints, %d headlines.",
            len(dossier.key_statistics),
            len(dossier.counter_arguments),
            len(dossier.recent_headlines),
        )
        return dossier

    except (APIError, json.JSONDecodeError, OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        logger.warning("[Research Engine] Gemini synthesis encountered error (%s) — using fallback dossier.", exc)
        return ResearchDossier(
            topic=cleaned_title,
            summary=f"Εξελίξεις και αναλύσεις για το ζήτημα: {cleaned_title}.",
            key_statistics=[],
            official_statements=[topic_context[:250]] if topic_context else [],
            counter_arguments=[],
            timeline_events=[],
            recent_headlines=unique_headlines[:5],
        )
