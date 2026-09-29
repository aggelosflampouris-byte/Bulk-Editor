"""
services/seo_preflight.py — Lightweight pre-processing SEO generation.

Generates an optimised SEO package for a video BEFORE full pipeline
processing. Uses only the source URL, title, niche, and channel context
(no transcription required) to give the user editable metadata upfront.

This differs from seo_generator.py which runs POST-transcription.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Niche SEO defaults ─────────────────────────────────────────────────────────
# Pre-built high-performing tag pools and structural templates per niche.
# Used both as instant defaults AND as few-shot guidance for Gemini.

_NICHE_TAG_POOLS: dict[str, list[str]] = {
    "politics": [
        "ελληνική πολιτική", "πολιτικά νέα", "ειδήσεις σήμερα", "κυβέρνηση",
        "βουλή", "κόμματα", "πολιτική Ελλάδα", "τελευταία νέα", "Shorts",
        "Greek politics", "αποκάλυψη", "σκάνδαλο", "συνέντευξη", "αναλύσεις",
        "DianismaNews", "πολιτική ανάλυση",
    ],
    "economics": [
        "ελληνική οικονομία", "οικονομικά νέα", "αγορά", "επενδύσεις",
        "ακρίβεια", "φόροι", "προϋπολογισμός", "τιμές", "Shorts", "οικονομία",
        "τράπεζες", "ευρώ", "ΑΕΠ", "οικονομικές ειδήσεις", "DianismaNews",
        "Greek economy",
    ],
    "technology": [
        "τεχνολογία", "τεχνητή νοημοσύνη", "AI", "smartphone", "gadgets",
        "νέα τεχνολογία", "review", "test", "Shorts", "tech", "software",
        "hardware", "future", "τεχνολογικά νέα", "DianismaNews", "Greek tech",
    ],
    "automotive": [
        "αυτοκίνητο", "car", "αυτοκινητόδρομος", "car vlog", "οδική δοκιμή",
        "service", "κινητήρας", "ταχύτητα", "Shorts", "car review",
        "ηλεκτρικό", "αυτοκίνητα Ελλάδα", "vlog", "DianismaNews",
        "Greek cars", "mechanic",
    ],
    "lifestyle": [
        "lifestyle", "καθημερινή ζωή", "vlog", "συνήθειες", "υγεία",
        "fitness", "γυμναστήριο", "διατροφή", "ομορφιά", "Shorts",
        "wellness", "tips", "DianismaNews", "ζωή", "ρουτίνα", "motivation",
    ],
    "custom": [
        "Shorts", "ειδήσεις", "DianismaNews", "viral", "trending",
        "Ελλάδα", "Greece", "ελληνικά", "fyp", "explore",
        "τελευταία νέα", "αποκαλύψεις", "ανάλυση", "σχολιασμός", "video",
    ],
}

_NICHE_DESCRIPTION_TEMPLATES: dict[str, str] = {
    "politics": (
        "🔥 Δείτε αυτό που δεν σας λένε για {topic}. Αναλύουμε τα πραγματικά δεδομένα.\n\n"
        "📌 Ποια είναι η αλήθεια πίσω από τις εξελίξεις; Μείνετε ενημερωμένοι.\n\n"
        "👇 Σχολιάστε και πείτε μας τη γνώμη σας!\n"
        "#Shorts #ΕλληνικήΠολιτική #Ελλάδα #Ειδήσεις #DianismaNews"
    ),
    "economics": (
        "📈 Τι σημαίνει αυτό για την τσέπη σας; Αναλύουμε τις τελευταίες οικονομικές εξελίξεις.\n\n"
        "💡 Όλα όσα πρέπει να γνωρίζετε για {topic} — με αριθμούς και πραγματικά δεδομένα.\n\n"
        "👇 Πώς επηρεάζει εσάς;\n"
        "#Shorts #Οικονομία #Ελλάδα #Ακρίβεια #DianismaNews"
    ),
    "technology": (
        "🤖 Αυτή η τεχνολογία αλλάζει τα πάντα. Δείτε τι πρέπει να ξέρετε για {topic}.\n\n"
        "⚡ Πλήρης ανάλυση και demo — δείτε αν αξίζει.\n\n"
        "👇 Εσείς τι πιστεύετε;\n"
        "#Shorts #Τεχνολογία #AI #Tech #DianismaNews"
    ),
    "automotive": (
        "🚗 Αυτό που πρέπει να ξέρετε για {topic}. Πλήρης car vlog και ανάλυση.\n\n"
        "🔧 Τεχνικά στοιχεία, αποδόσεις και πραγματικές εμπειρίες από τον δρόμο.\n\n"
        "👇 Ποιο αυτοκίνητο θα επιλέγατε;\n"
        "#Shorts #Αυτοκίνητο #CarVlog #Ελλάδα #DianismaNews"
    ),
    "lifestyle": (
        "✨ Αυτή η αλλαγή άλλαξε τη ζωή μου — δείτε πώς μπορεί να κάνει το ίδιο και σε εσάς.\n\n"
        "💪 Όλα τα tips και τα μυστικά για {topic} σε ένα video.\n\n"
        "👇 Εσείς το δοκιμάσατε;\n"
        "#Shorts #Lifestyle #Tips #Ελλάδα #DianismaNews"
    ),
    "custom": (
        "🔥 {topic} — μια αποκαλυπτική ματιά που δεν πρέπει να χάσετε.\n\n"
        "📌 Μείνετε ενημερωμένοι με τις πιο έγκυρες αναλύσεις.\n\n"
        "👇 Σχολιάστε παρακάτω!\n"
        "#Shorts #Ελλάδα #DianismaNews #Viral #Trending"
    ),
}

_PINNED_COMMENT_TEMPLATES: dict[str, str] = {
    "politics": "Πιστεύετε ότι η κυβέρνηση κρύβει κάτι; Γράψτε \"ΝΑΙ\" ή \"ΟΧΙ\" παρακάτω 👇",
    "economics": "Πόσο έχει επηρεαστεί η ζωή σας από την ακρίβεια; Γράψτε μας παρακάτω 👇",
    "technology": "Θα χρησιμοποιούσατε αυτή την τεχνολογία; Ψηφίστε: 🤖 ΝΑΙ ή ❌ ΟΧΙ 👇",
    "automotive": "Ποιο είναι το αυτοκίνητό σας; Γράψτε μας παρακάτω — τσεκάρουμε κάθε σχόλιο! 🚗",
    "lifestyle": "Ποια συνήθεια άλλαξε ΠΡΩΤΑ τη ζωή σας; Γράψτε μας 👇",
    "custom": "Ποια είναι η γνώμη σας; Γράψτε παρακάτω — απαντάμε σε κάθε σχόλιο! 👇",
}


@dataclass
class PreflightSeoPackage:
    """Editable SEO metadata package generated before pipeline processing."""
    title: str = ""
    curiosity_title: str = ""
    authority_title: str = ""
    contrarian_title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    pinned_comment: str = ""
    primary_keyword: str = ""
    niche: str = "custom"
    source: str = "template"   # "template" | "ai"

    @property
    def tags_display(self) -> str:
        return ", ".join(self.tags)


def get_template_defaults(niche: str, topic: str = "") -> PreflightSeoPackage:
    """
    Instantly return niche-specific SEO defaults — no API call, no latency.

    Args:
        niche:  Niche key matching _NICHE_TAG_POOLS (e.g. "politics").
        topic:  Optional topic string to interpolate into templates.

    Returns:
        PreflightSeoPackage populated with template defaults.
    """
    n = niche.lower() if niche.lower() in _NICHE_TAG_POOLS else "custom"
    topic_str = topic.strip() or "την τελευταία εξέλιξη"

    description = _NICHE_DESCRIPTION_TEMPLATES[n].replace("{topic}", topic_str)
    pinned = _PINNED_COMMENT_TEMPLATES[n]
    tags = list(_NICHE_TAG_POOLS[n])

    return PreflightSeoPackage(
        title=f"{topic_str[:80]}" if topic_str else "",
        curiosity_title="",
        authority_title="",
        contrarian_title="",
        description=description,
        tags=tags,
        pinned_comment=pinned,
        primary_keyword=topic_str.split()[0] if topic_str else "",
        niche=n,
        source="template",
    )


def generate_ai_seo(
    gemini_api_key: str,
    niche: str,
    source_title: str,
    source_url: str = "",
    brand_voice: str = "",
) -> PreflightSeoPackage:
    """
    Generate optimised SEO metadata via Gemini using only source context
    (no transcription required). Fast, lightweight, and runs in ~3 seconds.

    Args:
        gemini_api_key: Gemini API key.
        niche:          Detected or user-selected niche.
        source_title:   Title of the source video.
        source_url:     URL of the source video (optional).
        brand_voice:    Brand voice override from sidebar (optional).

    Returns:
        PreflightSeoPackage with AI-generated SEO fields.

    Raises:
        RuntimeError: On Gemini API failure.
    """
    if not gemini_api_key:
        raise RuntimeError("Gemini API key is required for AI SEO generation.")

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:
        raise RuntimeError(f"google-genai not installed: {exc}") from exc

    # Pull niche tag pool as few-shot context
    n = niche.lower() if niche.lower() in _NICHE_TAG_POOLS else "custom"
    tag_examples = ", ".join(_NICHE_TAG_POOLS[n][:8])

    brand_injection = (
        f"\nBrand Voice: {brand_voice}\n" if brand_voice else ""
    )

    prompt = f"""\
You are an elite YouTube Shorts SEO strategist specialising in Greek-language content for the channel @DianismaNews.
{brand_injection}
Generate a complete, high-CTR SEO package for a Short based ONLY on the following source context:

SOURCE TITLE: {source_title}
SOURCE URL: {source_url or "N/A"}
CONTENT NICHE: {niche}
EXAMPLE HIGH-PERFORMING TAGS FOR THIS NICHE: {tag_examples}

Respond ONLY with a valid JSON object — no markdown, no code fences:
{{
  "title": "<Greek title, 50-85 chars, complete phrase, 1-2 strategic emojis allowed>",
  "curiosity_title": "<Greek title with curiosity gap angle>",
  "authority_title": "<Greek title with authority/expert angle>",
  "contrarian_title": "<Greek title with controversial/contrarian angle>",
  "primary_keyword": "<1-2 word Greek keyword>",
  "description": "<Greek description: hook in first 2 lines, value expansion, CTA + 3-5 hashtags>",
  "pinned_comment": "<Greek engagement-trap comment — polarizing question or fill-in-the-blank>",
  "tags": ["<tag1>", ..., "<tag16>"]
}}

Rules:
- All titles and descriptions MUST be in Greek.
- Titles: 50-85 chars, never cut off mid-word, create strong curiosity gap.
- Tags: 12-16 items, mix of Greek search queries and English trend keywords.
- No direct quotes from the title; always reframe to maximize CTR.
"""

    client = genai.Client(api_key=gemini_api_key)

    # Try models with fallback
    models = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash"]
    raw_text = ""
    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(max_output_tokens=1500),
            )
            raw_text = response.text or ""
            if raw_text:
                break
        except Exception as exc:
            logger.warning("Gemini model %s failed: %s", model, exc)

    if not raw_text:
        raise RuntimeError("All Gemini models failed for SEO preflight generation.")

    # Parse JSON
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fenced:
        raw_text = fenced.group(1)
    obj_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not obj_match:
        raise RuntimeError(f"No JSON in Gemini response: {raw_text[:200]}")

    try:
        data = json.loads(obj_match.group(0))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON parse error: {exc}") from exc

    tags_raw = data.get("tags", [])
    tags = [str(t).strip() for t in tags_raw if str(t).strip()][:18]
    if len(tags) < 8:
        tags = tags + _NICHE_TAG_POOLS[n][:max(0, 8 - len(tags))]

    return PreflightSeoPackage(
        title=str(data.get("title", ""))[:100],
        curiosity_title=str(data.get("curiosity_title", ""))[:100],
        authority_title=str(data.get("authority_title", ""))[:100],
        contrarian_title=str(data.get("contrarian_title", ""))[:100],
        description=str(data.get("description", ""))[:5000],
        tags=tags,
        pinned_comment=str(data.get("pinned_comment", _PINNED_COMMENT_TEMPLATES[n]))[:500],
        primary_keyword=str(data.get("primary_keyword", ""))[:60],
        niche=n,
        source="ai",
    )
