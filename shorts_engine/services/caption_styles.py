"""
services/caption_styles.py — Diverse caption templates and occasion-based selection.

Provides distinct subtitle styles (Hormozi, Minimal, Neon Cyber, Automotive Industrial,
Viral TikTok, Documentary, Classic) with Greek typography support and intelligent
occasion-based automatic template resolution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptionStyleTemplate:
    """
    Immutable specification of an ASS subtitle style bundle.
    """

    id: str
    display_name: str
    emoji: str
    description: str
    font_name: str
    font_size: int
    bold: bool
    primary_color: str              # ASS format &HAABBGGRR (e.g. &H00FFFFFF for white)
    outline_color: str
    back_color: str                 # Shadow / box color
    highlight_primary_color: str    # Active spoken word font color
    highlight_outline_color: str
    highlight_back_color: str
    border_style: int               # 1 = outline + shadow, 3 = opaque background box
    outline_width: int
    shadow_depth: int
    spacing: int                    # Letter spacing in pixels
    occasion_tags: tuple[str, ...]

    def build_ass_styles(self, margin_v: int = 540) -> tuple[str, str]:
        """
        Build the base and highlight ASS Style lines.

        Format:
        Style: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,
               Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,
               Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
        """
        bold_flag = -1 if self.bold else 0

        # Base style for spoken text
        base_style = (
            f"Style: Default,{self.font_name},{self.font_size},"
            f"{self.primary_color},&H000000FF,{self.outline_color},{self.back_color},"
            f"{bold_flag},0,0,0,100,100,{self.spacing},0,{self.border_style},"
            f"{self.outline_width},{self.shadow_depth},2,80,80,{margin_v},1"
        )

        # Highlight style for active spoken word
        highlight_style = (
            f"Style: HighlightBox,{self.font_name},{self.font_size},"
            f"{self.highlight_primary_color},&H000000FF,{self.highlight_outline_color},{self.highlight_back_color},"
            f"{bold_flag},0,0,0,100,100,{self.spacing},0,{self.border_style},"
            f"{self.outline_width},{self.shadow_depth},2,80,80,{margin_v},1\n"
            f"Style: Highlight,{self.font_name},{self.font_size},"
            f"{self.highlight_primary_color},&H000000FF,{self.highlight_outline_color},{self.highlight_back_color},"
            f"{bold_flag},0,0,0,100,100,{self.spacing},0,{self.border_style},"
            f"{self.outline_width},{self.shadow_depth},2,80,80,{margin_v},1"
        )

        return base_style, highlight_style


# ── Registry of Diverse Caption Templates ─────────────────────────────────────

CAPTION_TEMPLATES: dict[str, CaptionStyleTemplate] = {
    # 1. Automotive & Garage Industrial: Sturdy, high contrast, hazard amber/yellow pop
    "CAR_PULSE_INDUSTRIAL": CaptionStyleTemplate(
        id="CAR_PULSE_INDUSTRIAL",
        display_name="Car Pulse Industrial",
        emoji="🏎️",
        description="Heavy industrial aesthetic with high-visibility hazard amber highlights. Ideal for car vlogs, mechanics & garage repairs.",
        font_name="DejaVu Sans",
        font_size=82,
        bold=True,
        primary_color="&H00FFFFFF",          # Pure white
        outline_color="&H00141414",          # Heavy asphalt dark outline
        back_color="&H88000000",             # Shadow
        highlight_primary_color="&H0000D7FF",# Vibrant safety / hazard yellow-amber
        highlight_outline_color="&H00000000",# Crisp black edge
        highlight_back_color="&H00000000",
        border_style=1,                      # Heavy outline + shadow
        outline_width=6,
        shadow_depth=3,
        spacing=1,
        occasion_tags=("automotive", "car", "mechanic", "diy", "garage", "vlog", "parts", "motor"),
    ),

    # 2. Hormozi Bold Punch: Massive uppercase, vivid contrast yellow/gold box
    "HORMOZI_PUNCH": CaptionStyleTemplate(
        id="HORMOZI_PUNCH",
        display_name="Hormozi Punch",
        emoji="⚡",
        description="High-energy, ultra-bold text with punchy golden-yellow active boxes. Maximizes retention and urgency.",
        font_name="DejaVu Sans",
        font_size=84,
        bold=True,
        primary_color="&H00FFFFFF",
        outline_color="&H00000000",
        back_color="&H80000000",
        highlight_primary_color="&H00000000",# Black text on vivid yellow bounding pill
        highlight_outline_color="&H0000FFFF",# Yellow pill box
        highlight_back_color="&H00000000",
        border_style=3,                      # Opaque highlight bounding box
        outline_width=14,
        shadow_depth=3,
        spacing=2,
        occasion_tags=("motivation", "business", "urgency", "viral", "energy", "shock"),
    ),

    # 3. Viral TikTok Bounce: Crisp white with electric neon lime green pop
    "VIRAL_TIKTOK_BOUNCE": CaptionStyleTemplate(
        id="VIRAL_TIKTOK_BOUNCE",
        display_name="Viral TikTok Bounce",
        emoji="🔥",
        description="Modern short-form dual-tone. Electric lime green highlights on deep black strokes. Universal viral pacing.",
        font_name="DejaVu Sans",
        font_size=78,
        bold=True,
        primary_color="&H00FFFFFF",
        outline_color="&H00101010",
        back_color="&H66000000",
        highlight_primary_color="&H0014FF39",# Vibrant neon lime green (&H00BBGGRR)
        highlight_outline_color="&H00000000",
        highlight_back_color="&H00000000",
        border_style=1,
        outline_width=5,
        shadow_depth=2,
        spacing=1,
        occasion_tags=("entertainment", "lifestyle", "tiktok", "vlog", "humor", "reaction"),
    ),

    # 4. Neon Cyber: Futuristic glowing cyan & magenta outline
    "NEON_CYBER": CaptionStyleTemplate(
        id="NEON_CYBER",
        display_name="Neon Cyberpunk",
        emoji="🔮",
        description="Vibrant cyan with glowing magenta/neon accents. Perfect for technology, AI, gaming, and night POV.",
        font_name="DejaVu Sans",
        font_size=78,
        bold=True,
        primary_color="&H00FFFF00",          # Bright electric cyan
        outline_color="&H001A0033",          # Deep purple outline
        back_color="&H80FF00FF",             # Magenta glow
        highlight_primary_color="&H00FF00FF",# Intense magenta / pink highlight
        highlight_outline_color="&H00000000",
        highlight_back_color="&H00FFFF00",
        border_style=1,
        outline_width=5,
        shadow_depth=3,
        spacing=1,
        occasion_tags=("technology", "gaming", "ai", "crypto", "night", "futuristic"),
    ),

    # 5. Elegant Minimal: Clean sans-serif, understated luxury / commentary
    "ELEGANT_MINIMAL": CaptionStyleTemplate(
        id="ELEGANT_MINIMAL",
        display_name="Elegant Minimalist",
        emoji="✨",
        description="Sophisticated, clean sans-serif with delicate warm accents. Ideal for luxury, finance, and calm reflection.",
        font_name="DejaVu Sans",
        font_size=70,
        bold=False,
        primary_color="&H00F5F5F5",          # Off-white / cream
        outline_color="&H00202020",
        back_color="&H40000000",
        highlight_primary_color="&H0000D0FF",# Muted warm amber gold
        highlight_outline_color="&H00111111",
        highlight_back_color="&H00000000",
        border_style=1,
        outline_width=3,
        shadow_depth=1,
        spacing=1,
        occasion_tags=("society", "finance", "minimal", "luxury", "podcast", "reflection"),
    ),

    # 6. Documentary Clean: Classic broadcast journalistic subtitle with dark pill box
    "DOCUMENTARY_CLEAN": CaptionStyleTemplate(
        id="DOCUMENTARY_CLEAN",
        display_name="Documentary Broadcast",
        emoji="📰",
        description="Crisp journalistic typography with translucent backdrop. High legibility for news, politics, and education.",
        font_name="DejaVu Sans",
        font_size=72,
        bold=True,
        primary_color="&H00FFFFFF",
        outline_color="&H00000000",
        back_color="&HB0121212",             # Translucent dark backing box
        highlight_primary_color="&H0033E6FF",# Soft journalistic yellow
        highlight_outline_color="&H00000000",
        highlight_back_color="&HB0121212",
        border_style=3,                      # Bounding box
        outline_width=10,
        shadow_depth=2,
        spacing=0,
        occasion_tags=("politics", "news", "education", "science", "history", "documentary"),
    ),

    # 7. Classic Yellow Box: Original baseline style
    "CLASSIC_YELLOW": CaptionStyleTemplate(
        id="CLASSIC_YELLOW",
        display_name="Classic Yellow Box",
        emoji="🟡",
        description="Classic high-contrast yellow highlight bounding box.",
        font_name="DejaVu Sans",
        font_size=76,
        bold=True,
        primary_color="&H00FFFFFF",
        outline_color="&H00000000",
        back_color="&H66000000",
        highlight_primary_color="&H00000000",
        highlight_outline_color="&H0000E6FF",
        highlight_back_color="&H00000000",
        border_style=3,
        outline_width=12,
        shadow_depth=3,
        spacing=1,
        occasion_tags=("generic", "classic", "default"),
    ),
}


# ── Intelligent Occasion-Based Selection ──────────────────────────────────────

def get_caption_style(style_id: str | None) -> CaptionStyleTemplate:
    """
    Lookup a caption template by ID, falling back to CAR_PULSE_INDUSTRIAL or CLASSIC_YELLOW.
    """
    if not style_id or style_id.lower() in ("auto", "none", "default"):
        return CAPTION_TEMPLATES["CLASSIC_YELLOW"]
    
    key = style_id.upper().strip()
    return CAPTION_TEMPLATES.get(key, CAPTION_TEMPLATES["CLASSIC_YELLOW"])


def recommend_caption_style(
    niche: str | None = None,
    topic_or_title: str | None = None,
    transcript_sample: str | None = None,
) -> CaptionStyleTemplate:
    """
    Determine the optimal caption style template dynamically based on:
      1. Explicit or detected content niche
      2. Key semantic concepts in the title / topic
      3. Energy level & sentiment in the transcript sample

    Returns:
        The highest-scoring CaptionStyleTemplate for the occasion.
    """
    combined_text = f"{niche or ''} {topic_or_title or ''} {transcript_sample or ''}".lower()

    # Priority 1: Automotive / Car Vlogs / Mechanics / Garage
    if any(k in combined_text for k in (
        "automotive", "car", "auto", "κινητήρας", "αμάξι", "αυτοκίνητο", "μοτέρ",
        "φλάντζα", "μηχανικός", "συνεργείο", "service", "τουρμπίνα", "turbo",
        " drift", "bmw", "audi", "mercedes", "toyota", "λάδια", "γκαζι", "exhaust"
    )):
        return CAPTION_TEMPLATES["CAR_PULSE_INDUSTRIAL"]

    # Priority 2: Technology / Gaming / AI / Cyber
    if any(k in combined_text for k in (
        "technology", "gaming", "ai", "τεχνολογία", "gamer", "crypto", "robot",
        "software", "playstation", "xbox", "cyber", "nvidia", "hardware", "app"
    )):
        return CAPTION_TEMPLATES["NEON_CYBER"]

    # Priority 3: Politics, News, History, Education, Science
    if any(k in combined_text for k in (
        "politics", "education", "science", "πολιτική", "ιστορία", "κυβέρνηση",
        "επιστήμη", "έρευνα", "βουλή", "οικονομία", "documentary", "news", "αρχαία"
    )):
        return CAPTION_TEMPLATES["DOCUMENTARY_CLEAN"]

    # Priority 4: High-energy Business, Urgency, Motivation
    if any(k in combined_text for k in (
        "business", "money", "χρήματα", "επιχείρηση", "πλούτος", "λάθος", "επικίνδυνο",
        "προσοχή", "μυστικό", "αποκάλυψη", "success", "mindset"
    )):
        return CAPTION_TEMPLATES["HORMOZI_PUNCH"]

    # Priority 5: Entertainment, Lifestyle, TikTok Viral
    if any(k in combined_text for k in (
        "entertainment", "lifestyle", "viral", "vlog", "φαγητό", "διακοπές",
        "celebrity", "show", "τάση", "gossip", "ταξίδι", "food"
    )):
        return CAPTION_TEMPLATES["VIRAL_TIKTOK_BOUNCE"]

    # Default fallback to classic yellow
    return CAPTION_TEMPLATES["CLASSIC_YELLOW"]


def list_caption_styles() -> list[CaptionStyleTemplate]:
    """Return all available caption style templates in recommended display order."""
    order = [
        "CAR_PULSE_INDUSTRIAL",
        "HORMOZI_PUNCH",
        "VIRAL_TIKTOK_BOUNCE",
        "NEON_CYBER",
        "ELEGANT_MINIMAL",
        "DOCUMENTARY_CLEAN",
        "CLASSIC_YELLOW",
    ]
    return [CAPTION_TEMPLATES[k] for k in order if k in CAPTION_TEMPLATES]
