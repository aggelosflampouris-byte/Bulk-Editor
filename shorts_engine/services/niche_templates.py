"""
services/niche_templates.py — Per-niche configuration presets for the Shorts Engine.

Each NicheTemplate bundles together all the production settings that should
vary by content type: clip duration, music bed, VFX grade, subtitle placement,
and the brand-voice injection sent to Gemini for SEO and clip selection.

Usage:
    from services.niche_templates import NICHE_TEMPLATES, get_template

    template = get_template("politics")
    settings = template.apply_to(settings)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Settings


@dataclass(frozen=True)
class NicheTemplate:
    """
    Immutable bundle of production settings for a specific content niche.

    All fields map 1-to-1 to Settings attributes so they can be applied
    without any transformation. Fields set to None mean "do not override —
    leave whatever the user configured manually".
    """

    name: str               # Internal key (used in config / prompt)
    display_name: str       # Label shown in the Streamlit sidebar
    emoji: str              # Prefix emoji for the sidebar label
    description: str        # One-sentence tooltip shown in the sidebar

    # ── Clip selection ─────────────────────────────────────────────────────
    clip_min_duration: float
    clip_max_duration: float

    # ── Audio ──────────────────────────────────────────────────────────────
    bg_music_track: str         # "ambient_calm" | "dramatic_pulse" | "upbeat_groove" | "none"
    bg_music_volume: float
    bg_music_ducking: bool

    # ── VFX / colour grade ────────────────────────────────────────────────
    vfx_preset_override: str | None  # Force a VFX grade or None for auto

    # ── Subtitles ─────────────────────────────────────────────────────────
    subtitle_position: str           # "lower_third" | "center" | "top"

    # ── Transcription ─────────────────────────────────────────────────────
    whisper_context_hint: str        # Maps to _DOMAIN_PROMPTS key

    # ── Brand voice / SEO ─────────────────────────────────────────────────
    brand_voice: str                 # Injected into Gemini prompts
    channel_niche_context: str       # Additional niche context for clip selection

    # ── Caption Template ──────────────────────────────────────────────────
    caption_style: str = "auto"      # e.g. "CAR_PULSE_INDUSTRIAL", "HORMOZI_PUNCH", "auto"

    def apply_to(self, settings: "Settings") -> "Settings":
        """
        Return a new Settings instance with this template's values applied.

        Only fields defined in the template are overridden; all other user
        settings (API keys, output dir, B-roll, transitions) are preserved.
        """
        from dataclasses import replace
        return replace(
            settings,
            clip_min_duration=self.clip_min_duration,
            clip_max_duration=self.clip_max_duration,
            bg_music_track=self.bg_music_track,
            bg_music_volume=self.bg_music_volume,
            bg_music_ducking=self.bg_music_ducking,
            subtitle_position=self.subtitle_position,
            whisper_context_hint=self.whisper_context_hint,
            brand_voice=self.brand_voice,
            niche_template=self.name,
            caption_style=self.caption_style,
        )

    @property
    def sidebar_label(self) -> str:
        """Full display label used in the sidebar selectbox."""
        return f"{self.emoji} {self.display_name}"


# ── Registry ───────────────────────────────────────────────────────────────────

NICHE_TEMPLATES: dict[str, NicheTemplate] = {
    "politics": NicheTemplate(
        name="politics",
        display_name="Politics / Economy",
        emoji="🏛️",
        description="Hard-hitting political and economic analysis. Authoritative and polarising tone.",
        clip_min_duration=40.0,
        clip_max_duration=58.0,
        bg_music_track="dramatic_pulse",
        bg_music_volume=0.15,
        bg_music_ducking=True,
        vfx_preset_override="DRAMATIC",
        subtitle_position="lower_third",
        whisper_context_hint="politics",
        brand_voice=(
            "Highly authoritative, analytical, and slightly polarising. Focus on hidden agendas, "
            "economic impact, and hard truths. Tone should be serious, urgent, and provocative. "
            "Write titles that feel like breaking news."
        ),
        channel_niche_context=(
            "Greek political analysis, government accountability, economic policy, "
            "corruption exposés, and geopolitical developments."
        ),
    ),
    "society": NicheTemplate(
        name="society",
        display_name="Society",
        emoji="👥",
        description="Social dynamics, human behaviour, and everyday Greek life.",
        clip_min_duration=35.0,
        clip_max_duration=50.0,
        bg_music_track="ambient_calm",
        bg_music_volume=0.18,
        bg_music_ducking=True,
        vfx_preset_override="WARMTH",
        subtitle_position="center",
        whisper_context_hint="society",
        brand_voice=(
            "Relatable, empathetic, and thought-provoking. Focus on human behaviour, "
            "social dynamics, and everyday realities Greeks face. Tone should spark "
            "intense debate and personal reflection."
        ),
        channel_niche_context=(
            "Greek society, social issues, mental health, family dynamics, "
            "generational divides, and everyday life stories."
        ),
    ),
    "science": NicheTemplate(
        name="science",
        display_name="Science",
        emoji="🔬",
        description="Mind-blowing scientific discoveries explained simply.",
        clip_min_duration=45.0,
        clip_max_duration=60.0,
        bg_music_track="ambient_calm",
        bg_music_volume=0.14,
        bg_music_ducking=True,
        vfx_preset_override="CINEMATIC",
        subtitle_position="lower_third",
        whisper_context_hint="science",
        brand_voice=(
            "Educational, mind-blowing, and highly factual. Focus on explaining complex "
            "scientific concepts simply, debunking myths, and highlighting future implications. "
            "Tone should be awe-inspiring and authoritative. Use analogies that Greeks relate to."
        ),
        channel_niche_context=(
            "Scientific discoveries, medical breakthroughs, climate science, "
            "space exploration, and technology science in Greek."
        ),
    ),
    "technology": NicheTemplate(
        name="technology",
        display_name="Technology",
        emoji="💻",
        description="AI, startups, and tech disruption content.",
        clip_min_duration=35.0,
        clip_max_duration=50.0,
        bg_music_track="upbeat_groove",
        bg_music_volume=0.17,
        bg_music_ducking=True,
        vfx_preset_override="VIBRANCE",
        subtitle_position="lower_third",
        whisper_context_hint="technology",
        brand_voice=(
            "Forward-looking, fast-paced, and analytical. Focus on innovation, disruption, "
            "and how tech changes daily life. Tone should be cutting-edge, enthusiastic, "
            "and slightly urgent. Reference real companies and products."
        ),
        channel_niche_context=(
            "AI, software, startups, Greek tech ecosystem, digital transformation, "
            "cybersecurity, and consumer technology."
        ),
    ),
    "entertainment": NicheTemplate(
        name="entertainment",
        display_name="Entertainment / Pop Culture",
        emoji="🎭",
        description="Viral pop culture, celebrity news, and entertainment.",
        clip_min_duration=28.0,
        clip_max_duration=45.0,
        bg_music_track="upbeat_groove",
        bg_music_volume=0.22,
        bg_music_ducking=True,
        vfx_preset_override="VIBRANCE",
        subtitle_position="center",
        whisper_context_hint="entertainment",
        brand_voice=(
            "Fun, energetic, and instantly relatable. Focus on drama, reactions, and "
            "pop culture moments. Tone should feel like you're telling a friend the "
            "juiciest gossip. Short punchy sentences. Max engagement."
        ),
        channel_niche_context=(
            "Greek and international celebrities, viral trends, reality TV, "
            "music industry, film reviews, and pop culture commentary."
        ),
    ),
    "business": NicheTemplate(
        name="business",
        display_name="Business / Finance",
        emoji="💼",
        description="Business strategy, entrepreneurship, and financial literacy.",
        clip_min_duration=40.0,
        clip_max_duration=58.0,
        bg_music_track="dramatic_pulse",
        bg_music_volume=0.15,
        bg_music_ducking=True,
        vfx_preset_override="DRAMATIC",
        subtitle_position="lower_third",
        whisper_context_hint="business",
        brand_voice=(
            "Authoritative, pragmatic, and value-driven. Focus on actionable insights, "
            "financial opportunities, and entrepreneurial mindset. Tone should combine "
            "authority with practical advice Greeks can apply immediately."
        ),
        channel_niche_context=(
            "Greek business ecosystem, entrepreneurship, personal finance, "
            "investment strategies, real estate, and economic opportunity."
        ),
    ),
    "education": NicheTemplate(
        name="education",
        display_name="Education / History",
        emoji="📚",
        description="Educational content, history, and knowledge-sharing.",
        clip_min_duration=45.0,
        clip_max_duration=60.0,
        bg_music_track="ambient_calm",
        bg_music_volume=0.12,
        bg_music_ducking=True,
        vfx_preset_override="SUBTLE",
        subtitle_position="lower_third",
        whisper_context_hint="education",
        brand_voice=(
            "Clear, engaging, and intellectually stimulating. Focus on making complex "
            "topics accessible and memorable. Tone should feel like the best teacher you "
            "ever had — passionate, clear, and never condescending."
        ),
        channel_niche_context=(
            "Greek history, ancient civilisation, modern Greek education, "
            "biography, philosophy, and general knowledge."
        ),
    ),
    "lifestyle": NicheTemplate(
        name="lifestyle",
        display_name="Lifestyle / Vlog",
        emoji="🌟",
        description="Personal stories, travel, food, and everyday vlogging.",
        clip_min_duration=28.0,
        clip_max_duration=45.0,
        bg_music_track="upbeat_groove",
        bg_music_volume=0.20,
        bg_music_ducking=True,
        vfx_preset_override="WARMTH",
        subtitle_position="center",
        whisper_context_hint="lifestyle",
        brand_voice=(
            "Warm, personal, and aspirational. Focus on authentic moments, emotional "
            "connection, and relatable life experiences. Tone should feel like a "
            "trusted friend sharing their life journey."
        ),
        channel_niche_context=(
            "Greek lifestyle, travel destinations in Greece, food, fashion, "
            "wellness, personal development, and daily life vlogs."
        ),
    ),
    "gaming": NicheTemplate(
        name="gaming",
        display_name="Gaming",
        emoji="🎮",
        description="Gaming commentary, reviews, and esports content.",
        clip_min_duration=25.0,
        clip_max_duration=40.0,
        bg_music_track="upbeat_groove",
        bg_music_volume=0.18,
        bg_music_ducking=True,
        vfx_preset_override="VIBRANCE",
        subtitle_position="top",
        whisper_context_hint="gaming",
        brand_voice=(
            "High-energy, humorous, and deeply in-the-know. Focus on epic moments, "
            "hot takes, and gaming culture. Tone should be enthusiastic with gamer slang "
            "that resonates with the Greek gaming community."
        ),
        channel_niche_context=(
            "Video games, esports, game reviews, streaming, "
            "Greek gaming community, and gaming culture."
        ),
    ),
    "automotive": NicheTemplate(
        name="automotive",
        display_name="Automotive / Car Vlogs",
        emoji="🏎️",
        description="Car builds, mechanic repairs, engine diagnostics, and driving vlogs.",
        clip_min_duration=35.0,
        clip_max_duration=52.0,
        bg_music_track="upbeat_groove",
        bg_music_volume=0.18,
        bg_music_ducking=True,
        vfx_preset_override="VIBRANCE",
        subtitle_position="lower_third",
        whisper_context_hint="automotive",
        brand_voice=(
            "Passionate, authentic gearhead and mechanic tone. Focus on mechanical problems, "
            "costly mistakes, horsepower, engine diagnostics, and practical car advice. "
            "Intriguing and urgent hooks."
        ),
        channel_niche_context=(
            "Automotive mechanics, Greek car community, track days, car tuning, "
            "diagnostic tests, and garage repairs."
        ),
        caption_style="CAR_PULSE_INDUSTRIAL",
    ),
    "custom": NicheTemplate(
        name="custom",
        display_name="Custom",
        emoji="⚙️",
        description="All settings configured manually. No preset applied.",
        clip_min_duration=35.0,
        clip_max_duration=50.0,
        bg_music_track="ambient_calm",
        bg_music_volume=0.20,
        bg_music_ducking=True,
        vfx_preset_override=None,
        subtitle_position="lower_third",
        whisper_context_hint="",
        brand_voice="",
        channel_niche_context="High-value, engaging content across any topic.",
        caption_style="auto",
    ),
}


def get_template(name: str) -> NicheTemplate:
    """
    Return the NicheTemplate for the given name.

    Falls back to the 'custom' template if the name is unknown.

    Args:
        name: Template key, e.g. 'politics', 'gaming'.

    Returns:
        The matching NicheTemplate.
    """
    return NICHE_TEMPLATES.get(name, NICHE_TEMPLATES["custom"])


def template_options() -> list[tuple[str, str]]:
    """
    Return (name, sidebar_label) pairs in display order for the sidebar selectbox.

    Returns:
        List of (internal_name, display_label) tuples.
    """
    order = [
        "custom",
        "automotive",
        "politics",
        "society",
        "science",
        "technology",
        "entertainment",
        "business",
        "education",
        "lifestyle",
        "gaming",
    ]
    return [(k, NICHE_TEMPLATES[k].sidebar_label) for k in order if k in NICHE_TEMPLATES]


def auto_detect_niche(
    title: str = "",
    description: str = "",
    text_sample: str = "",
) -> str:
    """
    Intelligently infer the content niche from video metadata and speech transcript.

    Returns:
        One of the registered niche keys (e.g. 'automotive', 'technology', 'politics')
        or 'custom' if confidence is low.
    """
    corpus = f"{title} {description} {text_sample}".lower()

    # 1. Automotive & Mechanics
    auto_keywords = (
        "κινητήρας", "αμάξι", "αυτοκίνητο", "μοτέρ", "φλάντζα", "μηχανικός",
        "συνεργείο", "service", "τουρμπίνα", "turbo", "drift", "bmw", "audi",
        "mercedes", "toyota", "λάδια", "γκάζι", "φρένα", "exhaust", "κιβώτιο",
        "καπό", "ιπποδύναμη", "άλογα", "κυβικά", "car vlog", "tuning"
    )
    if any(k in corpus for k in auto_keywords):
        return "automotive"

    # 2. Politics & Economy
    politics_keywords = (
        "κυβέρνηση", "βουλή", "υπουργός", "πρωθυπουργός", "εκλογές", "κόμμα",
        "προϋπολογισμός", "φόροι", "ακρίβεια", "διαφθορά", "πολιτική", "εε", "νατο"
    )
    if any(k in corpus for k in politics_keywords):
        return "politics"

    # 3. Technology & AI
    tech_keywords = (
        "τεχνολογία", "ai", "τεχνητή νοημοσύνη", "software", "coding", "προγραμματισμός",
        "crypto", "bitcoin", "hardware", "smartphone", "iphone", "android", "app"
    )
    if any(k in corpus for k in tech_keywords):
        return "technology"

    # 4. Gaming
    gaming_keywords = (
        "gaming", "gamer", "playstation", "xbox", "gameplay", "streamer",
        "esports", "fortnite", "gta", "minecraft", "counter-strike"
    )
    if any(k in corpus for k in gaming_keywords):
        return "gaming"

    # 5. Science & Education
    science_keywords = (
        "επιστήμη", "διάστημα", "nasa", "φυσική", "σύμπαν", "ιστορία",
        "αρχαία ελλάδα", "φιλοσοφία", "εγκέφαλος", "ανακάλυψη"
    )
    if any(k in corpus for k in science_keywords):
        return "science"

    # 6. Business & Finance
    biz_keywords = (
        "επιχείρηση", "επένδυση", "χρήματα", "κεφάλαιο", "startup", "πωλήσεις",
        "marketing", "real estate", "εισόδημα", "μετοχές"
    )
    if any(k in corpus for k in biz_keywords):
        return "business"

    # 7. Entertainment & Lifestyle
    ent_keywords = (
        "μουσική", "τραγούδι", "ηθοποιός", "ταινία", "σινεμά", "celebrity",
        "φαγητό", "συνταγή", "ταξίδι", "διακοπές", "γυμναστική", "vlog"
    )
    if any(k in corpus for k in ent_keywords):
        return "entertainment"

    return "custom"
