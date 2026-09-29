"""
ui/seo_settings_panel.py — SEO Pre-flight Settings Panel.

Renders a two-column footer panel at the bottom of the Autopilot Engine.
Left column: SEO metadata form (editable).
Right column: Title variants, tag cloud, pinned comment, upload slot.

Auto-fill logic:
  - On niche/source change: instantly loads niche template defaults.
  - On "Generate with AI" click: calls Gemini for optimised metadata.
  - Persisted in session state so values survive reruns and pre-fill
    into the pipeline's SEO generator for post-processing enrichment.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

# ── Session state key namespace ────────────────────────────────────────────────
_KEY_SEO   = "seo_preflight_package"
_KEY_NICHE = "seo_last_niche"
_KEY_TOPIC = "seo_last_topic"

# ── Niche → display label map ──────────────────────────────────────────────────
_NICHE_OPTIONS: list[tuple[str, str]] = [
    ("politics",    "🏛️ Politics & News"),
    ("economics",   "📈 Economics & Finance"),
    ("technology",  "💻 Technology & AI"),
    ("automotive",  "🚗 Automotive & Car Vlog"),
    ("lifestyle",   "✨ Lifestyle & Wellness"),
    ("custom",      "⚙️ Custom / Other"),
]


def _get_niche_from_template(selected_template_name: str) -> str:
    """Map the sidebar niche template name to a seo_preflight niche key."""
    mapping: dict[str, str] = {
        "politics":    "politics",
        "economics":   "economics",
        "technology":  "technology",
        "automotive":  "automotive",
        "lifestyle":   "lifestyle",
    }
    return mapping.get(selected_template_name.lower(), "custom")


def _auto_load_template_defaults(niche: str, topic: str) -> None:
    """Load niche template defaults into session state if niche/topic changed."""
    try:
        from services.seo_preflight import get_template_defaults
    except ImportError:
        from shorts_engine.services.seo_preflight import get_template_defaults

    pkg = get_template_defaults(niche=niche, topic=topic)
    st.session_state[_KEY_SEO] = pkg
    st.session_state[_KEY_NICHE] = niche
    st.session_state[_KEY_TOPIC] = topic


def render_seo_settings_panel(settings: Any, current_topic: str = "") -> None:
    """
    Render the SEO pre-flight footer panel inside the engine column.

    Provides a two-column layout:
      Left  (2/5): Editable core fields — title, description, tags, niche.
      Right (3/5): Title variants picker, tag cloud, pinned comment, timing.

    Changes to these fields are stored in st.session_state[_KEY_SEO] and
    are read by the pipeline to override / seed the post-process SEO generator.
    """
    # ── Detect current niche & topic from engine state ─────────────────────────
    selected_template = st.session_state.get("niche_template_select", "custom")
    detected_niche = _get_niche_from_template(selected_template)

    # Topic context: prioritize caller's current_topic, then session state
    topic_context = (
        current_topic.strip()
        or st.session_state.get("target_niche_input", "")
        or st.session_state.get("ap_url_input", "")
        or ""
    )

    # ── Auto-load when niche or topic changes ──────────────────────────────────
    pkg = st.session_state.get(_KEY_SEO)
    last_niche = st.session_state.get(_KEY_NICHE, "")
    last_topic = st.session_state.get(_KEY_TOPIC, "")

    if pkg is None or last_niche != detected_niche or (topic_context and last_topic != topic_context):
        _auto_load_template_defaults(niche=detected_niche, topic=topic_context)
        pkg = st.session_state[_KEY_SEO]

    # ── Section header ─────────────────────────────────────────────────────────
    st.markdown("---")
    hdr_l, hdr_r = st.columns([6, 1])
    with hdr_l:
        badge = (
            "<span style='background:#052e16;color:#4ade80;border:1px solid #14532d;"
            "border-radius:4px;padding:1px 8px;font-size:0.72rem;font-weight:600;"
            "margin-left:0.5rem;vertical-align:middle;'>AI-Generated</span>"
            if pkg.source == "ai" else
            "<span style='background:#18181b;color:#71717a;border:1px solid #27272a;"
            "border-radius:4px;padding:1px 8px;font-size:0.72rem;font-weight:600;"
            "margin-left:0.5rem;vertical-align:middle;'>Template Defaults</span>"
        )
        st.markdown(
            f"<div style='padding:0.35rem 0;'>"
            f"<strong style='font-size:1rem;color:#f4f4f5;'>📋 SEO & Metadata Settings</strong>"
            f"{badge}</div>",
            unsafe_allow_html=True,
        )
    with hdr_r:
        has_gemini = bool(getattr(settings, "gemini_api_key", ""))
        ai_btn_help = (
            "Generate optimised title, description, tags and pinned comment with Gemini"
            if has_gemini else "Add GEMINI_API_KEY to .env to enable AI generation"
        )
        if st.button(
            "✨ AI Fill",
            key="seo_ai_fill_btn",
            disabled=not has_gemini,
            help=ai_btn_help,
            use_container_width=True,
        ):
            with st.spinner("Generating SEO with Gemini…"):
                try:
                    from services.seo_preflight import generate_ai_seo
                except ImportError:
                    from shorts_engine.services.seo_preflight import generate_ai_seo

                source_title = (
                    topic_context
                    or st.session_state.get("url_meta_title", "")
                    or "Greek Short"
                )
                source_url = st.session_state.get("ap_url_input", "")
                try:
                    new_pkg = generate_ai_seo(
                        gemini_api_key=settings.gemini_api_key,
                        niche=detected_niche,
                        source_title=source_title,
                        source_url=source_url,
                        brand_voice=getattr(settings, "brand_voice", ""),
                    )
                    st.session_state[_KEY_SEO] = new_pkg
                    pkg = new_pkg
                    st.success("SEO package generated!", icon="✅")
                    st.rerun()
                except RuntimeError as exc:
                    st.error(f"AI SEO generation failed: {exc}")

    # ── Two-column SEO form layout ─────────────────────────────────────────────
    left_col, right_col = st.columns([2, 3], gap="medium")

    # ── LEFT: Core editable fields ─────────────────────────────────────────────
    with left_col:
        st.markdown(
            "<p style='color:#a1a1aa;font-size:0.78rem;font-weight:600;"
            "text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;'>"
            "CORE METADATA</p>",
            unsafe_allow_html=True,
        )

        # Niche selector — overrides the auto-detect
        niche_display = dict(_NICHE_OPTIONS)
        niche_keys = [k for k, _ in _NICHE_OPTIONS]
        niche_idx = niche_keys.index(detected_niche) if detected_niche in niche_keys else 0
        selected_niche = st.selectbox(
            "Content Niche",
            options=niche_keys,
            format_func=lambda k: niche_display.get(k, k),
            index=niche_idx,
            key="seo_niche_override",
            help="Auto-detected from your Niche Template selection. Override here if needed.",
        )
        if selected_niche != detected_niche:
            _auto_load_template_defaults(niche=selected_niche, topic=topic_context)
            pkg = st.session_state[_KEY_SEO]
            st.rerun()

        # Primary title (editable)
        new_title = st.text_input(
            "YouTube Title (Primary)",
            value=pkg.title,
            max_chars=100,
            key="seo_title_field",
            help="Max 100 chars. Aim for 50-85 for best mobile display.",
            placeholder="e.g. 🔥 Αυτό που δεν σας είπαν για την αύξηση των φόρων",
        )

        # Description (editable)
        new_desc = st.text_area(
            "Description",
            value=pkg.description,
            height=150,
            max_chars=5000,
            key="seo_description_field",
            help="Hook in first 2 lines. End with CTA + hashtags.",
        )

        # Primary keyword
        new_kw = st.text_input(
            "Primary Keyword",
            value=pkg.primary_keyword,
            max_chars=60,
            key="seo_primary_kw_field",
            help="Core 1-2 word search keyword for this video.",
        )

        # Tags (comma-separated)
        new_tags_raw = st.text_area(
            "Tags (comma-separated, max 500 chars total)",
            value=pkg.tags_display,
            height=80,
            key="seo_tags_field",
            help="12-18 tags mixing Greek queries and English trend keywords.",
        )

    # ── RIGHT: Title variants, pinned comment, upload timing ──────────────────
    with right_col:
        st.markdown(
            "<p style='color:#a1a1aa;font-size:0.78rem;font-weight:600;"
            "text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;'>"
            "TITLE VARIANTS & OPTIMISATION</p>",
            unsafe_allow_html=True,
        )

        # Title variant cards with quick-apply buttons
        if pkg.curiosity_title:
            c_box, c_btn = st.columns([5, 1])
            with c_box:
                st.markdown(
                    "<div style='background:#0d1117;border:1px solid #1d4ed8;"
                    "border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.4rem;'>"
                    "<div style='color:#93c5fd;font-size:0.7rem;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:0.06em;'>🎯 Curiosity Gap</div>"
                    f"<div style='color:#e4e4e7;font-size:0.85rem;margin-top:0.2rem;'>"
                    f"{pkg.curiosity_title}</div></div>",
                    unsafe_allow_html=True,
                )
            with c_btn:
                if st.button("Use", key="btn_use_curiosity", help="Set this as Primary Title"):
                    pkg.title = pkg.curiosity_title
                    st.session_state[_KEY_SEO] = pkg
                    st.rerun()

        if pkg.authority_title:
            a_box, a_btn = st.columns([5, 1])
            with a_box:
                st.markdown(
                    "<div style='background:#0d1117;border:1px solid #15803d;"
                    "border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.4rem;'>"
                    "<div style='color:#86efac;font-size:0.7rem;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:0.06em;'>📣 Authority</div>"
                    f"<div style='color:#e4e4e7;font-size:0.85rem;margin-top:0.2rem;'>"
                    f"{pkg.authority_title}</div></div>",
                    unsafe_allow_html=True,
                )
            with a_btn:
                if st.button("Use", key="btn_use_authority", help="Set this as Primary Title"):
                    pkg.title = pkg.authority_title
                    st.session_state[_KEY_SEO] = pkg
                    st.rerun()

        if pkg.contrarian_title:
            ct_box, ct_btn = st.columns([5, 1])
            with ct_box:
                st.markdown(
                    "<div style='background:#0d1117;border:1px solid #b45309;"
                    "border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.4rem;'>"
                    "<div style='color:#fbbf24;font-size:0.7rem;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:0.06em;'>⚡ Contrarian</div>"
                    f"<div style='color:#e4e4e7;font-size:0.85rem;margin-top:0.2rem;'>"
                    f"{pkg.contrarian_title}</div></div>",
                    unsafe_allow_html=True,
                )
            with ct_btn:
                if st.button("Use", key="btn_use_contrarian", help="Set this as Primary Title"):
                    pkg.title = pkg.contrarian_title
                    st.session_state[_KEY_SEO] = pkg
                    st.rerun()

        if not any([pkg.curiosity_title, pkg.authority_title, pkg.contrarian_title]):
            st.markdown(
                "<div style='background:#111113;border:1px dashed #3f3f46;"
                "border-radius:8px;padding:1rem;text-align:center;color:#52525b;"
                "font-size:0.82rem;'>Click <strong>✨ AI Fill</strong> to generate "
                "title variants</div>",
                unsafe_allow_html=True,
            )

        # Pinned comment
        st.markdown(
            "<p style='color:#a1a1aa;font-size:0.78rem;font-weight:600;"
            "text-transform:uppercase;letter-spacing:0.05em;margin:0.75rem 0 0.35rem;'>"
            "📌 PINNED COMMENT (Engagement Trap)</p>",
            unsafe_allow_html=True,
        )
        new_pinned = st.text_area(
            "Pinned Comment",
            value=pkg.pinned_comment,
            height=70,
            key="seo_pinned_comment_field",
            label_visibility="collapsed",
            help="Pin this comment immediately after upload to maximise early engagement signals.",
        )

        # Next upload slot
        st.markdown(
            "<p style='color:#a1a1aa;font-size:0.78rem;font-weight:600;"
            "text-transform:uppercase;letter-spacing:0.05em;margin:0.75rem 0 0.35rem;'>"
            "📅 NEXT OPTIMAL UPLOAD SLOT</p>",
            unsafe_allow_html=True,
        )
        try:
            from services.traffic_scheduler import get_greek_high_traffic_slots
        except ImportError:
            from shorts_engine.services.traffic_scheduler import get_greek_high_traffic_slots

        try:
            slots = get_greek_high_traffic_slots(count=3)
            for s in slots:
                st.markdown(
                    f"<div style='background:#0f1e14;border:1px solid #14532d;"
                    f"border-radius:6px;padding:0.3rem 0.7rem;margin-bottom:0.3rem;"
                    f"font-size:0.78rem;color:#4ade80;'>{s.display_str}</div>",
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            st.caption(f"Slots unavailable: {exc}")

    # ── Persist edited values back into session state ──────────────────────────
    parsed_tags = [t.strip() for t in new_tags_raw.split(",") if t.strip()]

    # Update the package with user edits (without triggering a rerun)
    try:
        from services.seo_preflight import PreflightSeoPackage
    except ImportError:
        from shorts_engine.services.seo_preflight import PreflightSeoPackage

    updated_pkg = PreflightSeoPackage(
        title=new_title.strip(),
        curiosity_title=pkg.curiosity_title,
        authority_title=pkg.authority_title,
        contrarian_title=pkg.contrarian_title,
        description=new_desc.strip(),
        tags=parsed_tags,
        pinned_comment=new_pinned.strip(),
        primary_keyword=new_kw.strip(),
        niche=selected_niche,
        source=pkg.source,
    )
    st.session_state[_KEY_SEO] = updated_pkg
