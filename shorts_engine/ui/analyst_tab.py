"""
ui/analyst_tab.py — AI Strategy Analyst Chat Tab.

Renders a premium chat interface inside the Autopilot Studio that:
  - Loads all project/channel analytics into a structured context snapshot.
  - Exposes that context in an expandable data panel.
  - Connects to the Qwen 2.5 72B model via OpenRouter for streaming responses.
  - Maintains per-session conversation history.
  - Allows the user to refresh the analytics snapshot mid-session.
"""

from __future__ import annotations

import logging
import os

import streamlit as st

logger = logging.getLogger(__name__)

# ── Session state keys ─────────────────────────────────────────────────────────
_KEY_ANALYST_HISTORY = "analyst_chat_history"
_KEY_ANALYST_CONTEXT = "analyst_analytics_context"
_KEY_ANALYST_INSTANCE = "analyst_instance"


def _init_analyst_session() -> None:
    """Initialise all session state keys required by the analyst tab."""
    if _KEY_ANALYST_HISTORY not in st.session_state:
        st.session_state[_KEY_ANALYST_HISTORY] = []
    if _KEY_ANALYST_CONTEXT not in st.session_state:
        st.session_state[_KEY_ANALYST_CONTEXT] = None
    if _KEY_ANALYST_INSTANCE not in st.session_state:
        st.session_state[_KEY_ANALYST_INSTANCE] = None


def _load_analytics_context():
    """Collect fresh analytics and cache the context + markdown in session state."""
    try:
        from services.analytics_collector import collect_analytics
    except ImportError:
        from shorts_engine.services.analytics_collector import collect_analytics

    ctx = collect_analytics()
    st.session_state[_KEY_ANALYST_CONTEXT] = ctx
    return ctx


def _get_or_create_analyst(api_key: str, analytics_markdown: str):
    """
    Return the cached AIAnalyst instance, creating it if necessary or
    refreshing its analytics context if the key has changed.
    """
    try:
        from services.ai_analyst import AIAnalyst
    except ImportError:
        from shorts_engine.services.ai_analyst import AIAnalyst

    existing = st.session_state.get(_KEY_ANALYST_INSTANCE)
    if existing is None:
        instance = AIAnalyst(api_key=api_key, analytics_markdown=analytics_markdown)
        st.session_state[_KEY_ANALYST_INSTANCE] = instance
        return instance

    # Hot-swap analytics if a refresh was triggered
    existing.refresh_analytics(analytics_markdown)
    return existing


def _render_analytics_panel(ctx) -> None:
    """Render the collapsible analytics data panel."""
    with st.expander("📊 Live Analytics Snapshot", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Projects", ctx.total_projects)
        col2.metric("Approved", ctx.approved_projects)
        col3.metric("Avg Virality", f"{ctx.avg_virality_score:.1f}/10")
        col4.metric("Avg Rating", f"{ctx.avg_user_rating:.1f}/5")

        if ctx.top_niches:
            st.markdown("**Top Niches**")
            niche_cols = st.columns(min(4, len(ctx.top_niches)))
            for i, (niche, count) in enumerate(ctx.top_niches[:4]):
                breakdown = ctx.niche_breakdown.get(niche, {})
                with niche_cols[i]:
                    st.markdown(
                        f"<div style='background:#18181b;border:1px solid #27272a;"
                        f"border-radius:8px;padding:0.75rem;text-align:center;'>"
                        f"<div style='color:#a1a1aa;font-size:0.72rem;text-transform:uppercase;"
                        f"letter-spacing:0.05em;'>{niche}</div>"
                        f"<div style='color:#f4f4f5;font-size:1.4rem;font-weight:600;'>{count}</div>"
                        f"<div style='color:#71717a;font-size:0.72rem;'>"
                        f"⚡{breakdown.get('avg_virality', 0):.1f} ⭐{breakdown.get('avg_rating', 0):.1f}"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

        if ctx.scheduled_upload_slots:
            st.markdown("**Next Optimal Upload Slots (Greece)**")
            for slot in ctx.scheduled_upload_slots[:5]:
                st.markdown(
                    f"<div style='background:#0f1e14;border:1px solid #14532d;"
                    f"border-radius:6px;padding:0.4rem 0.75rem;margin-bottom:0.35rem;"
                    f"font-size:0.82rem;color:#4ade80;'>{slot}</div>",
                    unsafe_allow_html=True,
                )


def _render_message_bubble(role: str, content: str) -> None:
    """Render a single chat message with styled bubble."""
    if role == "user":
        st.markdown(
            f"<div style='display:flex;justify-content:flex-end;margin-bottom:0.75rem;'>"
            f"<div style='background:#27272a;border:1px solid #3f3f46;border-radius:12px 12px 2px 12px;"
            f"padding:0.7rem 1rem;max-width:80%;color:#f4f4f5;font-size:0.9rem;line-height:1.5;'>"
            f"{content}"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='display:flex;justify-content:flex-start;margin-bottom:0.75rem;'>"
            f"<div style='background:#0d1117;border:1px solid #27272a;border-radius:12px 12px 12px 2px;"
            f"padding:0.7rem 1rem;max-width:88%;color:#e4e4e7;font-size:0.9rem;line-height:1.6;'>"
            f"<div style='color:#4ade80;font-size:0.72rem;font-weight:600;margin-bottom:0.35rem;"
            f"text-transform:uppercase;letter-spacing:0.06em;'>🤖 Strategy Analyst</div>"
            f"{content}"
            f"</div></div>",
            unsafe_allow_html=True,
        )


_STARTER_PROMPTS: list[str] = [
    "📈 Which niche should I focus on next based on performance data?",
    "🕐 When is the best time to upload this week for Greek viewers?",
    "🎬 What caption style gets the highest virality in my channel?",
    "✍️ How should I improve my hook writing for political content?",
    "🔍 What content gaps exist that I haven't explored yet?",
    "📋 Give me a full upload plan for the next 7 days.",
]


def render_analyst_tab() -> None:
    """Render the full AI Strategy Analyst chat tab."""
    _init_analyst_session()

    # ── API Key Guard ───────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    if not api_key:
        st.markdown("""
<div style="background:linear-gradient(135deg,#18181b,#1a1a2e);border:1px solid #3f3f46;
border-radius:12px;padding:2rem;text-align:center;margin-top:1rem;">
    <div style="font-size:2.5rem;margin-bottom:0.75rem;">🤖</div>
    <h3 style="color:#f4f4f5;margin:0 0 0.5rem 0;">AI Strategy Analyst</h3>
    <p style="color:#a1a1aa;margin:0 0 1.25rem 0;max-width:480px;margin-left:auto;margin-right:auto;">
        Powered by <strong>Qwen 2.5 72B</strong> via OpenRouter. Analyzes your channel data 
        and gives evidence-based recommendations on uploads, timing, editing, and SEO.
    </p>
    <div style="background:#451a03;border:1px solid #78350f;border-radius:8px;
    padding:0.75rem 1.25rem;display:inline-block;color:#fbbf24;font-size:0.85rem;">
        ⚠️ <strong>OPENROUTER_API_KEY</strong> not found in <code>.env</code><br>
        Get a free key at <a href="https://openrouter.ai/keys" target="_blank" 
        style="color:#fb923c;">openrouter.ai/keys</a> and add it to 
        <code>shorts_engine/.env</code>
    </div>
</div>
        """, unsafe_allow_html=True)
        return

    # ── Load Analytics ──────────────────────────────────────────────────────────
    ctx = st.session_state.get(_KEY_ANALYST_CONTEXT)
    if ctx is None:
        with st.spinner("Collecting analytics data…"):
            ctx = _load_analytics_context()

    # ── Compact data header with inline refresh ─────────────────────────────────
    kpi_l, kpi_m, kpi_r, refresh_col = st.columns([2, 2, 2, 1])
    kpi_l.metric("Projects", ctx.total_projects)
    kpi_m.metric("Avg Virality", f"{ctx.avg_virality_score:.1f}/10")
    kpi_r.metric("Avg Rating", f"{ctx.avg_user_rating:.1f}/5")
    with refresh_col:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        if st.button("🔄", key="analyst_refresh_btn", help="Refresh analytics snapshot",
                     use_container_width=True):
            st.session_state[_KEY_ANALYST_CONTEXT] = None
            st.session_state[_KEY_ANALYST_INSTANCE] = None
            st.rerun()

    _render_analytics_panel(ctx)

    # ── Analyst Instance ────────────────────────────────────────────────────────
    analyst = _get_or_create_analyst(
        api_key=api_key,
        analytics_markdown=ctx.to_markdown_summary(),
    )

    st.markdown("<hr style='border-color:#27272a;margin:1rem 0;'>", unsafe_allow_html=True)

    # ── Conversation History ────────────────────────────────────────────────────
    history = st.session_state[_KEY_ANALYST_HISTORY]

    if not history:
        # Welcome message + starter prompts
        st.markdown(
            "<div style='background:#0d1117;border:1px solid #1d4ed8;"
            "border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem;'>"
            "<div style='color:#4ade80;font-size:0.72rem;font-weight:600;"
            "text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem;'>"
            "🤖 Strategy Analyst</div>"
            "<div style='color:#e4e4e7;font-size:0.9rem;line-height:1.6;'>"
            f"Hello! I've loaded your analytics — <strong>{ctx.total_projects} projects</strong>, "
            f"avg virality <strong>{ctx.avg_virality_score:.1f}/10</strong>. "
            "Ask me anything about what to upload next, when to post, or how to improve "
            "your editing and metadata for @DianismaNews."
            "</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Quick questions:**")
        cols = st.columns(3)
        for i, prompt in enumerate(_STARTER_PROMPTS):
            with cols[i % 3]:
                if st.button(prompt, key=f"starter_{i}", use_container_width=True):
                    st.session_state[_KEY_ANALYST_HISTORY].append(
                        {"role": "user", "content": prompt}
                    )
                    st.rerun()
    else:
        # Render conversation history
        chat_container = st.container()
        with chat_container:
            for msg in history:
                _render_message_bubble(msg["role"], msg["content"])

    # ── Chat Input ──────────────────────────────────────────────────────────────
    user_input: str | None = st.chat_input(
        "Ask about uploads, timing, editing, SEO, content strategy…",
        key="analyst_chat_input",
    )

    if user_input and user_input.strip():
        # Append user message to display history
        st.session_state[_KEY_ANALYST_HISTORY].append(
            {"role": "user", "content": user_input.strip()}
        )

        # Render user bubble
        _render_message_bubble("user", user_input.strip())

        # Stream assistant response
        response_placeholder = st.empty()
        response_placeholder.markdown(
            "<div style='color:#71717a;font-size:0.85rem;font-style:italic;'>"
            "⏳ Thinking…</div>",
            unsafe_allow_html=True,
        )

        streamed_chunks: list[str] = []
        content_started = False
        try:
            for chunk in analyst.stream_response(user_input.strip()):
                # The first non-whitespace chunk marks transition from thinking → content
                stripped = chunk.strip()
                if stripped and not content_started:
                    content_started = True

                streamed_chunks.append(chunk)
                accumulated = "".join(streamed_chunks).strip()

                if not content_started:
                    # Still in thinking phase — show pulsing indicator
                    response_placeholder.markdown(
                        "<div style='color:#71717a;font-size:0.85rem;font-style:italic;'>"
                        "🧠 Analyzing data…</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    response_placeholder.markdown(
                        f"<div style='background:#0d1117;border:1px solid #27272a;"
                        f"border-radius:12px 12px 12px 2px;padding:0.7rem 1rem;"
                        f"color:#e4e4e7;font-size:0.9rem;line-height:1.6;'>"
                        f"<div style='color:#4ade80;font-size:0.72rem;font-weight:600;"
                        f"text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.35rem;'>"
                        f"🤖 Strategy Analyst</div>"
                        f"{accumulated}</div>",
                        unsafe_allow_html=True,
                    )

            full_response = "".join(streamed_chunks)
            if full_response:
                st.session_state[_KEY_ANALYST_HISTORY].append(
                    {"role": "assistant", "content": full_response}
                )
                st.rerun()

        except RuntimeError as exc:
            response_placeholder.error(f"**Analyst error:** {exc}")
            logger.error("AI Analyst streaming error: %s", exc)

    # ── Clear History ───────────────────────────────────────────────────────────
    if history:
        if st.button("🗑 Clear Conversation", key="analyst_clear_btn"):
            st.session_state[_KEY_ANALYST_HISTORY] = []
            if st.session_state.get(_KEY_ANALYST_INSTANCE):
                st.session_state[_KEY_ANALYST_INSTANCE].clear_history()
            st.rerun()
