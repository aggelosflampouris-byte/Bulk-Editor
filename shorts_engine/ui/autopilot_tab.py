import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


def render_autopilot_tab(settings: Any) -> None:
    st.markdown("""
<div style="background: linear-gradient(135deg, #18181b, #27272a); padding: 1.75rem 2rem; border-radius: 12px; border: 1px solid #3f3f46; margin-bottom: 1.5rem;">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h2 style="color: white; margin: 0 0 0.4rem 0;">🤖 Dianisma Autopilot Studio</h2>
            <p style="color: #a1a1aa; font-size: 0.95rem; margin: 0;">
                End-to-end Shorts extraction, transcription, B-roll composition, and SEO for <strong>@DianismaNews</strong>.
            </p>
        </div>
        <div style="background: #09090b; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #27272a; text-align: right;">
            <span style="color: #22c55e; font-size: 0.85rem; font-weight: 600;">● Official Data API</span>
            <div style="color: #71717a; font-size: 0.75rem;">@DianismaNews Dedicated</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # 1. Check YouTube Authentication & Channel Info
    try:
        from services.channel_analyzer import (
            DIANISMA_CHANNEL_HANDLE,
            DIANISMA_CHANNEL_URL,
            VideoMeta,
        )
        from services.youtube_uploader import (
            authenticate,
            fetch_my_recent_videos,
            get_channel_info,
            is_authenticated,
        )
        yt_connected = is_authenticated()
    except ImportError:
        from shorts_engine.services.channel_analyzer import (
            DIANISMA_CHANNEL_HANDLE,
            DIANISMA_CHANNEL_URL,
            VideoMeta,
        )
        from shorts_engine.services.youtube_uploader import (
            authenticate,
            fetch_my_recent_videos,
            get_channel_info,
            is_authenticated,
        )
        yt_connected = is_authenticated()

    dianisma_videos: list[VideoMeta] = []
    selected_video: VideoMeta | None = None

    if yt_connected:
        if "dianisma_channel_info" not in st.session_state or "dianisma_channel_videos" not in st.session_state:
            with st.spinner("Connecting to YouTube Data API for @DianismaNews..."):
                try:
                    yt_client = authenticate()
                    st.session_state["dianisma_channel_info"] = get_channel_info(yt_client)
                    st.session_state["dianisma_channel_videos"] = fetch_my_recent_videos(yt_client, max_videos=30)
                except (RuntimeError, OSError, ValueError) as exc:
                    logger.warning("Could not pre-fetch channel info: %s", exc)

        info = st.session_state.get("dianisma_channel_info", {})
        dianisma_videos = st.session_state.get("dianisma_channel_videos", [])

        # Channel Status Bar
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns([3, 1.5, 1.5, 1.5])
        with stat_col1:
            title = info.get("title", "Dianisma")
            st.markdown(f"**Channel:** `{title}` ({DIANISMA_CHANNEL_HANDLE})")
        with stat_col2:
            st.markdown(f"**Videos:** `{info.get('video_count', len(dianisma_videos)):,}`")
        with stat_col3:
            st.markdown(f"**Subscribers:** `{info.get('subscriber_count', 0):,}`")
        with stat_col4:
            if st.button("🔄 Refresh", use_container_width=True, help="Refresh video library from YouTube"):
                st.session_state.pop("dianisma_channel_info", None)
                st.session_state.pop("dianisma_channel_videos", None)
                st.rerun()

        st.markdown("---")

        # Source Selection Mode
        source_mode = st.radio(
            "Source Mode",
            [
                "⚡ Auto-Detect Highest Viral Breakout (YouTube Data API)",
                "📁 Select Specific Video from Channel Library",
            ],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

        if "Auto-Detect" in source_mode:
            st.info(
                "Autopilot will scan the recent uploads of **@DianismaNews** using real-time view counts, "
                "calculate the Relative Velocity Ratio (RVR), filter out already converted videos, "
                "and select the #1 breakout video.",
                icon="⚡",
            )
        else:
            if dianisma_videos:
                video_map = {
                    f"[{v.upload_date_display}] {v.title}  ({v.view_count:,} views • {v.duration_display})": v
                    for v in dianisma_videos
                }
                choice = st.selectbox("Choose video from Dianisma uploads:", list(video_map.keys()))
                if choice:
                    selected_video = video_map[choice]
            else:
                st.warning("No recent videos retrieved from the channel library.")

    else:
        st.warning(
            "⚠️ **YouTube Account Not Connected** — Authenticate in the **Upload to YouTube** tab "
            "to enable direct YouTube Data API v3 library access."
        )

    # Custom B-Roll Notification
    broll_path = st.session_state.get("custom_broll_path", "")
    if broll_path:
        st.success(f"Using Custom B-Roll: `{Path(broll_path).name}`")

    # Options Row
    opt_col1, opt_col2 = st.columns([1, 2])
    with opt_col1:
        num_shorts = st.number_input("Shorts to generate", min_value=1, max_value=3, value=1, step=1)
    with opt_col2, st.expander("⚙️ Advanced: Override Source URL"):
        override_url = st.text_input(
            "Custom Channel/Video URL Override",
            placeholder=DIANISMA_CHANNEL_URL,
        )

    st.markdown("")

    # Launch Button
    btn_label = f"🚀 Launch Autopilot for {'Selected Video' if selected_video else '@DianismaNews'}"
    if st.button(btn_label, type="primary", use_container_width=True):
        try:
            from services.autopilot import run_autopilot_pipeline
        except ImportError:
            from shorts_engine.services.autopilot import run_autopilot_pipeline

        # Clear previous result
        st.session_state.pop("autopilot_result", None)

        st.markdown("### Execution Log")
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.container()

        target = override_url.strip() if override_url.strip() else (DIANISMA_CHANNEL_URL if not selected_video else "")

        try:
            for msg, pct, data in run_autopilot_pipeline(
                target_url=target,
                settings=settings,
                broll_path=broll_path,
                num_videos=int(num_shorts),
                selected_video=selected_video,
            ):
                progress_bar.progress(pct)
                status_text.markdown(f"**{pct}%** — {msg}")
                log_container.write(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}")

                if pct == 100 and data:
                    st.session_state["autopilot_result"] = data
                    st.rerun()
        except Exception as e:
            st.error(f"Autopilot failed: {e}")
            logger.exception("Autopilot pipeline error")

    # Outside the button, render review UI
    autopilot_results = st.session_state.get("autopilot_result")
    if autopilot_results:
        try:
            from ui.components.review_card import render_review_and_approve_list
        except ImportError:
            from shorts_engine.ui.components.review_card import (
                render_review_and_approve_list,
            )
        render_review_and_approve_list(autopilot_results, key_prefix="ap")
