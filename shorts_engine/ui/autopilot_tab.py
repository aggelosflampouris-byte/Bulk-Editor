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
            <h2 style="color: white; margin: 0 0 0.4rem 0;">🤖 Autopilot Studio • Niche Trend Discovery</h2>
            <p style="color: #a1a1aa; font-size: 0.95rem; margin: 0;">
                Discovers fresh viral videos on YouTube in our niche (≤ 3 weeks old), extracts high-impact clips, blends AI breakdown & B-roll, and adds royalty-free background music for publishing to <strong>@DianismaNews</strong>.
            </p>
        </div>
        <div style="background: #09090b; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #27272a; text-align: right;">
            <span style="color: #22c55e; font-size: 0.85rem; font-weight: 600;">● YouTube Niche Discovery</span>
            <div style="color: #71717a; font-size: 0.75rem;">≤ 21 Days Old • Fresh Trends</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # 1. Check YouTube Authentication & Channel Info
    try:
        from services.channel_analyzer import (
            DIANISMA_CHANNEL_HANDLE,
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
            st.markdown(f"**Publish Channel:** `{title}` ({DIANISMA_CHANNEL_HANDLE})")
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
                "🌐 Niche Trend Discovery (Search YouTube for Niche Videos ≤ 3 Weeks Old — Recommended)",
                "📁 Select Specific Video from Channel Library (Manual Override)",
            ],
            index=0,
            horizontal=False,
        )

        niche_focus_query = ""
        if "Niche Trend" in source_mode:
            st.info(
                "⚡ **Niche Autopilot Active:** Searches YouTube for the freshest, highest-velocity videos in our niche "
                "(Greek politics, economics, debates) uploaded within the last **3 weeks (≤ 21 days)**. "
                "Automatically excludes older videos from **@DianismaNews** to ensure fresh external material.",
                icon="🌐",
            )
            niche_focus_query = st.text_input(
                "Niche Keyword / Topic Focus (Optional)",
                placeholder="e.g. ελληνική πολιτική, εξελίξεις οικονομία, συνεντεύξεις (Leave blank for auto-detected trending topics)",
                help="Leave blank to automatically discover the top breakout topics across the niche, or enter a specific focus topic.",
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

    # Instrumental Royalty-Free Background Music Section
    st.markdown("#### 🎵 Background Music (Instrumental & Royalty-Free)")
    m_col1, m_col2 = st.columns([1.8, 1.2])
    with m_col1:
        bg_choice = st.selectbox(
            "Music Track Bed",
            [
                "🎵 Ambient Calm (Instrumental • Royalty-Free)",
                "🔥 Dramatic Pulse (Instrumental • Royalty-Free)",
                "⚡ Upbeat Groove (Instrumental • Royalty-Free)",
                "🚫 None (Voice Only)",
            ],
            index=0,
            help="High-retention instrumental background bed with automated speech ducking (volume dips dynamically when voice is present).",
        )
        music_map = {
            "🎵 Ambient Calm (Instrumental • Royalty-Free)": "ambient_calm",
            "🔥 Dramatic Pulse (Instrumental • Royalty-Free)": "dramatic_pulse",
            "⚡ Upbeat Groove (Instrumental • Royalty-Free)": "upbeat_groove",
            "🚫 None (Voice Only)": "none",
        }
        selected_track = music_map.get(bg_choice, "ambient_calm")
        settings.enable_bg_music = (selected_track != "none")
        settings.bg_music_track = selected_track
    with m_col2:
        bg_vol = st.slider(
            "Music Volume",
            min_value=0.05,
            max_value=0.35,
            value=getattr(settings, "bg_music_volume", 0.15),
            step=0.01,
            format="%.2f",
            help="Subtle background audio level (0.15 = 15%). Voice ducking automatically compresses volume during narration.",
        )
        settings.bg_music_volume = bg_vol
        settings.bg_music_ducking = True

    # Options Row
    opt_col1, opt_col2, opt_col3 = st.columns([1, 1.8, 1.2])
    with opt_col1:
        num_shorts = st.number_input("Shorts to generate", min_value=1, max_value=3, value=1, step=1)
    with opt_col2:
        strategy_label = st.selectbox(
            "Production Mode",
            [
                "🔥 Hybrid: Speaker Clip + AI Breakdown (Recommended)",
                "🤖 Smart Auto-Select (Speaker Detect vs Full AI)",
                "👤 Speaker Preservation (B-Roll + Subtitle Masking)",
                "✨ Full AI Short Generation (AI Script + Voiceover + 9:16 Media)",
            ],
            index=0,
            help="Hybrid: Extracts a punchy 8-15s authentic clip of the speaker, masks old captions and burns new karaoke subtitles, then binds it seamlessly with an AI voiceover and 9:16 Pexels B-roll breakdown explaining the facts. Auto: detects speaker presence dynamically. Speaker: retains full speaker clip with B-roll. AI Gen: produces Short from scratch.",
        )
        strategy_map = {
            "🔥 Hybrid: Speaker Clip + AI Breakdown (Recommended)": "hybrid",
            "🤖 Smart Auto-Select (Speaker Detect vs Full AI)": "auto",
            "👤 Speaker Preservation (B-Roll + Subtitle Masking)": "speaker",
            "✨ Full AI Short Generation (AI Script + Voiceover + 9:16 Media)": "ai_gen",
        }
        production_strategy = strategy_map.get(strategy_label, "hybrid")
    with opt_col3, st.expander("⚙️ Override URL"):
        override_url = st.text_input(
            "Custom Channel/Video URL",
            placeholder="e.g. https://www.youtube.com/@competitor or video URL",
        )

    st.markdown("")

    # Launch Button
    btn_label = f"🚀 Launch Autopilot for {'Selected Video' if selected_video else ('Topic: ' + niche_focus_query if niche_focus_query else 'Niche Viral Trends (≤ 3 Weeks Old)')}"
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

        target = override_url.strip()

        try:
            for msg, pct, data in run_autopilot_pipeline(
                target_url=target,
                settings=settings,
                broll_path=broll_path,
                num_videos=int(num_shorts),
                selected_video=selected_video,
                production_strategy=production_strategy,
                niche_query=niche_focus_query,
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
