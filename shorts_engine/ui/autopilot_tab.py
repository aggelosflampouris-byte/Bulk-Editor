"""
ui/autopilot_tab.py — Unified Autopilot Studio.

Consolidates all intelligent sourcing (Niche Outlier Discovery, Direct URL,
Local File Drag-and-Drop, Channel Library), AI production strategies,
royalty-free audio ducking, and one-click YouTube Studio scheduling.
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


def render_autopilot_tab(settings: Any) -> None:
    """Render the unified Greek Shorts Autopilot Studio."""
    # ── Header Banner ──────────────────────────────────────────────────────────
    st.markdown("""
<div style="background: linear-gradient(135deg, #18181b, #27272a); padding: 1.75rem 2rem; border-radius: 12px; border: 1px solid #3f3f46; margin-bottom: 1.5rem;">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
        <div>
            <h2 style="color: white; margin: 0 0 0.4rem 0;">🤖 Greek Shorts Autopilot Studio</h2>
            <p style="color: #a1a1aa; font-size: 0.95rem; margin: 0; max-width: 780px;">
                Unified autonomous intelligence for <strong>@DianismaNews</strong>. Discovers viral niche trends (≤ 21 days),
                tracks speaker faces with YOLO, cross-checks facts via Gemini dossiers, burns dynamic highlight captions,
                and schedules ready-to-publish 9:16 Shorts directly to YouTube Studio.
            </p>
        </div>
        <div style="background: #09090b; padding: 0.6rem 1.2rem; border-radius: 8px; border: 1px solid #27272a; text-align: right;">
            <span style="color: #22c55e; font-size: 0.85rem; font-weight: 600;">● Autopilot Active</span>
            <div style="color: #71717a; font-size: 0.75rem;">YOLO • Gemini • Whisper • YouTube API</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── YouTube Authentication & Live Channel Status ────────────────────────────
    try:
        from services.channel_analyzer import (
            DIANISMA_CHANNEL_HANDLE,
            VideoMeta,
        )
        from services.youtube_uploader import (
            YouTubeAuthError,
            YouTubeUploadError,
            authenticate,
            fetch_my_recent_videos,
            get_channel_info,
            is_authenticated,
            revoke_token,
            upload_short,
        )
        yt_connected = is_authenticated()
    except ImportError:
        from shorts_engine.services.channel_analyzer import (
            DIANISMA_CHANNEL_HANDLE,
            VideoMeta,
        )
        from shorts_engine.services.youtube_uploader import (
            YouTubeAuthError,
            YouTubeUploadError,
            authenticate,
            fetch_my_recent_videos,
            get_channel_info,
            is_authenticated,
            revoke_token,
            upload_short,
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
                except (YouTubeAuthError, RuntimeError, OSError, ValueError) as exc:
                    logger.warning("Could not pre-fetch channel info: %s", exc)

        info = st.session_state.get("dianisma_channel_info", {})
        dianisma_videos = st.session_state.get("dianisma_channel_videos", [])

        # Live Channel Status Bar
        stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns([3, 1.4, 1.4, 1.2, 1.2])
        with stat_col1:
            title = info.get("title", "Dianisma")
            st.markdown(f"**Connected Channel:** `{title}` ({DIANISMA_CHANNEL_HANDLE})")
        with stat_col2:
            st.markdown(f"**Videos:** `{info.get('video_count', len(dianisma_videos)):,}`")
        with stat_col3:
            st.markdown(f"**Subscribers:** `{info.get('subscriber_count', 0):,}`")
        with stat_col4:
            if st.button("🔄 Refresh", use_container_width=True, help="Refresh video library from YouTube"):
                st.session_state.pop("dianisma_channel_info", None)
                st.session_state.pop("dianisma_channel_videos", None)
                st.rerun()
        with stat_col5:
            if st.button("🔓 Disconnect", use_container_width=True, help="Disconnect YouTube account"):
                revoke_token()
                st.session_state.pop("dianisma_channel_info", None)
                st.session_state.pop("dianisma_channel_videos", None)
                st.rerun()

    else:
        auth_box_col1, auth_box_col2 = st.columns([3.5, 1.5])
        with auth_box_col1:
            st.info(
                "🔒 **YouTube Account Not Connected:** Connect your YouTube channel to enable direct library access, "
                "competitor velocity analytics, and one-click scheduled uploads to YouTube Studio."
            )
        with auth_box_col2:
            if st.button("🔗 Connect YouTube Account", key="yt_connect_ap", type="primary", use_container_width=True):
                try:
                    with st.spinner("Opening Google OAuth2 login..."):
                        yt_cl = authenticate()
                        st.session_state["dianisma_channel_info"] = get_channel_info(yt_cl)
                        st.session_state["dianisma_channel_videos"] = fetch_my_recent_videos(yt_cl, max_videos=30)
                    st.success("Connected successfully!")
                    st.rerun()
                except (YouTubeAuthError, RuntimeError, OSError, ValueError) as exc:
                    st.error(f"Authentication failed: {exc}")

    st.markdown("---")

    # ── Unified Sourcing Mode ──────────────────────────────────────────────────
    st.markdown("### 🎯 1. Intelligent Sourcing Engine")
    source_options = [
        "🌐 Niche Trend Discovery (Smart Outlier & Velocity Scanner — Recommended)",
        "🔗 Direct Video / Channel URL (Process Any Link)",
        "📁 Local Video Files (Upload & Drag-and-Drop)",
        "📚 @DianismaNews Channel Library (Quick Picker)",
    ]

    # Pre-select based on queued state
    default_source_idx = 0
    if st.session_state.get("queued_url"):
        default_source_idx = 1
    elif st.session_state.get("queued_local_files"):
        default_source_idx = 2

    source_mode = st.radio(
        "Sourcing Mode",
        source_options,
        index=default_source_idx,
        horizontal=False,
    )

    niche_focus_query = ""
    override_url = ""
    local_saved_paths: list[Path] = []

    # ── Mode A: Niche Trend Discovery ──────────────────────────────────────────
    if "Niche Trend" in source_mode:
        st.info(
            "⚡ **Autonomous Niche Sourcing Active:** Searches YouTube for the freshest, highest-velocity videos in our niche "
            "(Greek politics, economics, debates) uploaded within the last **3 weeks (≤ 21 days)**. "
            "Excludes older @DianismaNews videos to prevent self-cannibalization and maximize discovery of breakout external topics.",
            icon="🌐",
        )
        col_nq, col_ap = st.columns([3.5, 1.5])
        with col_nq:
            niche_focus_query = st.text_input(
                "Niche Keyword / Topic Focus (Optional)",
                value=st.session_state.get("target_niche_input", ""),
                placeholder="e.g. ελληνική πολιτική, εξελίξεις οικονομία, συνεντεύξεις (Leave blank for auto-detected trending topics)",
                help="Leave blank to automatically discover the top breakout topics across the niche, or enter a specific focus topic.",
            )
        with col_ap:
            st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
            if st.button("✨ Auto-Pick Niche with Gemini", use_container_width=True):
                if not settings.gemini_api_key:
                    st.error("Add Gemini API key in sidebar first.")
                else:
                    with st.spinner("Finding high-retention niche..."):
                        try:
                            from google import genai
                            client = genai.Client(api_key=settings.gemini_api_key)
                            prompt = (
                                "Suggest ONE highly engaging, breakout topic in Greek politics, economics, "
                                "or current affairs suitable for a viral YouTube Short. "
                                "Respond with ONLY the topic in 2-5 Greek words. No quotes, no markdown."
                            )
                            response = client.models.generate_content(
                                model="gemini-3.5-flash",
                                contents=prompt,
                            )
                            auto_topic = response.text.strip().replace('"', "")
                            st.session_state["target_niche_input"] = auto_topic
                            st.rerun()
                        except (RuntimeError, OSError, ValueError) as exc:
                            st.error(f"Auto-pick failed: {exc}")

        # Outlier & Velocity Radar Drawer (from niche_explorer_tab)
        with st.expander("📊 Live Niche Outliers & Channel Velocity Radar", expanded=False):
            st.markdown("#### Real-Time Competitor Velocity & Outlier Scanner")
            st.caption(
                "Scans candidate videos, calculates Relative View Ratio (RVR) vs channel median, "
                "and identifies breakout topics uploaded in the last 21 days."
            )
            col_sc1, col_sc2 = st.columns([2, 1])
            with col_sc1:
                scan_depth = st.slider("Scan Depth (Videos per source)", 10, 50, 25, step=5)
            with col_sc2:
                st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
                run_scan = st.button("📡 Scan Outliers & Velocity", type="secondary", use_container_width=True)

            if run_scan:
                with st.spinner("Scanning niche velocity and historical medians..."):
                    try:
                        from services.channel_analyzer import (
                            analyze_niche,
                        )
                        from services.niche_sourcing import fetch_niche_videos_via_ytdlp

                        q = niche_focus_query.strip() if niche_focus_query.strip() else "ελληνική πολιτική ειδήσεις"
                        scanned_vids = fetch_niche_videos_via_ytdlp(
                            query=q,
                            max_results=int(scan_depth),
                            max_age_days=21,
                            output_dir=settings.output_dir,
                        )
                        st.session_state["radar_videos"] = scanned_vids
                        if scanned_vids and settings.gemini_api_key:
                            insights = analyze_niche(
                                scanned_vids,
                                q,
                                gemini_api_key=settings.gemini_api_key,
                                target_niche=q,
                            )
                            st.session_state["radar_insights"] = insights
                    except (RuntimeError, OSError, ValueError) as exc:
                        st.error(f"Radar scan error: {exc}")

            radar_vids: list[VideoMeta] = st.session_state.get("radar_videos", [])
            radar_ins = st.session_state.get("radar_insights")

            if radar_ins:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### 🗂️ Breakout Topic Clusters")
                    st.info(radar_ins.topic_clusters or "—")
                    st.markdown("##### 📅 Best Upload Window")
                    st.info(radar_ins.best_upload_window or "—")
                with c2:
                    st.markdown("##### 🕳️ Content Gaps")
                    st.warning(radar_ins.content_gaps or "—")
                    st.markdown("##### 🔥 Virality Patterns")
                    st.success(radar_ins.virality_patterns or "—")

            if radar_vids:
                try:
                    from services.cache_manager import is_video_already_processed
                    from services.channel_analyzer import find_viral_recent_videos
                except ImportError:
                    from shorts_engine.services.cache_manager import (
                        is_video_already_processed,
                    )
                    from shorts_engine.services.channel_analyzer import (
                        find_viral_recent_videos,
                    )

                viral_recent = find_viral_recent_videos(radar_vids, max_results=10)
                st.markdown("##### 🔥 Top Outlier Candidates")

                unclipped = [
                    vrv for vrv in viral_recent
                    if not is_video_already_processed(vrv.video.url, settings.output_dir)
                ]
                if unclipped and st.button("⚡ Queue Top Unclipped Outlier", type="secondary"):
                    top_v = unclipped[0].video
                    st.session_state["queued_url"] = top_v.url
                    st.session_state["queued_video_meta"] = top_v
                    st.rerun()

                for vrv in viral_recent:
                    v = vrv.video
                    clipped = is_video_already_processed(v.url, settings.output_dir)
                    badge = "✂️ Already Clipped" if clipped else "✨ Fresh Outlier"
                    b_color = "#064e3b" if clipped else "#1e3a8a"
                    st.markdown(
                        f"""
<div style="border:1px solid #27272a;border-radius:8px;padding:10px 14px;margin-bottom:8px;background:#111113;display:flex;justify-content:space-between;align-items:center;">
    <div>
        <strong style="color:#f4f4f5;">{v.title}</strong>
        <div style="color:#71717a;font-size:0.8rem;margin-top:2px;">
            {v.view_count:,} views • {v.upload_date_display} • {v.duration_display}
            <span style="background:{b_color};color:white;border-radius:4px;padding:1px 6px;font-size:0.75rem;margin-left:8px;">{badge}</span>
            <span style="background:#311b92;color:#b388ff;border-radius:4px;padding:1px 6px;font-size:0.75rem;margin-left:4px;">{vrv.rvr_display}</span>
        </div>
    </div>
</div>
""",
                        unsafe_allow_html=True,
                    )

    # ── Mode B: Direct URL Input ───────────────────────────────────────────────
    elif "Direct Video" in source_mode:
        st.info("🔗 **Process Any Link:** Provide a YouTube video, TikTok, Instagram link, or competitor channel URL.", icon="🔗")
        default_val = st.session_state.pop("queued_url", "")
        override_url = st.text_input(
            "Video / Channel URL",
            value=default_val,
            placeholder="e.g. https://www.youtube.com/watch?v=... or https://www.youtube.com/@channel",
            key="ap_url_input",
        )

    # ── Mode C: Local Video File Upload (Drag & Drop) ──────────────────────────
    elif "Local Video" in source_mode:
        st.info(
            "📁 **Local Video Processing:** Upload raw video files from your computer. "
            "Autopilot will transcribe them, track the speaker's face with YOLO, identify the most viral moments, "
            "and craft high-impact 9:16 Shorts with dynamic highlight captions.",
            icon="📁",
        )
        uploaded_files = st.file_uploader(
            "Drop your Greek-speech video clips here",
            type=["mp4", "mov", "mkv", "avi", "webm"],
            accept_multiple_files=True,
            key="ap_video_uploader",
            help="Upload one or more video clips to process through the full Autopilot pipeline.",
        )

        if uploaded_files:
            c1, c2, c3, c4 = st.columns(4)
            total_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
            with c1:
                st.metric("Videos Queued", len(uploaded_files))
            with c2:
                st.metric("Total Size", f"{total_mb:.1f} MB")
            with c3:
                st.metric("Whisper Model", settings.whisper_model_size.capitalize())
            with c4:
                st.metric("Outro Loaded", "Yes" if settings.outro_path else "No")

            # Save uploaded files into session-managed temp paths
            upload_tmp = Path(tempfile.gettempdir()) / "shorts_autopilot_uploads"
            upload_tmp.mkdir(parents=True, exist_ok=True)
            local_saved_paths = []
            for uf in uploaded_files:
                dest = upload_tmp / uf.name
                dest.write_bytes(uf.read())
                local_saved_paths.append(dest)

    # ── Mode D: Dianisma Channel Library Picker ────────────────────────────────
    elif "DianismaNews" in source_mode:
        if dianisma_videos:
            video_map = {
                f"[{v.upload_date_display}] {v.title} ({v.view_count:,} views • {v.duration_display})": v
                for v in dianisma_videos
            }
            choice = st.selectbox("Choose video from @DianismaNews uploads:", list(video_map.keys()))
            if choice:
                selected_video = video_map[choice]
        else:
            st.warning("No recent videos retrieved from the channel library. Ensure YouTube is connected.")

    st.markdown("---")

    # ── Production Strategy & Audio Bed ────────────────────────────────────────
    st.markdown("### ⚙️ 2. Production Strategy & Audio Configuration")

    strat_col, count_col = st.columns([3, 1])
    with strat_col:
        strategy_label = st.selectbox(
            "Production Mode",
            [
                "🔥 Hybrid: Speaker Clip + AI Breakdown (Recommended)",
                "🤖 Smart Auto-Select (Speaker Detect vs Full AI)",
                "👤 Speaker Preservation (B-Roll + Subtitle Masking)",
                "✨ Full AI Short Generation (AI Script + Voiceover + 9:16 Media)",
            ],
            index=0,
            help=(
                "Hybrid: Extracts an authentic 8-15s speaker hook with YOLO face centering and highlight captions, "
                "then seamlessly weaves in an AI breakdown dossier with 9:16 B-roll. "
                "Auto: dynamically detects on-camera speaker presence. "
                "Speaker: preserves full clip with B-roll cutaways. "
                "AI Gen: generates complete Short from scratch."
            ),
        )
        strategy_map = {
            "🔥 Hybrid: Speaker Clip + AI Breakdown (Recommended)": "hybrid",
            "🤖 Smart Auto-Select (Speaker Detect vs Full AI)": "auto",
            "👤 Speaker Preservation (B-Roll + Subtitle Masking)": "speaker",
            "✨ Full AI Short Generation (AI Script + Voiceover + 9:16 Media)": "ai_gen",
        }
        production_strategy = strategy_map.get(strategy_label, "hybrid")

    with count_col:
        num_shorts = st.number_input(
            "Shorts to Generate",
            min_value=1,
            max_value=5,
            value=len(local_saved_paths) if local_saved_paths else 1,
            step=1,
        )

    # Royalty-Free Background Audio Bed
    st.markdown("#### 🎵 Background Music Bed (Instrumental & Royalty-Free)")
    m_col1, m_col2 = st.columns([2, 1])
    with m_col1:
        bg_choice = st.selectbox(
            "Music Track Bed",
            [
                "🎵 Ambient Calm (Instrumental • Royalty-Free)",
                "🔥 Dramatic Pulse (Instrumental • Royalty-Free)",
                "⚡ Upbeat Groove (Instrumental • Royalty-Free)",
                "🎬 Cinematic Suspense (Instrumental • Royalty-Free)",
                "🚫 None (Voice Only)",
            ],
            index=0,
            help="High-retention instrumental background bed with automated speech ducking (volume dynamically dips during speech).",
        )
        music_map = {
            "🎵 Ambient Calm (Instrumental • Royalty-Free)": "ambient_calm",
            "🔥 Dramatic Pulse (Instrumental • Royalty-Free)": "dramatic_pulse",
            "⚡ Upbeat Groove (Instrumental • Royalty-Free)": "upbeat_groove",
            "🎬 Cinematic Suspense (Instrumental • Royalty-Free)": "cinematic_suspense",
            "🚫 None (Voice Only)": "none",
        }
        selected_track = music_map.get(bg_choice, "ambient_calm")
        settings.enable_bg_music = (selected_track != "none")
        settings.bg_music_track = selected_track
    with m_col2:
        bg_vol = st.slider(
            "Music Volume (Voice Ducking Active)",
            min_value=0.05,
            max_value=0.35,
            value=getattr(settings, "bg_music_volume", 0.15),
            step=0.01,
            format="%.2f",
            help="Subtle background audio level (0.15 = 15%). Voice ducking automatically compresses music volume during narration.",
        )
        settings.bg_music_volume = bg_vol
        settings.bg_music_ducking = True

    # ── Caption Template & Occasion Styling ────────────────────────────────────
    st.markdown("#### 🎨 Caption Template & Occasion Styling")
    cap_c1, cap_c2 = st.columns([2, 1])
    with cap_c1:
        caption_choices = [
            "🤖 Auto-Detect (Dynamic AI Selection based on Occasion)",
            "🏎️ Car Pulse Industrial (Automotive & Workshop)",
            "⚡ Hormozi Punch (High-Energy Bold Yellow Box)",
            "🔥 Viral TikTok Bounce (Neon Lime Green Highlights)",
            "🔮 Neon Cyberpunk (Electric Cyan & Magenta)",
            "✨ Elegant Minimalist (Clean Aesthetic & Luxury)",
            "📰 Documentary Broadcast (Journalistic Translucent Box)",
            "🟡 Classic Yellow Box (Baseline Subtitle)",
        ]
        caption_map = {
            "🤖 Auto-Detect (Dynamic AI Selection based on Occasion)": "auto",
            "🏎️ Car Pulse Industrial (Automotive & Workshop)": "CAR_PULSE_INDUSTRIAL",
            "⚡ Hormozi Punch (High-Energy Bold Yellow Box)": "HORMOZI_PUNCH",
            "🔥 Viral TikTok Bounce (Neon Lime Green Highlights)": "VIRAL_TIKTOK_BOUNCE",
            "🔮 Neon Cyberpunk (Electric Cyan & Magenta)": "NEON_CYBER",
            "✨ Elegant Minimalist (Clean Aesthetic & Luxury)": "ELEGANT_MINIMAL",
            "📰 Documentary Broadcast (Journalistic Translucent Box)": "DOCUMENTARY_CLEAN",
            "🟡 Classic Yellow Box (Baseline Subtitle)": "CLASSIC_YELLOW",
        }
        selected_cap_label = st.selectbox(
            "Caption Style Preset",
            caption_choices,
            index=0,
            help="Choose a visual subtitle theme. When set to Auto-Detect, the AI automatically picks the best template matching the video's content niche (e.g. Car Pulse for car vlogs, Neon for tech, Documentary for news).",
        )
        settings.caption_style = caption_map.get(selected_cap_label, "auto")

    with cap_c2:
        try:
            from services.asset_library import list_assets
            from services.project_memory import list_project_records
            past_projects = list_project_records(limit=10)
            st.metric("AI Memory Projects", len(past_projects), help="Number of past approved video projects indexed in local dataset for continuous few-shot learning.")
        except Exception:
            st.metric("AI Memory Dataset", "Active")

    # Local Project Memory & Asset Library Expander
    with st.expander("🧠 View AI Project Memory & Local Asset Library", expanded=False):
        mem_tab, asset_tab = st.tabs(["📚 Project Memory Dataset", "📁 Local Media Assets"])
        with mem_tab:
            try:
                from services.project_memory import list_project_records
                recs = list_project_records(limit=10)
                if recs:
                    st.caption("The AI continuously references these past projects to emulate proven hooks, pacing, and styling:")
                    for r in recs:
                        st.markdown(
                            f"• **{r.source_title}** (`{r.niche.upper()}` • {r.caption_style}) — Hook: *\"{r.hook_text}\"* "
                            f"(Rating: {'⭐' * r.user_rating})"
                        )
                else:
                    st.info("No past projects recorded yet. As you process videos, approved projects will be automatically remembered here.")
            except Exception as e:
                st.info(f"Memory ready: {e}")

        with asset_tab:
            try:
                from services.asset_library import list_assets
                all_assets = list_assets()
                st.caption(f"Local Asset Repository: `{len(all_assets)}` total items indexed.")
                c_sfx, c_mus, c_fx = st.columns(3)
                with c_sfx:
                    st.markdown("**🔊 Sound Effects (SFX)**")
                    for a in [x for x in all_assets if x.category == "sfx"][:5]:
                        st.text(f"• {a.name} ({a.file_extension})")
                with c_mus:
                    st.markdown("**🎵 Music Tracks**")
                    for a in [x for x in all_assets if x.category == "music"][:5]:
                        st.text(f"• {a.name} ({a.file_extension})")
                with c_fx:
                    st.markdown("**🎬 Visual Effects & Overlays**")
                    for a in [x for x in all_assets if x.category in ("effects", "green_screen")][:5]:
                        st.text(f"• {a.name} ({a.file_extension})")
            except Exception as e:
                st.info(f"Asset library ready: {e}")

    broll_path = st.session_state.get("custom_broll_path", "")
    if broll_path:
        st.success(f"Custom B-Roll Folder Active: `{Path(broll_path).name}`")

    settings_errors = settings.validate()
    if settings_errors:
        for err in settings_errors:
            st.warning(err)

    st.markdown("")

    # ── Launch Autopilot ───────────────────────────────────────────────────────
    if local_saved_paths:
        target_descr = f"{len(local_saved_paths)} Local Video Clip(s)"
    elif selected_video:
        target_descr = f"Selected Video: '{selected_video.title[:30]}...'"
    elif override_url.strip():
        target_descr = f"URL: {override_url.strip()[:40]}"
    elif niche_focus_query.strip():
        target_descr = f"Topic: '{niche_focus_query.strip()}'"
    else:
        target_descr = "Niche Viral Trends (≤ 21 Days Old)"

    btn_label = f"🚀 Launch Autopilot for {target_descr}"
    can_launch = not settings_errors and (
        "Local Video" not in source_mode or len(local_saved_paths) > 0
    )

    if st.button(btn_label, type="primary", use_container_width=True, disabled=not can_launch):
        try:
            from services.autopilot import run_autopilot_pipeline
        except ImportError:
            from shorts_engine.services.autopilot import run_autopilot_pipeline

        st.session_state.pop("autopilot_result", None)

        st.markdown("### ⚡ Autopilot Live Execution")
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
                local_video_paths=local_saved_paths if local_saved_paths else None,
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

    # ── Review & Approve Generated Shorts ──────────────────────────────────────
    autopilot_results = st.session_state.get("autopilot_result")
    if autopilot_results:
        try:
            from ui.components.review_card import render_review_and_approve_list
        except ImportError:
            from shorts_engine.ui.components.review_card import (
                render_review_and_approve_list,
            )
        render_review_and_approve_list(autopilot_results, key_prefix="ap")

    # ── Output Library / Past Generated Shorts Drawer ─────────────────────────
    st.markdown("---")
    with st.expander("📁 Finished Shorts Library (Ready for Upload & Review)", expanded=False):
        st.markdown("#### Previously Rendered Shorts")
        output_dir = Path(settings.output_dir)
        all_clips: list[Path] = []
        if output_dir.is_dir():
            all_clips = sorted(
                output_dir.rglob("*_short.mp4"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

        if not all_clips:
            st.info("No rendered Shorts found in the output directory yet.")
        else:
            st.caption(f"Found {len(all_clips)} finished clip(s) in `{output_dir}`.")
            for idx, clip_path in enumerate(all_clips[:10]):
                seo_json = clip_path.parent / f"seo_{clip_path.stem.replace('_short', '')}.json"
                seo_meta: dict = {}
                if seo_json.is_file():
                    try:
                        seo_meta = json.loads(seo_json.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError) as exc:
                        logger.debug("Failed to load SEO JSON: %s", exc)

                c_vid, c_info = st.columns([1, 2])
                with c_vid:
                    st.video(str(clip_path))
                with c_info:
                    vid_title = st.text_input(
                        "Title",
                        value=seo_meta.get("title", clip_path.stem)[:100],
                        max_chars=100,
                        key=f"out_title_{idx}_{clip_path.name}",
                    )
                    vid_desc = st.text_area(
                        "Description",
                        value=seo_meta.get("description", "")[:5000],
                        height=100,
                        key=f"out_desc_{idx}_{clip_path.name}",
                    )
                    vid_tags = st.text_input(
                        "Tags",
                        value=", ".join(seo_meta.get("tags", [])),
                        key=f"out_tags_{idx}_{clip_path.name}",
                    )
                    if yt_connected:
                        if st.button(f"🚀 Upload to YouTube ({clip_path.name})", key=f"btn_out_up_{idx}", type="secondary"):
                            with st.spinner("Uploading to YouTube..."):
                                try:
                                    tag_l = [t.strip() for t in vid_tags.split(",") if t.strip()]
                                    yt = authenticate()
                                    v_id = upload_short(
                                        youtube_client=yt,
                                        video_path=clip_path,
                                        title=vid_title,
                                        description=vid_desc,
                                        tags=tag_l,
                                    )
                                    st.success(f"🎉 **Uploaded!** [View on YouTube](https://www.youtube.com/watch?v={v_id})")
                                    st.balloons()
                                except (YouTubeUploadError, YouTubeAuthError, RuntimeError, OSError, ValueError) as exc:
                                    st.error(f"Upload failed: {exc}")
                    else:
                        st.caption("Connect your YouTube account above to upload this clip.")
                st.markdown("<hr style='margin: 0.5rem 0; border-color: #27272a;'>", unsafe_allow_html=True)
