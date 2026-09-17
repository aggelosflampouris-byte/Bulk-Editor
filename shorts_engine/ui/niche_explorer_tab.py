import logging
from datetime import datetime

import streamlit as st

from shorts_engine.config import Settings

logger = logging.getLogger(__name__)

def render_niche_explorer_tab(settings: Settings) -> None:
    _, main_col, _ = st.columns([1, 6, 1])
    with main_col:
        st.markdown("### 📊 My Channel Analytics")
        
        # --- AUTH CHECK ---
        try:
            from services.youtube_uploader import (
                authenticate,
                fetch_channel_analytics,
                fetch_my_recent_videos,
                get_analytics_client,
                get_channel_info,
                is_authenticated,
            )
            yt_connected = is_authenticated()
        except ImportError:
            yt_connected = False
        
        analyze_mine_clicked = False
        
        if yt_connected:
            st.success("✅ **YouTube API Connected:** Pulling deep real-time analytics for your channel.")
            
            # Niche Selector & Auto-Picker
            st.markdown("#### 1. Target Niche")
            target_niche = st.text_input("Target Niche / Topic (Optional)", placeholder="e.g. Greek Mythology, Financial Advice", key="target_niche_input", label_visibility="collapsed")
            auto_pick = st.button("✨ Auto-Pick Niche for Me", use_container_width=False)
            
            if auto_pick:
                if not settings.gemini_api_key:
                    st.error("Add Gemini API key in sidebar first.")
                else:
                    with st.spinner("Finding best niche..."):
                        try:
                            yt_client = authenticate()
                            yt_analytics = get_analytics_client()
                            channel_info = get_channel_info(yt_client)
                            analytics = fetch_channel_analytics(yt_analytics, days=30)
                            from google import genai
                            client = genai.Client(api_key=settings.gemini_api_key)
                            prompt = f"Analyze this channel's analytics and suggest ONE highly profitable, low-competition niche for YouTube Shorts. Analytics: {analytics}. Channel: {channel_info.get('title')}. Respond with ONLY the niche topic in 2-5 words. No markdown, no quotes."
                            response = client.models.generate_content(
                                model="gemini-3.5-flash",
                                contents=prompt,
                            )
                            auto_niche = response.text.strip().replace('"', '')
                            st.session_state["target_niche_input"] = auto_niche
                            st.rerun()
                        except Exception as e:
                            st.error(f"Auto-pick failed: {e}")

            st.markdown("#### 2. Scan Depth")
            max_videos_mine = st.number_input("Max recent videos to analyze", min_value=5, max_value=100, value=30, step=5, key="channel_max_mine", label_visibility="collapsed")
            
            st.markdown("#### 3. Execution")
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                analyze_mine_clicked = st.button("📊 Analyze My Channel", use_container_width=True)
            with col_btn2:
                generate_ideas_clicked = st.button(
                    f"🚀 Generate Video for '{target_niche.strip()}'", 
                    type="primary", 
                    use_container_width=True, 
                    disabled=not target_niche.strip()
                )
        else:
            st.warning("⚠️ **YouTube Account Not Connected**")
            st.markdown(
                "Please connect your YouTube channel in the **Upload to YouTube** tab to unlock deep channel analytics and AI insights."
            )

        if yt_connected and generate_ideas_clicked and target_niche.strip():
            if not settings.gemini_api_key:
                st.error("A Gemini API key is required. Add it to the sidebar.")
            else:
                from shorts_engine.services.autopilot import run_autopilot_pipeline
                st.session_state.pop("niche_autopilot_result", None)
                st.markdown("### Auto-Pilot Execution Log")
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_container = st.container()
                
                try:
                    broll_path = st.session_state.get("custom_broll_path", "")
                    for msg, pct, data in run_autopilot_pipeline(target_niche.strip(), settings, broll_path, num_videos=3):
                        progress_bar.progress(pct)
                        status_text.markdown(f"**{pct}%** — {msg}")
                        log_container.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                        
                        if pct == 100 and data:
                            st.session_state["niche_autopilot_result"] = data
                            st.rerun()
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    logger.exception("Niche Autopilot failed")

        # Render Autopilot Review UI if available
        niche_ap_results = st.session_state.get("niche_autopilot_result")
        if niche_ap_results:
            from shorts_engine.ui.components.review_card import render_review_and_approve_list
            render_review_and_approve_list(niche_ap_results, key_prefix="niche_ap")

        # Scan & analyze
        if yt_connected and analyze_mine_clicked:
            if not settings.gemini_api_key:
                st.error("A Gemini API key is required for the AI analysis. Add it to the sidebar.")
            else:
                with st.spinner("Fetching authenticated YouTube Analytics and Data..."):
                    try:
                        yt_client = authenticate()
                        yt_analytics = get_analytics_client()
                        
                        channel_info = get_channel_info(yt_client)
                        channel_name = channel_info.get("title", "My Channel")
                        
                        analytics = fetch_channel_analytics(yt_analytics, days=30)
                        st.session_state["yt_analytics_30d"] = analytics
                        
                        videos = fetch_my_recent_videos(yt_client, max_videos=int(max_videos_mine))
                        st.session_state["channel_videos"] = videos
                        st.session_state["channel_url"] = f"Authenticated Channel: {channel_name}"
                    except Exception as exc:
                        st.error(f"**Analytics scan failed:** {exc}")
                        st.session_state["channel_videos"] = []

                if st.session_state.get("channel_videos"):
                    with st.spinner("Running AI analysis with Gemini..."):
                        try:
                            from services.channel_analyzer import analyze_niche
                            insights = analyze_niche(
                                st.session_state["channel_videos"],
                                st.session_state["channel_url"],
                                gemini_api_key=settings.gemini_api_key,
                                analytics_data=st.session_state.get("yt_analytics_30d"),
                                target_niche=target_niche.strip() if target_niche else None
                            )
                            st.session_state["channel_insights"] = insights
                        except Exception as exc:
                            st.error(f"**AI analysis failed:** {exc}")

        # Render insights
        insights_data = st.session_state.get("channel_insights")
        if insights_data:
            from services.channel_analyzer import NicheInsights
            ins: NicheInsights = insights_data

            st.markdown("---")
            
            # Render analytics summary if we have it
            analytics = st.session_state.get("yt_analytics_30d")
            if analytics:
                st.markdown("### 📊 My Channel Analytics (Last 30 Days)")
                cols = st.columns(3)
                cols[0].metric("Total Views", f"{analytics.get('views', 0):,}")
                cols[1].metric("Subscribers Gained", f"{analytics.get('subscribersGained', 0):,}")
                cols[2].metric("Watch Time (Mins)", f"{analytics.get('estimatedMinutesWatched', 0):,}")
                st.markdown("---")

            st.markdown(
                f"**{ins.total_videos_analysed} videos analysed** for `{ins.query}`"
            )

            # AI Analysis Cards
            if ins.analysis_error:
                st.warning(f"AI analysis partially failed: {ins.analysis_error}")

            if ins.topic_clusters or ins.content_gaps:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 🗂️ Topic Clusters")
                    st.info(ins.topic_clusters or "—")
                    st.markdown("#### 📅 Best Upload Window")
                    st.info(ins.best_upload_window or "—")
                with c2:
                    st.markdown("#### 🕳️ Content Gaps")
                    st.warning(ins.content_gaps or "—")
                    st.markdown("#### 🔥 Virality Patterns")
                    st.success(ins.virality_patterns or "—")

                st.markdown("#### 🎬 Short Recommendations")
                st.markdown(ins.short_recommendations or "—")

            # Top Videos Table
            st.markdown("---")
            st.markdown("#### 📈 Top Performing Videos")
            if ins.top_videos:
                import pandas as pd
                top_df = pd.DataFrame([
                    {
                        "Title": v.title[:60],
                        "Views": f"{v.view_count:,}",
                        "Duration": v.duration_display,
                        "Uploaded": v.upload_date_display,
                        "URL": v.url,
                    }
                    for v in ins.top_videos
                ])
                st.dataframe(top_df, use_container_width=True, hide_index=True)

            # ── Viral Recent Picks (Primary Feature) ──────────────────
            st.markdown("---")
            st.markdown("#### 🔥 Recent Viral Picks — Last 3 Weeks")
            st.caption(
                "Videos uploaded within the last 21 days, scored by virality "
                "(views × recency × engagement rate). Click **Start Clipping** to "
                "extract Shorts immediately."
            )

            viral_list = ins.viral_recent
            if ins.competitor_viral_recent:
                viral_list = ins.competitor_viral_recent
                st.info(f"**Competitor Discovery:** Sourced viral videos across YouTube for the AI-detected niche: `{ins.suggested_search_query}`")

            if viral_list:
                for vrv in viral_list:
                    v = vrv.video
                    with st.container():
                        st.markdown(
                            f"""
<div style="border:1px solid #27272a;border-radius:8px;padding:14px 18px;margin-bottom:10px;background:#111113;">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
<span style="font-size:1.05rem;font-weight:600;color:#f4f4f5;flex:1;">{v.title}</span>
<span style="background:#3f3f46;color:#a1a1aa;border-radius:4px;padding:2px 8px;font-size:0.75rem;">{vrv.virality_label}</span>
  </div>
  <div style="margin-top:6px;font-size:0.78rem;color:#71717a;">
⏱ {v.duration_display} &nbsp;·&nbsp;
👁 {v.view_count:,} views &nbsp;·&nbsp;
👍 {v.like_count:,} likes &nbsp;·&nbsp;
📅 {vrv.days_old} day{"s" if vrv.days_old != 1 else ""} ago &nbsp;·&nbsp;
⚡ Virality score: <strong style="color:#f4f4f5;">{vrv.score_display}</strong>
&nbsp;<a href="{v.url}" target="_blank" style="color:#6366f1;">[Open ↗]</a>
  </div>
</div>
""",
                            unsafe_allow_html=True,
                        )
                        clip_col, _ = st.columns([2, 5])
                        with clip_col:
                            def _on_clip_click(url=v.url):
                                st.session_state["queued_url"] = url
                                st.session_state["queued_url_autostart"] = True
                                st.session_state["active_tab"] = "Video URL"

                            st.button(
                                "🎬 Start Clipping",
                                key=f"clip_viral_{v.video_id}",
                                type="primary",
                                use_container_width=True,
                                on_click=_on_clip_click,
                            )
            else:
                st.info(
                    "No videos uploaded in the last 3 weeks were found that are suitable "
                    "for Short extraction. This may happen if the channel hasn't posted "
                    "recently, or if recent uploads are already Shorts (< 5 min)."
                )

            # ── All-Time Short Candidates (Secondary) ──────────────────
            with st.expander("✂️ All-Time Best for Short Extraction", expanded=False):
                if ins.short_candidates:
                    for i, v in enumerate(ins.short_candidates, 1):
                        with st.container():
                            cc1, cc2 = st.columns([5, 1])
                            with cc1:
                                st.markdown(
                                    f"**{i}. {v.title}** &nbsp;&nbsp; "
                                    f"`{v.duration_display}` · "
                                    f"{v.view_count:,} views · "
                                    f"[Open ↗]({v.url})",
                                    unsafe_allow_html=True,
                                )
                            with cc2:
                                def _on_queue_click(url=v.url):
                                    st.session_state["queued_url"] = url
                                    st.session_state["active_tab"] = "Video URL"

                                st.button(
                                    "Send to URL Tab",
                                    key=f"queue_candidate_{i}",
                                    help="Send this video URL to the Video URL tab for processing.",
                                    on_click=_on_queue_click,
                                )
                else:
                    st.info("No long-form videos found suitable for Short extraction in this batch.")

        elif not analyze_mine_clicked:
            st.markdown("""
        <div style="text-align:center;padding:3rem 0;border:1px dashed #27272a;border-radius:6px;margin-top:1rem;">
            <p style="font-size:0.88rem;color:#71717a;margin:0;">
                Enter a target niche (optional) and click the primary button above to get started
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Chatbot UI added below everything else inside `if insights_data` block, wait I will add it if insights_data exists
        if st.session_state.get("channel_insights"):
            st.markdown("---")
            st.markdown("### 💬 Viral Video Researcher")
            st.markdown("Chat with the AI about your analytics, niche strategy, or ask for script hooks.")
            
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []
            
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            if chat_prompt := st.chat_input("Ask for hook ideas, analysis, or trends..."):
                st.session_state.chat_messages.append({"role": "user", "content": chat_prompt})
                with st.chat_message("user"):
                    st.markdown(chat_prompt)
                    
                with st.chat_message("assistant"):
                    if not settings.gemini_api_key:
                        st.error("Please add a Gemini API key.")
                    else:
                        with st.spinner("Thinking..."):
                            try:
                                from google import genai
                                client = genai.Client(api_key=settings.gemini_api_key)
                                ins = st.session_state["channel_insights"]
                                anal = st.session_state.get("yt_analytics_30d", {})
                                
                                sys_ctx = (
                                    f"You are an expert YouTube Viral Video Researcher.\n"
                                    f"Channel 30-day Analytics:\n{anal}\n\n"
                                    f"Channel Insights:\nTarget Niche: {ins.query}\n"
                                    f"Topics: {ins.topic_clusters}\nGaps: {ins.content_gaps}\n"
                                    f"Recommendations: {ins.short_recommendations}\n"
                                    f"Virality patterns: {ins.virality_patterns}\n"
                                )
                                
                                chat_history = "Chat History:\n"
                                for m in st.session_state.chat_messages[:-1]:
                                    chat_history += f"{m['role'].capitalize()}: {m['content']}\n"
                                    
                                full_prompt = sys_ctx + "\n" + chat_history + f"\nUser: {chat_prompt}\nAssistant:"
                                
                                response = client.models.generate_content(
                                    model="gemini-3.5-flash",
                                    contents=full_prompt,
                                )
                                st.markdown(response.text)
                                st.session_state.chat_messages.append({"role": "assistant", "content": response.text})
                            except Exception as e:
                                st.error(f"Chat failed: {e}")


