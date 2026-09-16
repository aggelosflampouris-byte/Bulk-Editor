import streamlit as st
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

def render_autopilot_tab(settings) -> None:
    st.markdown("""
<div style="background: linear-gradient(135deg, #1f1c2c, #928DAB); padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
    <h2 style="color: white; margin-top: 0;">🤖 Autopilot Mode</h2>
    <p style="color: #e0e0e0; font-size: 1.1rem; margin-bottom: 0;">
        Enter a YouTube channel URL to automatically analyze, download, transcribe, clip, compose, SEO-optimise, and schedule the most viral video for upload.
    </p>
</div>
""", unsafe_allow_html=True)

    channel_url = st.text_input("YouTube Channel URL (e.g. @MrBeast)", placeholder="https://www.youtube.com/@MrBeast")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.info("Ensure you have YouTube authentication tokens saved, as Autopilot will automatically schedule uploads.", icon="ℹ️")
        
    broll_path = st.session_state.get("custom_broll_path", "")
    if broll_path:
        st.success(f"Using Custom B-Roll: {Path(broll_path).name}")

    if st.button("🚀 Launch Autopilot", type="primary", use_container_width=True):
        if not channel_url:
            st.error("Please enter a channel URL.")
            return
            
        from shorts_engine.services.autopilot import run_autopilot_pipeline
        
        # Clear previous result
        st.session_state.pop("autopilot_result", None)
        
        st.markdown("### Execution Log")
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.container()
        
        try:
            for msg, pct, data in run_autopilot_pipeline(channel_url, settings, broll_path):
                progress_bar.progress(pct)
                status_text.markdown(f"**{pct}%** — {msg}")
                log_container.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                
                if pct == 100 and data:
                    st.session_state["autopilot_result"] = data
                    st.rerun()
        except Exception as e:
            st.error(f"Autopilot failed: {e}")
            logger.exception("Autopilot pipeline error")

    # Outside the button, render the review UI
    autopilot_results = st.session_state.get("autopilot_result")
    if autopilot_results:
        st.markdown("---")
        st.markdown(f"### 📝 Review & Approve Auto-Generated Videos")
        
        for idx, res in enumerate(autopilot_results):
            with st.expander(f"🎬 Video {idx+1}: {res['seo'].title}", expanded=True):
                video_path = res["path"]
                seo = res["seo"]
                publish_at = res["publish_at"]
                
                col_vid, col_meta = st.columns([1, 2])
                with col_vid:
                    st.video(str(video_path))
                    
                with col_meta:
                    edit_title = st.text_input("Title", value=seo.title, key=f"ap_title_{idx}")
                    edit_desc = st.text_area("Description", value=seo.description, height=150, key=f"ap_desc_{idx}")
                    edit_tags = st.text_input("Tags (comma separated)", value=", ".join(seo.tags), key=f"ap_tags_{idx}")
                    st.info(f"Scheduled for: **{publish_at.strftime('%Y-%m-%d %H:%M UTC')}**")
                    
                    if st.button(f"✅ Approve & Schedule Upload (Video {idx+1})", type="primary", use_container_width=True, key=f"ap_btn_{idx}"):
                        with st.spinner("Uploading to YouTube..."):
                            try:
                                from shorts_engine.services.youtube_uploader import upload_short, authenticate
                                tag_list = [t.strip() for t in edit_tags.split(",") if t.strip()]
                                
                                yt = authenticate()
                                video_id = upload_short(
                                    youtube_client=yt,
                                    video_path=video_path,
                                    title=edit_title,
                                    description=edit_desc,
                                    tags=tag_list,
                                    publish_at=publish_at
                                )
                                st.success(f"🎉 **Upload Complete!** [View on YouTube Studio](https://www.youtube.com/watch?v={video_id})")
                                st.balloons()
                            except Exception as e:
                                st.error(f"Upload failed: {e}")
