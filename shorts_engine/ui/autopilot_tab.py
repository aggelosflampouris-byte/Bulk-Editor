import logging
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

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
    
    col1, _ = st.columns([1, 1])
    with col1:
        st.info("Ensure you have YouTube authentication tokens saved, as Autopilot will automatically schedule uploads.", icon="ℹ️")
        
    broll_path = st.session_state.get("custom_broll_path", "")
    if broll_path:
        st.success(f"Using Custom B-Roll: {Path(broll_path).name}")

    if st.button("🚀 Launch Autopilot", type="primary", use_container_width=True):
        if not channel_url:
            st.error("Please enter a channel URL.")
            return
            
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
        
        try:
            for msg, pct, data in run_autopilot_pipeline(channel_url, settings, broll_path):
                progress_bar.progress(pct)
                status_text.markdown(f"**{pct}%** — {msg}")
                log_container.write(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}")
                
                if pct == 100 and data:
                    st.session_state["autopilot_result"] = data
                    st.rerun()
        except Exception as e:
            st.error(f"Autopilot failed: {e}")
            logger.exception("Autopilot pipeline error")

    # Outside the button, render the review UI
    autopilot_results = st.session_state.get("autopilot_result")
    if autopilot_results:
        try:
            from ui.components.review_card import render_review_and_approve_list
        except ImportError:
            from shorts_engine.ui.components.review_card import (
                render_review_and_approve_list,
            )
        render_review_and_approve_list(autopilot_results, key_prefix="ap")
