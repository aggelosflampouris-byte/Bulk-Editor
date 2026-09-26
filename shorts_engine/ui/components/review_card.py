import logging
from typing import Any

import streamlit as st

try:
    from services.youtube_uploader import authenticate, upload_short
except ImportError:
    from shorts_engine.services.youtube_uploader import authenticate, upload_short

logger = logging.getLogger(__name__)

def render_review_and_approve_list(results: list[dict[str, Any]], key_prefix: str) -> None:
    """
    Renders a list of auto-generated videos for user review and approval.
    Handles the UI for editing metadata and the YouTube upload API call.
    """
    if not results:
        return

    st.markdown("---")
    st.markdown(f"### 📝 Review & Approve {len(results)} Auto-Generated Videos")
    
    for idx, res in enumerate(results):
        with st.expander(f"🎬 Video {idx+1}: {res['seo'].title}", expanded=True):
            video_path = res["path"]
            seo = res["seo"]
            publish_at = res["publish_at"]
            
            col_vid, col_meta = st.columns([1, 2])
            with col_vid:
                st.video(str(video_path))
                
            with col_meta:
                edit_title = st.text_input("Title", value=seo.title, key=f"{key_prefix}_title_{idx}")
                edit_desc = st.text_area("Description", value=seo.description, height=150, key=f"{key_prefix}_desc_{idx}")
                edit_tags = st.text_input("Tags (comma separated)", value=", ".join(seo.tags), key=f"{key_prefix}_tags_{idx}")

                schedule_info = res.get("schedule_display")
                if not schedule_info:
                    schedule_info = f"📅 Scheduled for: **{publish_at.strftime('%Y-%m-%d %H:%M UTC')}**"
                st.info(schedule_info)

                if st.button(f"✅ Approve & Schedule Upload (Video {idx+1})", type="primary", use_container_width=True, key=f"{key_prefix}_upload_{idx}"):
                    with st.spinner("Uploading to YouTube..."):
                        try:
                            tag_list = [t.strip() for t in edit_tags.split(",") if t.strip()]
                            yt = authenticate()
                            video_id = upload_short(
                                youtube_client=yt,
                                video_path=video_path,
                                title=edit_title,
                                description=edit_desc,
                                tags=tag_list,
                                publish_at=publish_at,
                            )
                            st.success(f"🎉 **Upload Complete!** [View on YouTube Studio](https://www.youtube.com/watch?v={video_id})")
                            st.balloons()
                        except (RuntimeError, OSError, ValueError) as e:
                            logger.exception("Upload failed.")
                            st.error(f"Upload failed: {e}")
