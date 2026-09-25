"""
ui/components/result_card.py — Reusable card component for rendering processed Shorts results.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

try:
    from pipeline import ProcessingResult
    from services.seo_generator import get_download_filename
except ImportError:
    from shorts_engine.pipeline import ProcessingResult
    from shorts_engine.services.seo_generator import get_download_filename


def get_result_download_filename(result: ProcessingResult, extension: str = "mp4") -> str:
    """
    Determine the download filename for a processed result.
    Uses the SEO title if present, sanitized for safe cross-platform file saving.
    Falls back to the original output file name.
    """
    seo_title = result.seo.title if (result.seo and result.seo.title) else None
    fallback_name = result.output_file.name if result.output_file else "clip.mp4"
    return get_download_filename(
        title=seo_title,
        fallback_filename=fallback_name,
        extension=extension,
    )


def render_result_card(result: ProcessingResult, index: int) -> None:
    """Render a single ProcessingResult as a styled card."""
    with st.container(border=True):
        col_title, col_status = st.columns([5, 1])
        with col_title:
            card_title = (
                result.seo.title
                if (result.seo and result.seo.title)
                else result.input_file.name
            )
            st.markdown(f"**{index + 1}. {card_title}**")
        with col_status:
            status_badge = (
                '<span class="badge badge-success">Success</span>'
                if result.success
                else '<span class="badge badge-error">Failed</span>'
            )
            st.markdown(status_badge, unsafe_allow_html=True)

        if result.success and result.output_file:
            # Two-column layout: 9:16 vertical video player left, metadata right.
            vid_col, meta_col = st.columns([1, 2.2])

            with vid_col:
                if result.output_file.is_file():
                    st.video(str(result.output_file))
                    dl_filename = get_result_download_filename(result, extension="mp4")
                    st.download_button(
                        label="Download Short (.mp4)",
                        data=result.output_file.read_bytes(),
                        file_name=dl_filename,
                        mime="video/mp4",
                        key=f"video_download_{index}",
                        use_container_width=True,
                    )

                    # Option to set destination folder and save directly on disk
                    with st.expander("📁 Save to Custom Folder", expanded=False):
                        default_dest = st.session_state.get(
                            "custom_output_dir", str(Path.home() / "Downloads")
                        )
                        dest_folder_val = st.text_input(
                            "Destination Folder",
                            value=default_dest,
                            key=f"dest_folder_val_{index}",
                            help="Local directory path to save a copy of this short.",
                        )
                        if st.button(
                            "💾 Save Copy to Folder",
                            key=f"btn_save_dest_{index}",
                            use_container_width=True,
                        ):
                            try:
                                target_dir = Path(dest_folder_val).expanduser().resolve()
                                target_dir.mkdir(parents=True, exist_ok=True)
                                target_file = target_dir / dl_filename
                                shutil.copy2(str(result.output_file), str(target_file))

                                # Also copy companion SEO JSON if available
                                seo_json_name = f"seo_{result.output_file.stem.replace('_short','')}.json"
                                src_seo = result.output_file.parent / seo_json_name
                                if src_seo.is_file():
                                    shutil.copy2(str(src_seo), str(target_dir / seo_json_name))

                                st.success(f"✓ Saved to: `{target_file}`")
                            except Exception as exc:
                                st.error(f"Failed to save: {exc}")

            with meta_col:
                st.markdown("**Output File**")
                st.code(str(result.output_file), language=None)

                # Show viral hook if detected
                if result.hook_text:
                    st.markdown(
                        f'<div style="margin:0.5rem 0;padding:0.5rem 0.75rem;'
                        f'background:#18181b;border-left:2px solid #52525b;border-radius:4px;">'
                        f'<span style="font-size:0.72rem;font-weight:600;color:#a1a1aa;text-transform:uppercase;letter-spacing:0.04em;">Viral Hook '
                        f'(Score: {result.virality_score:.1f}/10)</span><br>'
                        f'<span style="font-size:0.82rem;color:#d4d4d8;font-style:italic;">"{result.hook_text}"</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Show Inflowave retention audit if available
                if getattr(result, "retention_audit", None):
                    audit = result.retention_audit
                    grade_color = "#10b981" if audit.grade in ("S", "A") else ("#f59e0b" if audit.grade == "B" else "#ef4444")
                    st.markdown(
                        f'<div style="margin:0.5rem 0;padding:0.6rem 0.85rem;'
                        f'background:#18181b;border:1px solid #27272a;border-left:4px solid {grade_color};border-radius:6px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<span style="font-weight:700;color:#f4f4f5;font-size:0.88rem;">⚡ Retention Score: {audit.score}/100</span>'
                        f'<span style="background:{grade_color}22;color:{grade_color};font-weight:700;padding:2px 8px;border-radius:4px;font-size:0.8rem;">Grade {audit.grade}</span>'
                        f'</div>'
                        f'<div style="margin-top:0.35rem;font-size:0.75rem;color:#a1a1aa;display:flex;gap:12px;">'
                        f'<span>🎯 Hook Speed: <b>{audit.hook_speed_score}/100</b> ({audit.intro_silence_seconds:.2f}s)</span>'
                        f'<span>⏱️ Pacing: <b>{audit.pacing_cadence_score}/100</b> ({audit.words_per_second:.1f} wps)</span>'
                        f'<span>🔥 Curiosity: <b>{audit.curiosity_gap_score}/100</b></span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if audit.actionable_recommendations:
                        with st.expander("🔍 Retention & Pacing Diagnostics", expanded=False):
                            for rec in audit.actionable_recommendations:
                                st.markdown(f"- {rec}")

                # Show what B-roll query was searched so the user can verify
                if result.broll_query:
                    st.markdown(
                        f'<span style="font-size:0.8rem;color:#71717a;">'
                        f'B-Roll query: <code>{result.broll_query}</code></span>',
                        unsafe_allow_html=True,
                    )

                if result.seo:
                    st.markdown("**SEO Metadata**")
                    st.markdown(f"**Title:** {result.seo.title}")
                    st.markdown(f"**Curiosity Angle:** *{result.seo.curiosity_title}*")
                    st.markdown(f"**Authority Angle:** *{result.seo.authority_title}*")
                    st.markdown(f"**Contrarian Angle:** *{result.seo.contrarian_title}*")
                    if result.seo.primary_keyword:
                        st.markdown(f"**Primary Keyword:** `{result.seo.primary_keyword}`")

                    # Tags as technical badges
                    tags_html = "".join(
                        f'<span class="badge badge-tag">{tag}</span>'
                        for tag in result.seo.tags
                    )
                    st.markdown(tags_html, unsafe_allow_html=True)

                    st.markdown("**YouTube Tags (Comma-separated for YouTube Studio):**")
                    st.code(result.seo.youtube_tags_display, language=None)

                    with st.expander("Full Description & Multi-Platform"):
                        st.markdown("**YouTube Description**")
                        st.text(result.seo.description)
                        if result.seo.pinned_comment:
                            st.markdown("**Pinned Comment**")
                            st.info(result.seo.pinned_comment)

                    # Download SEO JSON
                    seo_json_path = result.output_file.parent / f"seo_{result.output_file.stem.replace('_short','')}.json"
                    if seo_json_path.is_file():
                        seo_dl_filename = get_result_download_filename(result, extension="json")
                        st.download_button(
                            label="Download SEO JSON",
                            data=seo_json_path.read_bytes(),
                            file_name=seo_dl_filename,
                            mime="application/json",
                            key=f"seo_download_{index}",
                        )

        if result.error:
            st.error(f"**Error:** {result.error}")

        if result.warnings:
            for w in result.warnings:
                st.markdown(
                    f'<span class="badge badge-warn">{w}</span>',
                    unsafe_allow_html=True,
                )
