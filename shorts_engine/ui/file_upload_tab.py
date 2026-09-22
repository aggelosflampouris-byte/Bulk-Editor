"""
ui/file_upload_tab.py — Tab component for local file upload and batch processing.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import streamlit as st

try:
    from config import Settings
    from pipeline import ProcessingResult, ProgressCallback, run_batch
    from ui.components.result_card import render_result_card
except ImportError:
    from shorts_engine.config import Settings
    from shorts_engine.pipeline import ProcessingResult, ProgressCallback, run_batch
    from shorts_engine.ui.components.result_card import render_result_card


def _make_progress_callback(
    progress_bar: st.delta_generator.DeltaGenerator,
    stage_text: st.delta_generator.DeltaGenerator,
    total: int,
) -> ProgressCallback:
    """Build a ProgressCallback that updates Streamlit progress UI elements."""
    def callback(current: int, _total: int, stage: str) -> None:
        fraction = current / max(total, 1)
        if "Transcribing audio:" in stage:
            m = re.search(r"\((\d+)%\)", stage)
            if m:
                sub_pct = int(m.group(1)) / 100.0
                fraction = (current + sub_pct * 0.5) / max(total, 1)
        pct = int(min(1.0, fraction) * 100)
        progress_bar.progress(
            min(1.0, fraction),
            text=f"Processing ({pct}%): [{current + 1}/{total}] {stage}",
        )
        stage_text.markdown(
            f"**[{current + 1}/{total}]** {stage}",
            unsafe_allow_html=False,
        )

    return callback


def render_file_upload_tab(settings: Settings) -> None:
    """Render the local file upload and batch processing tab."""
    _, main_col, _ = st.columns([1, 6, 1])
    with main_col:
        st.markdown("### Upload Video Clips")
        uploaded_files = st.file_uploader(
            "Drop your Greek-speech video clips here",
            type=["mp4", "mov", "mkv", "avi", "webm"],
            accept_multiple_files=True,
            key="video_uploader",
            help="Upload one or more video clips. They will be processed in order.",
        )

        if uploaded_files:
            col1, col2, col3, col4 = st.columns(4)
            total_size_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)

            with col1:
                st.markdown(
                    f'<div class="metric-value">{len(uploaded_files)}</div>'
                    f'<div class="metric-label">Videos Queued</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<div class="metric-value">{total_size_mb:.0f} MB</div>'
                    f'<div class="metric-label">Total Upload Size</div>',
                    unsafe_allow_html=True,
                )
            with col3:
                model_label = settings.whisper_model_size.capitalize()
                st.markdown(
                    f'<div class="metric-value">{model_label}</div>'
                    f'<div class="metric-label">Whisper Model</div>',
                    unsafe_allow_html=True,
                )
            with col4:
                outro_label = "Yes" if settings.outro_path else "No"
                st.markdown(
                    f'<div class="metric-value">{outro_label}</div>'
                    f'<div class="metric-label">Outro Loaded</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

        settings_errors = settings.validate()
        if settings_errors:
            for err in settings_errors:
                st.warning(err)

        run_disabled = (
            not uploaded_files
            or bool(settings_errors)
            or st.session_state.get("is_processing", False)
        )

        col_run, col_clear = st.columns([3, 1])
        with col_run:
            run_clicked = st.button(
                "Process All Clips",
                type="primary",
                disabled=run_disabled,
                use_container_width=True,
                key="run_button",
            )
        with col_clear:
            if st.button("Clear Results", use_container_width=True, key="clear_button"):
                st.session_state["results"] = []
                st.rerun()

    # ── File Upload Processing ─────────────────────────────────────────────
    if run_clicked and uploaded_files and not settings_errors:
        st.session_state["is_processing"] = True
        st.session_state["results"] = []

        saved_paths: list[Path] = []
        with st.spinner("Saving uploaded files..."):
            upload_tmp = tempfile.mkdtemp(prefix="shorts_uploads_")
            for uf in uploaded_files:
                dest = Path(upload_tmp) / uf.name
                dest.write_bytes(uf.read())
                saved_paths.append(dest)

        st.markdown("### Processing...")
        progress_bar = st.progress(0.0)
        stage_text = st.empty()

        progress_cb = _make_progress_callback(progress_bar, stage_text, len(saved_paths))

        with st.spinner("Running pipeline..."):
            results = run_batch(
                video_paths=saved_paths,
                settings=settings,
                progress_cb=progress_cb,
            )

        st.session_state["results"] = results
        st.session_state["is_processing"] = False
        st.rerun()

    # ── Render Batch Results ───────────────────────────────────────────────
    batch_results: list[ProcessingResult] = st.session_state.get("results", [])
    if batch_results:
        _, results_col, _ = st.columns([1, 6, 1])
        with results_col:
            st.markdown("---")
            st.markdown(f"### Processed Clips ({len(batch_results)})")
            for idx, res in enumerate(batch_results):
                render_result_card(res, idx)
