"""
app.py — Streamlit UI for the Batch Greek Shorts Processing Engine.

Architecture contract:
  - This file contains ZERO business logic, FFmpeg calls, or model inference.
  - It delegates ALL processing to config.py and pipeline.run_batch().
  - Its sole responsibility is collecting user input, driving the batch,
    and rendering results.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

from config import Settings, assert_system_binaries, inject_ffmpeg_path
from pipeline import ProcessingResult, run_batch

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Greek Shorts Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/YOUR_USERNAME/shorts-engine",
        "Report a bug": "https://github.com/YOUR_USERNAME/shorts-engine/issues",
    },
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark premium background */
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #12102a 50%, #0a1628 100%);
        color: #e8e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(18, 16, 42, 0.95);
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }

    /* Input labels */
    .stTextInput label, .stSelectbox label, .stFileUploader label,
    .stSlider label, .stCheckbox label {
        color: #c4b5fd !important;
        font-weight: 500;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
    }

    /* Text inputs */
    .stTextInput input, .stSelectbox select {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 8px !important;
        color: #e8e8f0 !important;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        padding: 0.6rem 2rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #6d28d9, #4338ca);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
        transform: translateY(-1px);
    }

    /* 9:16 video preview container */
    .video-container-916 {
        width: 100%;
        max-width: 340px;
        margin: 0 auto 1rem auto;
        aspect-ratio: 9 / 16;
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(139, 92, 246, 0.3);
        background: #000;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .video-container-916 video {
        width: 100% !important;
        height: 100% !important;
        object-fit: contain;
    }

    /* Result cards */
    .result-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        transition: border-color 0.2s ease;
    }
    .result-card:hover {
        border-color: rgba(139, 92, 246, 0.5);
    }
    .result-card.success {
        border-left: 3px solid #10b981;
    }
    .result-card.failure {
        border-left: 3px solid #ef4444;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .badge-success { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .badge-error   { background: rgba(239,68,68,0.15);  color: #ef4444;  border: 1px solid rgba(239,68,68,0.3);  }
    .badge-warn    { background: rgba(245,158,11,0.15); color: #f59e0b;  border: 1px solid rgba(245,158,11,0.3); }
    .badge-tag     { background: rgba(139,92,246,0.15); color: #a78bfa;  border: 1px solid rgba(139,92,246,0.3); }

    /* Metric value */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    /* Code blocks */
    pre, code {
        background: rgba(0,0,0,0.3) !important;
        border: 1px solid rgba(139, 92, 246, 0.15) !important;
        border-radius: 6px !important;
        color: #c4b5fd !important;
        font-size: 0.82rem !important;
    }

    /* Divider */
    hr {
        border-color: rgba(139, 92, 246, 0.15) !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #7c3aed, #4f46e5) !important;
    }

    /* Warning / info boxes */
    .stAlert {
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ───────────────────────────────────────────────

def _init_session_state() -> None:
    """Initialise all session state keys with default values if not present."""
    defaults: dict[str, object] = {
        "results": [],
        "is_processing": False,
        "current_stage": "",
        "progress_value": 0.0,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar() -> Settings:
    """
    Render the settings sidebar and return a populated Settings instance.

    API keys are read exclusively from the .env file / environment — no
    sidebar inputs are exposed for them.  A status row shows whether each
    key was found.

    Returns:
        Settings object built from sidebar inputs.
    """
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")

        # ── API Key Status (read-only) ─────────────────────────────────────
        import os as _os
        pexels_loaded = bool(_os.environ.get("PEXELS_API_KEY", "").strip())
        gemini_loaded = bool(_os.environ.get("GEMINI_API_KEY", "").strip())

        st.markdown("### 🔑 API Keys")
        pexels_icon = "✅" if pexels_loaded else "❌"
        gemini_icon = "✅" if gemini_loaded else "❌"
        st.markdown(
            f"{pexels_icon} **Pexels** — {'loaded from .env' if pexels_loaded else 'not found'}  \n"
            f"{gemini_icon} **Gemini** — {'loaded from .env' if gemini_loaded else 'not found'}"
        )
        if not pexels_loaded or not gemini_loaded:
            st.caption("Add missing keys to `shorts_engine/.env` and restart.")

        st.markdown("---")
        st.markdown("### 🎙️ Transcription")
        model_size = st.selectbox(
            "Whisper Model Size",
            options=["tiny", "base", "small", "medium", "large-v3"],
            index=1,  # default: base
            help=(
                "Larger models are more accurate but slower.\n"
                "base ≈ 30s per minute of audio on CPU."
            ),
            key="whisper_model_size_select",
        )

        st.markdown("---")
        st.markdown("### 🎬 B-Roll Overlay")
        broll_start = st.slider(
            "Overlay Start (seconds)",
            min_value=0.0,
            max_value=30.0,
            value=3.0,
            step=0.5,
            help="Seconds from the start of the main video at which B-roll begins.",
            key="broll_start_slider",
        )
        broll_duration = st.slider(
            "Overlay Duration (seconds)",
            min_value=2.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            key="broll_duration_slider",
        )

        st.markdown("---")
        st.markdown("### 🎞️ Outro Bumper")
        outro_file = st.file_uploader(
            "Upload Outro (optional)",
            type=["mp4", "mov", "mkv", "avi"],
            help="If provided, this clip is appended to every processed Short.",
            key="outro_uploader",
        )

        outro_path: Optional[Path] = None
        if outro_file is not None:
            # Persist the uploaded outro to a session-scoped temp file
            if "outro_tmp_path" not in st.session_state:
                import tempfile as _tf
                suffix = Path(outro_file.name).suffix
                tmp = _tf.NamedTemporaryFile(
                    delete=False, suffix=suffix, prefix="outro_"
                )
                tmp.write(outro_file.read())
                tmp.flush()
                tmp.close()
                st.session_state["outro_tmp_path"] = tmp.name
            outro_path = Path(st.session_state["outro_tmp_path"])
            st.success(f"✓ Outro loaded: {outro_file.name}")
        else:
            # Clear stale temp path if user removed the file
            if "outro_tmp_path" in st.session_state:
                del st.session_state["outro_tmp_path"]

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.72rem;color:#555;text-align:center'>"
            "Greek Shorts Engine · CPU-only<br>"
            "Keys loaded from <code>.env</code> — never exposed in UI."
            "</div>",
            unsafe_allow_html=True,
        )

    return Settings(
        whisper_model_size=str(model_size),
        broll_start_offset=float(broll_start),
        broll_overlay_duration=float(broll_duration),
        outro_path=outro_path,
    )


# ── Result Card Renderer ───────────────────────────────────────────────────────

def _render_result_card(result: ProcessingResult, index: int) -> None:
    """Render a single ProcessingResult as a styled card."""
    card_class = "result-card success" if result.success else "result-card failure"
    status_badge = (
        '<span class="badge badge-success">✓ Success</span>'
        if result.success
        else '<span class="badge badge-error">✗ Failed</span>'
    )
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)

    col_title, col_status = st.columns([5, 1])
    with col_title:
        st.markdown(f"**{index + 1}. {result.input_file.name}**")
    with col_status:
        st.markdown(status_badge, unsafe_allow_html=True)

    if result.success and result.output_file:
        # 9:16 video preview container
        if result.output_file.is_file():
            st.markdown(
                '<div class="video-container-916">',
                unsafe_allow_html=True,
            )
            st.video(str(result.output_file))
            st.markdown('</div>', unsafe_allow_html=True)

        col_out, col_seo = st.columns(2)

        with col_out:
            st.markdown("**📁 Output file**")
            st.code(str(result.output_file), language=None)

        with col_seo:
            if result.seo:
                st.markdown("**📝 SEO Metadata**")
                st.markdown(f"**Title:** {result.seo.title}")

                # Tags as coloured badges
                tags_html = "".join(
                    f'<span class="badge badge-tag">#{tag}</span>'
                    for tag in result.seo.tags
                )
                st.markdown(tags_html, unsafe_allow_html=True)

                with st.expander("Full Description"):
                    st.text(result.seo.description)

                # Download SEO JSON button
                seo_json_path = result.output_file.parent / f"seo_{result.output_file.stem.replace('_short','')}.json"
                if seo_json_path.is_file():
                    st.download_button(
                        label="⬇ Download SEO JSON",
                        data=seo_json_path.read_bytes(),
                        file_name=seo_json_path.name,
                        mime="application/json",
                        key=f"seo_download_{index}",
                    )

        # Download processed video
        if result.output_file.is_file():
            st.download_button(
                label="⬇ Download Short (.mp4)",
                data=result.output_file.read_bytes(),
                file_name=result.output_file.name,
                mime="video/mp4",
                key=f"video_download_{index}",
            )

    if result.error:
        st.error(f"**Error:** {result.error}")

    if result.warnings:
        for w in result.warnings:
            st.markdown(
                f'<span class="badge badge-warn">⚠ {w}</span>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ── Progress Callback ──────────────────────────────────────────────────────────

def _make_progress_callback(
    progress_bar: "st.delta_generator.DeltaGenerator",
    stage_text: "st.delta_generator.DeltaGenerator",
    total: int,
) -> "pipeline.ProgressCallback":
    """
    Build a ProgressCallback that updates Streamlit progress UI elements.

    Args:
        progress_bar: st.progress widget reference.
        stage_text:   st.empty widget for the stage label.
        total:        Total number of videos in the batch.

    Returns:
        A callable matching the ProgressCallback signature.
    """
    def callback(current: int, _total: int, stage: str) -> None:
        fraction = current / max(total, 1)
        progress_bar.progress(fraction)
        stage_text.markdown(
            f"**[{current + 1}/{total}]** {stage}",
            unsafe_allow_html=False,
        )

    return callback


def main() -> None:
    """Entry point for the Streamlit application."""
    _init_session_state()

    # Inject bundled FFmpeg into PATH before any processing
    inject_ffmpeg_path()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    settings = _render_sidebar()

    # ── Hero Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem;">
        <div style="font-size:3rem;margin-bottom:0.5rem;">🎬</div>
        <h1 style="font-size:2.2rem;font-weight:700;margin:0;
                   background:linear-gradient(135deg,#a78bfa,#60a5fa);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Greek Shorts Engine
        </h1>
        <p style="color:#888;margin-top:0.4rem;font-size:1rem;">
            Bulk-convert raw Greek clips → 9:16 YouTube Shorts with subtitles, B-roll &amp; SEO
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── System Check ───────────────────────────────────────────────────────────
    try:
        assert_system_binaries()
    except RuntimeError as exc:
        st.error(f"⛔ **System requirement not met:**\n\n{exc}")
        st.stop()

    # ── Centred Content Column ─────────────────────────────────────────────────
    _, main_col, _ = st.columns([1, 6, 1])
    with main_col:

        # ── File Uploader ──────────────────────────────────────────────────────
        st.markdown("### 📂 Upload Raw Video Clips")
        uploaded_files = st.file_uploader(
            "Drop your Greek-speech video clips here",
            type=["mp4", "mov", "mkv", "avi", "webm"],
            accept_multiple_files=True,
            key="video_uploader",
            help="Upload one or more video clips. They will be processed in order.",
        )

        # ── Metrics Row ───────────────────────────────────────────────────────
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
                outro_label = "✓ Yes" if settings.outro_path else "—"
                st.markdown(
                    f'<div class="metric-value">{outro_label}</div>'
                    f'<div class="metric-label">Outro Loaded</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

        # ── Validate Settings Before Allowing Run ──────────────────────────────
        settings_errors = settings.validate()
        if settings_errors:
            for err in settings_errors:
                st.warning(f"⚠️ {err}")

        # ── Run Button ─────────────────────────────────────────────────────────
        run_disabled = (
            not uploaded_files
            or bool(settings_errors)
            or st.session_state["is_processing"]
        )

        col_run, col_clear = st.columns([3, 1])
        with col_run:
            run_clicked = st.button(
                "🚀 Process All Clips",
                type="primary",
                disabled=run_disabled,
                use_container_width=True,
                key="run_button",
            )
        with col_clear:
            if st.button("🗑 Clear Results", use_container_width=True, key="clear_button"):
                st.session_state["results"] = []
                st.rerun()

    # ── Processing (outside the column so progress spans full width) ───────────
    if run_clicked and uploaded_files and not settings_errors:
        st.session_state["is_processing"] = True
        st.session_state["results"] = []

        # Save uploaded files to a persistent temp directory
        # (Streamlit UploadedFile objects are BytesIO; we need real paths for FFmpeg)
        saved_paths: list[Path] = []
        with st.spinner("Saving uploaded files..."):
            upload_tmp = tempfile.mkdtemp(prefix="shorts_uploads_")
            for uf in uploaded_files:
                dest = Path(upload_tmp) / uf.name
                dest.write_bytes(uf.read())
                saved_paths.append(dest)

        # Progress indicators
        st.markdown("### ⏳ Processing...")
        progress_bar = st.progress(0.0)
        stage_text = st.empty()

        progress_cb = _make_progress_callback(progress_bar, stage_text, len(saved_paths))

        with st.spinner("Running pipeline..."):
            results = run_batch(
                video_paths=saved_paths,
                settings=settings,
                progress_cb=progress_cb,
            )

        progress_bar.progress(1.0)
        stage_text.markdown("**✓ Batch complete!**")
        st.session_state["results"] = results
        st.session_state["is_processing"] = False

        # Clean up the upload temp dir
        import shutil as _shutil
        _shutil.rmtree(upload_tmp, ignore_errors=True)

        st.rerun()

    # ── Results ────────────────────────────────────────────────────────────────
    results: list[ProcessingResult] = st.session_state.get("results", [])
    if results:
        n_success = sum(1 for r in results if r.success)
        n_fail = len(results) - n_success

        st.markdown("### 📊 Results")
        cols = st.columns(3)
        with cols[0]:
            st.markdown(
                f'<div class="metric-value" style="color:#10b981">{n_success}</div>'
                f'<div class="metric-label">Succeeded</div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            color = "#ef4444" if n_fail > 0 else "#10b981"
            st.markdown(
                f'<div class="metric-value" style="color:{color}">{n_fail}</div>'
                f'<div class="metric-label">Failed</div>',
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                f'<div class="metric-value">{len(results)}</div>'
                f'<div class="metric-label">Total Processed</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        for idx, result in enumerate(results):
            _render_result_card(result, idx)

    elif not uploaded_files:
        # Empty state illustration
        st.markdown("""
        <div style="text-align:center;padding:4rem 0;opacity:0.4;">
            <div style="font-size:5rem;">🎥</div>
            <p style="font-size:1.1rem;margin-top:1rem;">
                Upload Greek video clips to get started
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

