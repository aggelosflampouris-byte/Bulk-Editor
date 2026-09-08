"""
app.py — Streamlit UI for the Greek Shorts Processing Engine.

Architecture contract:
  - This file contains ZERO business logic, FFmpeg calls, or model inference.
  - It delegates ALL processing to config.py, pipeline.run_batch(), and
    pipeline.run_url_pipeline().
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
from pipeline import ProcessingResult, run_batch, run_url_pipeline
from services.clip_selector import ClipCandidate
from services.downloader import probe_url_metadata

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

    /* Constrain st.video() player to a 9:16 portrait shape.
       Streamlit renders video as an <iframe> inside [data-testid=stVideo].
       We target it inside .video-col to avoid affecting other players. */
    .video-col [data-testid="stVideo"],
    .video-col [data-testid="stVideo"] > div,
    .video-col [data-testid="stVideo"] iframe {
        width: 100% !important;
        aspect-ratio: 9 / 16 !important;
        height: auto !important;
        border-radius: 12px;
        overflow: hidden;
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
        # File upload pipeline state
        "results": [],
        "is_processing": False,
        "current_stage": "",
        "progress_value": 0.0,
        # URL pipeline state
        "url_candidates": [],       # list[ClipCandidate] from AI selection
        "url_results": [],          # list[ProcessingResult] from assembly
        "url_is_analyzing": False,  # True while download+transcribe+select runs
        "url_is_rendering": False,  # True while per-clip assembly runs
        "url_meta_title": "",       # Source video title (for display)
        "url_meta_channel": "",
        "url_meta_duration": "",
        "url_selected_indices": [], # list[int] of user-checked clip indices
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
            index=4,  # default: large-v3 for maximum Greek accuracy & millisecond timestamps
            help=(
                "faster-whisper-large-v3 delivers highest Greek accuracy and millisecond timestamps.\n"
                "Use small or base for fast 1–2 minute previews on CPU for long YouTube videos."
            ),
            key="whisper_model_size_select",
        )
        whisper_beam_size = st.select_slider(
            "Whisper Beam Size",
            options=[1, 2, 5],
            value=1,
            help=(
                "1 = greedy search (3x faster on CPU, highly recommended).\n"
                "5 = full beam search (highest accuracy, slowest on CPU)."
            ),
            key="whisper_beam_size_select",
        )

        st.markdown("---")
        st.markdown("### 🧠 Content Analysis & Viral Hooks")
        enable_highlight_scoring = st.checkbox(
            "Qwen 2.5 Highlight Scoring",
            value=True,
            help="Analyzes Greek transcript to detect high-retention 30–60s hooks and viral cut points using Qwen 2.5-32B/72B (with Gemini fallback).",
            key="enable_highlight_scoring_check",
        )
        with st.expander("Qwen LLM Endpoint Config", expanded=False):
            qwen_api_base = st.text_input(
                "API Base URL",
                value=_os.environ.get("QWEN_API_BASE", "http://localhost:11434/v1"),
                help="OpenAI-compatible REST endpoint (Ollama, vLLM, OpenRouter, etc.)",
                key="qwen_api_base_input",
            )
            qwen_model = st.text_input(
                "Model Name",
                value=_os.environ.get("QWEN_MODEL", "qwen2.5:32b"),
                help="Model name identifier, e.g. qwen2.5:32b or qwen2.5:72b",
                key="qwen_model_input",
            )

        st.markdown("---")
        st.markdown("### 🎯 Dynamic 9:16 Auto-Framing")
        enable_face_tracking = st.checkbox(
            "YOLO Face Speaker Tracking",
            value=True,
            help="Tracks active speaker coordinates frame-by-frame with EMA smoothing to keep the subject centered when cropping 16:9 to 9:16 vertical Shorts.",
            key="enable_face_tracking_check",
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
        st.markdown("### ✂️ Clip Selection (URL Mode)")
        max_clips = st.slider(
            "Max Clips per Video",
            min_value=3,
            max_value=10,
            value=10,
            step=1,
            help="Maximum number of clips Gemini will select from the source video (at least 3).",
            key="max_clips_slider",
        )
        clip_min_dur = st.slider(
            "Min Clip Duration (s)",
            min_value=20,
            max_value=45,
            value=35,
            step=5,
            key="clip_min_dur_slider",
        )
        clip_max_dur = st.slider(
            "Max Clip Duration (s)",
            min_value=35,
            max_value=60,
            value=50,
            step=5,
            key="clip_max_dur_slider",
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
        whisper_beam_size=int(whisper_beam_size),
        enable_highlight_scoring=bool(enable_highlight_scoring),
        qwen_api_base=str(qwen_api_base).strip(),
        qwen_model=str(qwen_model).strip(),
        enable_face_tracking=bool(enable_face_tracking),
        broll_start_offset=float(broll_start),
        broll_overlay_duration=float(broll_duration),
        min_clips=3,
        max_clips=int(max_clips),
        clip_min_duration=float(clip_min_dur),
        clip_max_duration=float(max(clip_max_dur, clip_min_dur + 5)),
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
        # Two-column layout: video player left, metadata right.
        # Using st.columns() to constrain width — st.video() cannot be
        # nested inside an HTML <div> injected via unsafe_allow_html.
        vid_col, meta_col = st.columns([2, 3])

        with vid_col:
            st.markdown('<div class="video-col">', unsafe_allow_html=True)
            if result.output_file.is_file():
                st.video(str(result.output_file))
                st.download_button(
                    label="⬇ Download Short (.mp4)",
                    data=result.output_file.read_bytes(),
                    file_name=result.output_file.name,
                    mime="video/mp4",
                    key=f"video_download_{index}",
                    use_container_width=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with meta_col:
            st.markdown("**📁 Output file**")
            st.code(str(result.output_file), language=None)

            # Show viral hook if detected
            if result.hook_text:
                st.markdown(
                    f'<div style="margin:0.4rem 0 0.6rem 0;padding:0.4rem 0.7rem;'
                    f'background:rgba(124,58,237,0.12);border-left:3px solid #7c3aed;border-radius:6px;">'
                    f'<span style="font-size:0.75rem;font-weight:600;color:#c4b5fd;">🎯 VIRAL HOOK '
                    f'(Virality: {result.virality_score:.1f}/10)</span><br>'
                    f'<span style="font-size:0.83rem;color:#e8e8f0;font-style:italic;">"{result.hook_text}"</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Show what B-roll query was searched so the user can verify
            if result.broll_query:
                st.markdown(
                    f'<span style="font-size:0.8rem;color:#888;">'
                    f'🔍 B-roll query: <code>{result.broll_query}</code></span>',
                    unsafe_allow_html=True,
                )

            if result.seo:
                st.markdown("**📝 SEO Metadata**")
                st.markdown(f"**Title:** {result.seo.title}")

                # Tags as coloured badges
                tags_html = "".join(
                    f'<span class="badge badge-tag">{tag}</span>'
                    for tag in result.seo.tags
                )
                st.markdown(tags_html, unsafe_allow_html=True)

                st.markdown("**📋 YouTube Tags (Comma-separated for YouTube Studio):**")
                st.code(result.seo.youtube_tags_display, language=None)

                with st.expander("Full Description"):
                    st.text(result.seo.description)

                # Download SEO JSON
                seo_json_path = result.output_file.parent / f"seo_{result.output_file.stem.replace('_short','')}.json"
                if seo_json_path.is_file():
                    st.download_button(
                        label="⬇ Download SEO JSON",
                        data=seo_json_path.read_bytes(),
                        file_name=seo_json_path.name,
                        mime="application/json",
                        key=f"seo_download_{index}",
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
        if "Transcribing audio:" in stage:
            import re as _re
            m = _re.search(r"\((\d+)%\)", stage)
            if m:
                sub_pct = int(m.group(1)) / 100.0
                fraction = (current + sub_pct * 0.5) / max(total, 1)
        progress_bar.progress(min(1.0, fraction))
        stage_text.markdown(
            f"**[{current + 1}/{total}]** {stage}",
            unsafe_allow_html=False,
        )

    return callback


# ── URL Candidate Table ────────────────────────────────────────────────────────


def _render_url_candidate_table(candidates: list[ClipCandidate]) -> list[int]:
    """
    Render an interactive table of AI-selected clip candidates.

    Each row has a checkbox, clip index, duration, time range, SEO title,
    and hook summary. Returns the list of 1-based clip indices that remain
    checked when the user proceeds to rendering.

    Args:
        candidates: List of ClipCandidate objects from clip_selector.

    Returns:
        List of 1-based clip indices that are selected for rendering.
    """
    selected_indices: list[int] = []

    st.markdown(
        "<div style='font-size:0.85rem;color:#888;margin-bottom:0.8rem;'>"
        "Review the AI-selected clips below. Uncheck any you don't want to render."
        "</div>",
        unsafe_allow_html=True,
    )

    # Column headers
    hdr_cols = st.columns([0.5, 0.5, 1.5, 2.5, 3])
    headers = ["", "#", "Duration", "Time Range", "AI SEO Title"]
    for col, hdr in zip(hdr_cols, headers):
        col.markdown(f"<span style='font-size:0.75rem;color:#888;font-weight:600;"
                     f"text-transform:uppercase;letter-spacing:0.05em'>{hdr}</span>",
                     unsafe_allow_html=True)

    st.markdown("<hr style='margin:0.4rem 0;border-color:rgba(139,92,246,0.15)'>",
                unsafe_allow_html=True)

    for clip in candidates:
        row_cols = st.columns([0.5, 0.5, 1.5, 2.5, 3])

        with row_cols[0]:
            checked = st.checkbox(
                label="",
                value=True,
                key=f"clip_check_{clip.index}",
                label_visibility="collapsed",
            )

        with row_cols[1]:
            st.markdown(
                f"<span style='font-size:0.9rem;font-weight:600;color:#a78bfa'>"
                f"#{clip.index}</span>",
                unsafe_allow_html=True,
            )

        with row_cols[2]:
            st.markdown(
                f"<span style='font-size:0.88rem;color:#e8e8f0'>"
                f"{clip.duration_display}</span>",
                unsafe_allow_html=True,
            )

        with row_cols[3]:
            st.markdown(
                f"<span style='font-size:0.8rem;color:#888'>"
                f"{clip.start_display} → {clip.end_display}</span>",
                unsafe_allow_html=True,
            )

        with row_cols[4]:
            # Truncate title for display
            display_title = clip.seo.title[:55] + "…" if len(clip.seo.title) > 55 else clip.seo.title
            st.markdown(
                f"<span style='font-size:0.88rem;color:#e8e8f0'>"
                f"<strong>{display_title}</strong></span>",
                unsafe_allow_html=True,
            )

        # Hook, SEO & tags in an expander below the row
        with st.expander(f"↳ Hook, SEO & Tags — Clip #{clip.index}", expanded=False):
            st.markdown(f"**Hook:** {clip.hook_summary}")
            st.markdown(f"**B-Roll query:** `{clip.broll_query}`")
            st.markdown(f"**AI Title:** {clip.seo.title}")
            st.markdown(f"**Description:** {clip.seo.description}")
            st.markdown("**📋 YouTube Tags (Copy & Paste directly into YouTube Studio Tags box):**")
            st.code(clip.seo.youtube_tags_display, language=None)

        if checked:
            selected_indices.append(clip.index)

    return selected_indices


def _render_url_results(results: list[ProcessingResult], candidates: list[ClipCandidate]) -> None:
    """
    Render assembled URL pipeline results alongside their clip-level metadata.

    Enriches each ProcessingResult card with the matching ClipCandidate's
    hook summary, so the UI shows the virality rationale alongside the video.

    Args:
        results:    List of ProcessingResult from run_url_pipeline.
        candidates: List of ClipCandidate (for matching hook summaries).
    """
    candidate_map: dict[str, ClipCandidate] = {
        f"clip_{c.index:02d}": c for c in candidates
    }

    n_success = sum(1 for r in results if r.success)
    n_fail = len(results) - n_success

    st.markdown("### 📊 Shorts Generated")
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.markdown(
            f'<div class="metric-value" style="color:#10b981">{n_success}</div>'
            f'<div class="metric-label">Succeeded</div>',
            unsafe_allow_html=True,
        )
    with summary_cols[1]:
        color = "#ef4444" if n_fail > 0 else "#10b981"
        st.markdown(
            f'<div class="metric-value" style="color:{color}">{n_fail}</div>'
            f'<div class="metric-label">Failed</div>',
            unsafe_allow_html=True,
        )
    with summary_cols[2]:
        st.markdown(
            f'<div class="metric-value">{len(results)}</div>'
            f'<div class="metric-label">Total Clips</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    for idx, result in enumerate(results):
        # Attempt to match this result to its ClipCandidate by filename stem
        stem = result.output_file.stem.replace("_short", "") if result.output_file else ""
        matched_clip = candidate_map.get(stem)

        _render_result_card(result, idx)

        # Show hook summary below the card if we have it
        if matched_clip and result.success:
            st.markdown(
                f'<div style="margin:-0.6rem 0 1rem 0;padding:0.5rem 0.8rem;'
                f'background:rgba(139,92,246,0.06);border-radius:6px;'
                f'font-size:0.8rem;color:#888;">'
                f'💡 <em>{matched_clip.hook_summary}</em>'
                f'</div>',
                unsafe_allow_html=True,
            )


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

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_upload, tab_url = st.tabs(["📂 File Upload", "🔗 Video URL"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — FILE UPLOAD (existing behaviour, untouched)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_upload:
        _, main_col, _ = st.columns([1, 6, 1])
        with main_col:

            st.markdown("### 📂 Upload Raw Video Clips")
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
                    outro_label = "✓ Yes" if settings.outro_path else "—"
                    st.markdown(
                        f'<div class="metric-value">{outro_label}</div>'
                        f'<div class="metric-label">Outro Loaded</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("")

            settings_errors = settings.validate()
            if settings_errors:
                for err in settings_errors:
                    st.warning(f"⚠️ {err}")

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

            import shutil as _shutil
            _shutil.rmtree(upload_tmp, ignore_errors=True)

            st.rerun()

        # ── Upload Results ─────────────────────────────────────────────────────
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
            st.markdown("""
            <div style="text-align:center;padding:4rem 0;opacity:0.4;">
                <div style="font-size:5rem;">🎥</div>
                <p style="font-size:1.1rem;margin-top:1rem;">
                    Upload Greek video clips to get started
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — VIDEO URL (new two-stage Analyze → Render flow)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_url:
        _, url_col, _ = st.columns([1, 6, 1])
        with url_col:

            st.markdown("### 🔗 Paste a Video URL")
            st.markdown(
                "<div style='font-size:0.85rem;color:#888;margin-bottom:0.8rem;'>"
                "Supports YouTube, Vimeo, and direct MP4/video links. "
                "The engine will transcribe the full video once, then Gemini will "
                "select the best moments as clips."
                "</div>",
                unsafe_allow_html=True,
            )

            url_input = st.text_input(
                "Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                key="url_input",
                label_visibility="collapsed",
            )

            settings_errors_url = settings.validate()
            if settings_errors_url:
                for err in settings_errors_url:
                    st.warning(f"⚠️ {err}")

            # ── Stage 1: Analyze Button ────────────────────────────────────────
            col_analyze, col_url_clear = st.columns([3, 1])
            with col_analyze:
                analyze_clicked = st.button(
                    "🔍 Analyze Video",
                    type="primary",
                    disabled=(
                        not url_input.strip()
                        or bool(settings_errors_url)
                        or st.session_state["url_is_analyzing"]
                        or st.session_state["url_is_rendering"]
                    ),
                    use_container_width=True,
                    key="analyze_button",
                )
            with col_url_clear:
                if st.button("🗑 Clear", use_container_width=True, key="url_clear_button"):
                    for key in [
                        "url_candidates", "url_results", "url_is_analyzing",
                        "url_is_rendering", "url_meta_title", "url_meta_channel",
                        "url_meta_duration", "url_selected_indices",
                    ]:
                        st.session_state[key] = [] if "candidates" in key or "results" in key or "indices" in key else (False if "is_" in key else "")
                    st.rerun()

        # ── Analyze Execution (outside column for full-width progress) ─────────
        if analyze_clicked and url_input.strip() and not settings_errors_url:
            st.session_state["url_is_analyzing"] = True
            st.session_state["url_candidates"] = []
            st.session_state["url_results"] = []

            st.markdown("### ⏳ Analyzing...")
            url_progress_bar = st.progress(0.0)
            url_stage_text = st.empty()

            stage_messages = [
                "Probing URL...",
                "Downloading video...",
                "Transcribing audio...",
                "Correcting transcript...",
                "Selecting best clips with AI...",
            ]
            stage_idx = [0]

            def _url_analysis_cb(current: int, total: int, stage: str) -> None:
                url_stage_text.markdown(f"**{stage}**")
                # Advance simulated progress based on stage keywords
                if "Probing" in stage:
                    url_progress_bar.progress(0.05)
                elif "Downloading" in stage:
                    url_progress_bar.progress(0.15)
                elif "Transcribing audio:" in stage:
                    import re as _re
                    m = _re.search(r"\((\d+)%\)", stage)
                    if m:
                        pct_val = int(m.group(1)) / 100.0
                        url_progress_bar.progress(min(0.70, 0.20 + 0.50 * pct_val))
                    else:
                        url_progress_bar.progress(0.20)
                elif "Transcribing" in stage:
                    url_progress_bar.progress(0.20)
                elif "Correcting" in stage:
                    url_progress_bar.progress(0.75)
                elif "Selecting" in stage:
                    url_progress_bar.progress(0.85)

            try:
                with st.spinner("Analyzing video (this may take several minutes)..."):
                    candidates, _no_results = run_url_pipeline(
                        url=url_input.strip(),
                        settings=settings,
                        clip_indices=[],  # Empty = select clips but don't render yet
                        progress_cb=_url_analysis_cb,
                    )

                url_progress_bar.progress(1.0)
                url_stage_text.markdown("**✓ Analysis complete! Review clips below.**")
                st.session_state["url_candidates"] = candidates
                st.session_state["url_is_analyzing"] = False

                # Capture source metadata from the pipeline's probe step
                # (stored in progress callback side-effects; use first candidate as proxy)
                if candidates:
                    st.session_state["url_selected_indices"] = [c.index for c in candidates]

            except (ValueError, RuntimeError) as exc:
                url_progress_bar.progress(0.0)
                url_stage_text.empty()
                st.error(f"**Analysis failed:** {exc}")
                st.session_state["url_is_analyzing"] = False

            st.rerun()

        # ── Display Analysis Results & Candidate Table ─────────────────────────
        candidates: list[ClipCandidate] = st.session_state.get("url_candidates", [])
        if candidates:
            st.markdown("---")
            st.markdown(f"### 🎯 {len(candidates)} Clip Candidates Found")

            selected_indices = _render_url_candidate_table(candidates)

            # Update session state with current checkbox state
            st.session_state["url_selected_indices"] = selected_indices

            st.markdown("")
            n_selected = len(selected_indices)

            # ── Stage 2: Render Button ─────────────────────────────────────────
            col_render, _ = st.columns([3, 1])
            with col_render:
                render_clicked = st.button(
                    f"🚀 Generate {n_selected} Short{'s' if n_selected != 1 else ''}",
                    type="primary",
                    disabled=(
                        n_selected == 0
                        or st.session_state["url_is_rendering"]
                        or not url_input.strip()
                    ),
                    use_container_width=True,
                    key="render_button",
                )

            if n_selected == 0:
                st.warning("Select at least one clip to render.")

            # ── Render Execution ───────────────────────────────────────────────
            if render_clicked and n_selected > 0 and url_input.strip():
                st.session_state["url_is_rendering"] = True
                st.session_state["url_results"] = []

                st.markdown("### ⏳ Rendering clips...")
                render_progress = st.progress(0.0)
                render_stage_text = st.empty()

                def _render_cb(current: int, total: int, stage: str) -> None:
                    fraction = (current + 0.5) / max(total, 1)
                    render_progress.progress(min(fraction, 0.99))
                    render_stage_text.markdown(f"**{stage}**")

                try:
                    with st.spinner("Rendering selected clips..."):
                        _all_candidates, render_results = run_url_pipeline(
                            url=url_input.strip(),
                            settings=settings,
                            clip_indices=selected_indices,
                            progress_cb=_render_cb,
                        )

                    render_progress.progress(1.0)
                    render_stage_text.markdown("**✓ All clips rendered!**")
                    st.session_state["url_results"] = render_results
                    st.session_state["url_is_rendering"] = False

                except (ValueError, RuntimeError) as exc:
                    render_progress.progress(0.0)
                    render_stage_text.empty()
                    st.error(f"**Rendering failed:** {exc}")
                    st.session_state["url_is_rendering"] = False

                st.rerun()

        elif not url_input.strip() and not st.session_state.get("url_is_analyzing"):
            st.markdown("""
            <div style="text-align:center;padding:4rem 0;opacity:0.4;">
                <div style="font-size:5rem;">🔗</div>
                <p style="font-size:1.1rem;margin-top:1rem;">
                    Paste a video URL and click Analyze to get started
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── URL Pipeline Results ───────────────────────────────────────────────
        url_results: list[ProcessingResult] = st.session_state.get("url_results", [])
        if url_results:
            st.markdown("---")
            _render_url_results(url_results, candidates)


if __name__ == "__main__":
    main()

