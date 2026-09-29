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

import logging
import os
import sys
import tempfile
from pathlib import Path

# Ensure sys.path includes both shorts_engine and its parent directory
# so that both direct (`import config`) and package (`import shorts_engine.config`) imports resolve.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_APP_DIR.parent))

import streamlit as st
from config import (
    DEFAULT_OUTPUT_DIR,
    Settings,
    assert_system_binaries,
    inject_ffmpeg_path,
)
from services.niche_templates import get_template, template_options

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
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Minimal neutral dark background */
    .stApp {
        background: #09090b;
        color: #f4f4f5;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #111113;
        border-right: 1px solid #27272a;
    }

    /* Input labels */
    .stTextInput label, .stSelectbox label, .stFileUploader label,
    .stSlider label, .stCheckbox label {
        color: #a1a1aa !important;
        font-weight: 500;
        font-size: 0.8rem;
        letter-spacing: 0.01em;
    }

    /* Text inputs and selects */
    .stTextInput input, .stSelectbox select {
        background: #18181b !important;
        border: 1px solid #27272a !important;
        border-radius: 6px !important;
        color: #f4f4f5 !important;
    }
    .stTextInput input:focus, .stSelectbox select:focus {
        border-color: #52525b !important;
        box-shadow: none !important;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: #f4f4f5;
        border: 1px solid #f4f4f5;
        border-radius: 6px;
        color: #09090b;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.5rem 1.25rem;
        transition: opacity 0.15s ease, background 0.15s ease;
        box-shadow: none;
    }
    .stButton > button[kind="primary"]:hover {
        background: #e4e4e7;
        border-color: #e4e4e7;
        color: #09090b;
        box-shadow: none;
        transform: none;
    }

    /* Secondary / outline buttons */
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]) {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 6px;
        color: #e4e4e7;
        font-weight: 500;
        font-size: 0.85rem;
        box-shadow: none;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover {
        background: #27272a;
        border-color: #3f3f46;
        color: #ffffff;
        box-shadow: none;
        transform: none;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #27272a;
        gap: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #71717a;
        font-weight: 500;
        font-size: 0.88rem;
        padding-bottom: 0.6rem;
    }
    .stTabs [aria-selected="true"] {
        color: #f4f4f5 !important;
        border-bottom-color: #f4f4f5 !important;
    }

    /* Constrain st.video() player to true 9:16 portrait dimensions */
    [data-testid="stVideo"] {
        max-width: 280px !important;
        width: 100% !important;
        aspect-ratio: 9 / 16 !important;
        margin: 0 auto !important;
    }
    [data-testid="stVideo"] video {
        width: 100% !important;
        height: 100% !important;
        aspect-ratio: 9 / 16 !important;
        max-height: 498px !important;
        border-radius: 8px !important;
        border: 1px solid #27272a !important;
        overflow: hidden !important;
        object-fit: contain !important;
        background: #09090b !important;
        display: block !important;
    }

    /* Container cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #27272a !important;
        border-radius: 6px !important;
        background: #121214 !important;
        margin-bottom: 1rem !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: #121214 !important;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 500;
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
    }
    .badge-success { background: #052e16; color: #4ade80; border: 1px solid #14532d; }
    .badge-error   { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }
    .badge-warn    { background: #451a03; color: #fbbf24; border: 1px solid #78350f; }
    .badge-tag     { background: #18181b; color: #d4d4d8; border: 1px solid #27272a; }

    /* Metric value */
    .metric-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #f4f4f5;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #71717a;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Code blocks */
    pre, code {
        background: #141416 !important;
        border: 1px solid #27272a !important;
        border-radius: 4px !important;
        color: #d4d4d8 !important;
        font-size: 0.8rem !important;
    }

    /* Divider */
    hr {
        border-color: #27272a !important;
    }

    /* Progress bar */
    [data-testid="stProgress"] {
        margin: 0.5rem 0 1rem 0;
    }
    [data-testid="stProgressBarTrack"],
    [data-testid="stProgress"] [role="progressbar"] {
        background-color: #18181b !important;
        border: 1px solid #27272a !important;
        border-radius: 4px !important;
        overflow: hidden !important;
    }
    [data-testid="stProgressBarTrack"] > div,
    [data-testid="stProgress"] [role="progressbar"] > div > div {
        background-color: #22c55e !important;
        background: #22c55e !important;
    }
    [data-testid="stProgress"] p,
    [data-testid="stProgress"] span {
        color: #a1a1aa !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }

    /* Alerts */
    .stAlert {
        border-radius: 6px !important;
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

    The sidebar is driven by a Niche Template selectbox at the top. Selecting
    a template pre-fills all dependent controls with sensible defaults for that
    content type; the user can still override any individual setting.

    Returns:
        Settings object built from sidebar inputs.
    """
    with st.sidebar:
        st.markdown("## Settings")

        # Hardware Acceleration Status
        try:
            from services.hw_encoder import get_encoder_name, get_optimal_threads
            active_encoder = get_encoder_name()
            opt_threads = get_optimal_threads()
            encoder_labels = {
                "h264_qsv": "⚡ Intel Quick Sync (QSV)",
                "h264_amf": "⚡ AMD AMF (Radeon)",
                "h264_vaapi": "⚡ Linux VA-API",
                "libx264": f"💻 CPU ({opt_threads} threads, thermal-capped)",
            }
            st.caption(f"**Hardware:** {encoder_labels.get(active_encoder, active_encoder)}")
        except Exception:
            pass

        st.markdown("---")

        # ── API Key Status (read-only) ─────────────────────────────────────
        pexels_loaded = bool(os.environ.get("PEXELS_API_KEY", "").strip())
        gemini_loaded = bool(os.environ.get("GEMINI_API_KEY", "").strip())
        openrouter_loaded = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())

        st.markdown("### API Keys")
        pexels_status = '<span style="color:#22c55e;font-size:0.8rem;">Connected</span>' if pexels_loaded else '<span style="color:#ef4444;font-size:0.8rem;">Missing</span>'
        gemini_status = '<span style="color:#22c55e;font-size:0.8rem;">Connected</span>' if gemini_loaded else '<span style="color:#ef4444;font-size:0.8rem;">Missing</span>'
        or_status = '<span style="color:#22c55e;font-size:0.8rem;">Connected</span>' if openrouter_loaded else '<span style="color:#ef4444;font-size:0.8rem;">Missing</span>'
        st.markdown(
            f"**Pexels** — {pexels_status}  \n"
            f"**Gemini** — {gemini_status}  \n"
            f"**OpenRouter (AI Analyst)** — {or_status}",
            unsafe_allow_html=True,
        )
        if not pexels_loaded or not gemini_loaded or not openrouter_loaded:
            st.caption("Add missing keys to `shorts_engine/.env` and restart.")

        st.markdown("---")

        # ── Niche Template ─────────────────────────────────────────────────
        st.markdown("### 🎯 Niche Template")

        options = template_options()   # list of (name, label)
        option_names = [n for n, _ in options]

        selected_template_name = st.selectbox(
            "Content Niche",
            options=option_names,
            format_func=lambda n: dict(options).get(n, n),
            index=0,
            help=(
                "Select your channel's content niche to auto-configure clip duration, "
                "music, VFX grade, and subtitle placement. You can still override any "
                "individual setting below."
            ),
            key="niche_template_select",
        )

        active_template = get_template(selected_template_name)
        if selected_template_name != "custom":
            st.caption(f"*{active_template.description}*")

        custom_brand_voice = st.text_input(
            "Brand Voice Override (Optional)",
            value="",
            placeholder="e.g. Sarcastic, funny gamer...",
            help="Append a custom voice to the template's preset. Leave blank to use the template's voice.",
            key="brand_voice_input",
        )

        # Compose final brand voice
        if active_template.brand_voice and custom_brand_voice.strip():
            final_brand_voice = f"{active_template.brand_voice} ADDITIONALLY: {custom_brand_voice.strip()}"
        elif custom_brand_voice.strip():
            final_brand_voice = custom_brand_voice.strip()
        else:
            final_brand_voice = active_template.brand_voice

        st.markdown("---")
        st.markdown("### Transcription")
        model_size = st.selectbox(
            "Whisper Model Size",
            options=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
            index=4,  # default: large-v3
            help=(
                "faster-whisper-large-v3 delivers highest Greek accuracy and millisecond timestamps.\n"
                "large-v3-turbo offers near-identical accuracy at 3x higher speed.\n"
                "Use small or base for fast previews on CPU."
            ),
            key="whisper_model_size_select",
        )
        whisper_compute_type = st.selectbox(
            "Whisper Precision / Compute Type",
            options=["int8_float32", "int8", "float32"],
            index=0,
            format_func=lambda x: {
                "int8_float32": "⚡ int8_float32 (Balanced: High accuracy & fast)",
                "int8": "🚀 int8 (Fastest, lowest memory)",
                "float32": "🎯 float32 (Maximum precision, unquantized)",
            }.get(x, x),
            help=(
                "int8_float32 uses 8-bit quantized weights with 32-bit accumulators to avoid "
                "Greek diacritic and accent misclassifications without CPU thermal throttling."
            ),
            key="whisper_compute_type_select",
        )
        whisper_beam_size = st.select_slider(
            "Whisper Beam Size",
            options=[1, 2, 5],
            value=5,
            help=(
                "1 = greedy search (fastest).\n"
                "5 = full beam search (highest accuracy for Greek, slower on CPU)."
            ),
            key="whisper_beam_size_select",
        )
        # Domain context follows the template by default but can be overridden
        domain_options = ["", "politics", "society", "science", "technology", "entertainment", "business", "education", "lifestyle", "gaming"]
        template_hint = active_template.whisper_context_hint
        default_domain_idx = domain_options.index(template_hint) if template_hint in domain_options else 0
        whisper_context_hint = st.selectbox(
            "Domain Context",
            options=domain_options,
            index=default_domain_idx,
            format_func=lambda x: {
                "": "Generic (auto-detect)",
                "politics": "🏛️ Politics / Economy",
                "society": "👥 Society",
                "science": "🔬 Science",
                "technology": "💻 Technology",
                "entertainment": "🎭 Entertainment / Pop Culture",
                "business": "💼 Business / Finance",
                "education": "📚 Education / History",
                "lifestyle": "🌟 Lifestyle / Vlog",
                "gaming": "🎮 Gaming",
            }.get(x, x),
            help=(
                "Inject domain vocabulary into the Whisper transcription prompt and Gemini SEO generator for higher accuracy.\n"
                "Automatically set by the Niche Template — override here if needed."
            ),
            key="whisper_context_hint_select",
        )

        st.markdown("---")
        st.markdown("### Subtitles")
        subtitle_position = st.selectbox(
            "Caption Position",
            options=["lower_third", "center", "top"],
            index=["lower_third", "center", "top"].index(active_template.subtitle_position),
            format_func=lambda x: {
                "lower_third": "Lower Third (default for Shorts)",
                "center": "Center (lifestyle / entertainment)",
                "top": "Top (gaming, avoid platform UI overlap)",
            }.get(x, x),
            help="Vertical position of animated captions in the 9:16 frame.",
            key="subtitle_position_select",
        )

        st.markdown("---")
        st.markdown("### Auto-Framing")
        enable_face_tracking = st.checkbox(
            "YOLO Face Speaker Tracking",
            value=True,
            help="Tracks active speaker face keypoints frame-by-frame to keep the subject centered when cropping 16:9 to 9:16 vertical Shorts.",
            key="enable_face_tracking_check",
        )

        st.markdown("---")
        st.markdown("### Clip Selection")
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
            max_value=50,
            value=int(active_template.clip_min_duration),
            step=5,
            key="clip_min_dur_slider",
        )
        clip_max_dur = st.slider(
            "Max Clip Duration (s)",
            min_value=30,
            max_value=60,
            value=int(active_template.clip_max_duration),
            step=5,
            key="clip_max_dur_slider",
        )
        max_source_duration_minutes = st.slider(
            "Max Source Video Length (min)",
            min_value=5,
            max_value=240,
            value=120,
            step=5,
            help="Reject source videos longer than this. Prevents accidentally processing 4-hour livestreams.",
            key="max_source_dur_slider",
        )

        st.markdown("---")
        st.markdown("### Outro Bumper")
        outro_file = st.file_uploader(
            "Upload Outro (optional)",
            type=["mp4", "mov", "mkv", "avi"],
            help="If provided, this clip is appended to every processed Short.",
            key="outro_uploader",
        )

        outro_path: Path | None = None
        if outro_file is not None:
            if "outro_tmp_path" not in st.session_state:
                suffix = Path(outro_file.name).suffix
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="outro_")
                tmp.write(outro_file.read())
                tmp.flush()
                tmp.close()
                st.session_state["outro_tmp_path"] = tmp.name
            outro_path = Path(st.session_state["outro_tmp_path"])
            st.success(f"Outro loaded: {outro_file.name}")
        else:
            if "outro_tmp_path" in st.session_state:
                del st.session_state["outro_tmp_path"]

        st.markdown("---")
        st.markdown("### 📁 Output Destination")
        default_output_str = str(st.session_state.get("custom_output_dir", DEFAULT_OUTPUT_DIR))
        output_dir_input = st.text_input(
            "Destination Directory",
            value=default_output_str,
            help="Folder where processed shorts, cut clips, and SEO JSON files are saved.",
            key="output_dir_input",
        )
        resolved_output_dir = (
            Path(output_dir_input).expanduser().resolve()
            if output_dir_input.strip()
            else DEFAULT_OUTPUT_DIR
        )
        st.session_state["custom_output_dir"] = str(resolved_output_dir)

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.72rem;color:#555;text-align:center'>"
            "Greek Shorts Engine<br>"
            "Keys loaded from <code>.env</code> — never exposed in UI."
            "</div>",
            unsafe_allow_html=True,
        )

    return Settings(
        whisper_model_size=str(model_size),
        whisper_compute_type=str(whisper_compute_type),
        whisper_beam_size=int(whisper_beam_size),
        whisper_context_hint=str(whisper_context_hint),

        enable_face_tracking=bool(enable_face_tracking),
        enable_vfx=False,
        broll_start_offset=3.0,
        broll_overlay_duration=5.0,
        broll_ken_burns=False,
        broll_split_screen=False,
        transition_type="fade",
        transition_duration=0.35,
        enable_bg_music=True,
        bg_music_track=active_template.bg_music_track,
        bg_music_path=None,
        bg_music_volume=active_template.bg_music_volume,
        bg_music_ducking=active_template.bg_music_ducking,
        min_clips=3,
        max_clips=int(max_clips),
        clip_min_duration=float(clip_min_dur),
        clip_max_duration=float(max(clip_max_dur, clip_min_dur + 5)),
        max_source_duration_seconds=int(max_source_duration_minutes * 60),
        brand_voice=final_brand_voice,
        tts_engine="edge",
        tts_voice="el-GR-NestorasNeural",
        tts_pitch="-4Hz",
        tts_speed=1.0,
        niche_template=selected_template_name,
        subtitle_position=str(subtitle_position),
        subtitle_mode="dynamic",
        enable_dynamic_zoom=False,
        outro_path=outro_path,
        output_dir=resolved_output_dir,
    )


def main() -> None:
    """Entry point for the Streamlit application."""
    _init_session_state()

    # Inject bundled FFmpeg into PATH before any processing
    inject_ffmpeg_path()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    settings = _render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:1.5rem 0 1rem;">
        <h1 style="font-size:1.75rem;font-weight:600;margin:0;color:#f4f4f5;letter-spacing:-0.02em;">
            Greek Shorts Engine
        </h1>
        <p style="color:#71717a;margin-top:0.35rem;font-size:0.9rem;">
            Batch convert Greek clips into 9:16 vertical Shorts with subtitles, B-roll, and SEO metadata.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── System Check ───────────────────────────────────────────────────────────
    try:
        assert_system_binaries()
    except RuntimeError as exc:
        st.error(f"**System requirement not met:**\n\n{exc}")
        st.stop()

    # ── Autopilot Studio ───────────────────────────────────────────────────────
    try:
        from ui.autopilot_tab import render_autopilot_tab
    except ImportError:
        from shorts_engine.ui.autopilot_tab import render_autopilot_tab

    render_autopilot_tab(settings)


if __name__ == "__main__":
    main()

