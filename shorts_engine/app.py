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
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from config import Settings, assert_system_binaries, inject_ffmpeg_path
from pipeline import ProcessingResult, run_batch, run_url_pipeline
from services.clip_selector import ClipCandidate
from services.seo_generator import get_download_filename

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

    API keys are read exclusively from the .env file / environment — no
    sidebar inputs are exposed for them.  A status row shows whether each
    key was found.

    Returns:
        Settings object built from sidebar inputs.
    """
    with st.sidebar:
        st.markdown("## Settings")
        st.markdown("---")

        # ── API Key Status (read-only) ─────────────────────────────────────
        pexels_loaded = bool(os.environ.get("PEXELS_API_KEY", "").strip())
        gemini_loaded = bool(os.environ.get("GEMINI_API_KEY", "").strip())

        st.markdown("### API Keys")
        pexels_status = '<span style="color:#22c55e;font-size:0.8rem;">Connected</span>' if pexels_loaded else '<span style="color:#ef4444;font-size:0.8rem;">Missing</span>'
        gemini_status = '<span style="color:#22c55e;font-size:0.8rem;">Connected</span>' if gemini_loaded else '<span style="color:#ef4444;font-size:0.8rem;">Missing</span>'
        st.markdown(
            f"**Pexels** — {pexels_status}  \n"
            f"**Gemini** — {gemini_status}",
            unsafe_allow_html=True,
        )
        if not pexels_loaded or not gemini_loaded:
            st.caption("Add missing keys to `shorts_engine/.env` and restart.")

        st.markdown("---")
        st.markdown("### Transcription")
        model_size = st.selectbox(
            "Whisper Model Size",
            options=["tiny", "base", "small", "medium", "large-v3"],
            index=4,  # default: large-v3
            help=(
                "faster-whisper-large-v3 delivers highest Greek accuracy and millisecond timestamps.\n"
                "Use small or base for fast 1–2 minute previews on CPU for long YouTube videos."
            ),
            key="whisper_model_size_select",
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
        whisper_context_hint = st.selectbox(
            "Domain Context",
            options=["", "politics", "society", "science", "technology", "entertainment", "business", "education", "lifestyle", "gaming"],
            index=0,
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
                "Select the topic closest to your video's content."
            ),
            key="whisper_context_hint_select",
        )

        st.markdown("---")
        st.markdown("### Content Analysis & Hooks")
        st.markdown("---")
        st.markdown("### Auto-Framing")
        enable_face_tracking = st.checkbox(
            "YOLO Face Speaker Tracking",
            value=True,
            help="Tracks active speaker coordinates frame-by-frame with EMA smoothing to keep the subject centered when cropping 16:9 to 9:16 vertical Shorts.",
            key="enable_face_tracking_check",
        )

        st.markdown("---")
        st.markdown("### VFX & Colour Grading")
        enable_vfx = st.checkbox(
            "Enable Auto VFX / Colour Grade",
            value=True,
            help=(
                "Analyses the transcript and YOLO scene detections to automatically "
                "select and apply one of five colour grades:\n"
                "• VIBRANCE — high-energy/hype moments\n"
                "• DRAMATIC — suspense/tension keywords\n"
                "• WARMTH — person-dominant clips\n"
                "• CINEMATIC — action/object scenes without people\n"
                "• SUBTLE — mild universal lift (fallback)"
            ),
            key="enable_vfx_check",
        )
        if enable_vfx:
            st.caption(
                "Preset selected automatically per clip from transcript keywords "
                "and YOLO scene analysis. Applied after subtitle burn-in, before "
                "background music."
            )

        st.markdown("---")
        st.markdown("### B-Roll Overlay")
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
        
        broll_ken_burns = st.checkbox(
            "Enable Ken Burns Effect",
            value=True,
            help="Apply a slow, continuous zoom-in to B-roll to keep the viewer engaged.",
            key="broll_ken_burns_check",
        )
        broll_split_screen = st.checkbox(
            "Split-Screen B-Roll",
            value=False,
            help="Show the B-roll in the top half and the speaker in the bottom half.",
            key="broll_split_screen_check",
        )

        st.markdown("---")
        st.markdown("### Transitions")
        transition_style = st.selectbox(
            "Transition Style",
            options=["fade", "flash", "none"],
            format_func=lambda x: {
                "fade": "Soft Fade (Crossfade)",
                "flash": "White Flash",
                "none": "Cut (None)",
            }.get(x, x),
            index=0,
            help="Soft transition applied between the main clip and B-roll, and into the Outro.",
            key="transition_style_select",
        )
        transition_duration = st.slider(
            "Transition Duration (s)",
            min_value=0.15,
            max_value=0.80,
            value=0.35,
            step=0.05,
            help="Duration of the fade or flash transition.",
            key="transition_duration_slider",
        )

        st.markdown("---")
        st.markdown("### SEO Settings")
        
        seo_preset = st.selectbox(
            "Virality Preset",
            options=["None", "Politics/Economy", "Society", "Science", "Technology"],
            index=0,
            help="Select a high-virality persona preset tailored to your content's niche.",
            key="seo_preset_select",
        )
        
        custom_brand_voice = st.text_input(
            "Custom Brand Voice / Persona (Optional)",
            value="",
            placeholder="e.g. Sarcastic, funny gamer...",
            help="Inject your specific channel personality. This will be combined with the preset if selected.",
            key="brand_voice_input",
        )
        
        # Combine preset and custom voice
        preset_mapping = {
            "None": "",
            "Politics/Economy": "Highly authoritative, analytical, and slightly polarizing. Focus on hidden agendas, economic impact, and hard truths. Tone should be serious, urgent, and provocative.",
            "Society": "Relatable, empathetic, and thought-provoking. Focus on human behavior, social dynamics, and everyday realities. Tone should spark intense debate and personal reflection.",
            "Science": "Educational, mind-blowing, and highly factual. Focus on explaining complex concepts simply, debunking myths, and highlighting future implications. Tone should be awe-inspiring and authoritative.",
            "Technology": "Forward-looking, fast-paced, and analytical. Focus on innovation, disruption, and how tech changes daily life. Tone should be cutting-edge, enthusiastic, and slightly urgent."
        }
        
        selected_preset = preset_mapping.get(seo_preset, "")
        if selected_preset and custom_brand_voice.strip():
            final_brand_voice = f"{selected_preset} ADDITIONALLY: {custom_brand_voice.strip()}"
        elif selected_preset:
            final_brand_voice = selected_preset
        else:
            final_brand_voice = custom_brand_voice.strip()

        st.markdown("---")
        st.markdown("### Clip Selection (URL Mode)")
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
        st.markdown("### Outro Bumper")
        outro_file = st.file_uploader(
            "Upload Outro (optional)",
            type=["mp4", "mov", "mkv", "avi"],
            help="If provided, this clip is appended to every processed Short.",
            key="outro_uploader",
        )

        outro_path: Path | None = None
        if outro_file is not None:
            # Persist the uploaded outro to a session-scoped temp file
            if "outro_tmp_path" not in st.session_state:
                suffix = Path(outro_file.name).suffix
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix, prefix="outro_"
                )
                tmp.write(outro_file.read())
                tmp.flush()
                tmp.close()
                st.session_state["outro_tmp_path"] = tmp.name
            outro_path = Path(st.session_state["outro_tmp_path"])
            st.success(f"Outro loaded: {outro_file.name}")
        else:
            # Clear stale temp path if user removed the file
            if "outro_tmp_path" in st.session_state:
                del st.session_state["outro_tmp_path"]

        st.markdown("---")
        st.markdown("### Background Music")
        enable_bg_music = st.checkbox(
            "Enable Background Music",
            value=True,
            help="Layers subtle ambient background music under speech with automatic ducking.",
            key="enable_bg_music_check",
        )

        bg_music_track = "ambient_calm"
        custom_music_path: Path | None = None
        bg_music_vol = 0.20
        bg_music_duck = True

        if enable_bg_music:
            bg_music_track = st.selectbox(
                "Sound Bed Preset",
                options=["ambient_calm", "dramatic_pulse", "upbeat_groove", "custom", "none"],
                format_func=lambda x: {
                    "ambient_calm": "Ambient Calm (Warm Acoustic Pad)",
                    "dramatic_pulse": "Dramatic Pulse (Tension Drone)",
                    "upbeat_groove": "Upbeat Groove (Modern Light)",
                    "custom": "Upload Custom Track",
                    "none": "None",
                }.get(x, x),
                index=0,
                help="Select a bundled royalty-free sound bed or upload your own audio.",
                key="bg_music_track_select",
            )

            if bg_music_track == "custom":
                custom_music_file = st.file_uploader(
                    "Upload Music Track (.mp3, .wav, .m4a)",
                    type=["mp3", "wav", "m4a", "aac"],
                    help="Custom audio file to use as background music.",
                    key="custom_music_uploader",
                )
                if custom_music_file is not None:
                    if "custom_music_tmp_path" not in st.session_state:
                        suffix = Path(custom_music_file.name).suffix
                        tmp_music = tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix, prefix="bgm_"
                        )
                        tmp_music.write(custom_music_file.read())
                        tmp_music.flush()
                        tmp_music.close()
                        st.session_state["custom_music_tmp_path"] = tmp_music.name
                    custom_music_path = Path(st.session_state["custom_music_tmp_path"])
                    st.success(f"Track loaded: {custom_music_file.name}")
                else:
                    if "custom_music_tmp_path" in st.session_state:
                        del st.session_state["custom_music_tmp_path"]

            bg_music_vol = st.slider(
                "Music Volume",
                min_value=0.02,
                max_value=0.40,
                value=0.20,
                step=0.01,
                format="%.2f",
                help="Volume of background music relative to speech (20% recommended).",
                key="bg_music_vol_slider",
            )

            bg_music_duck = st.checkbox(
                "Speech Ducking",
                value=True,
                help="Automatically lowers background music when the speaker is talking so words remain 100% intelligible.",
                key="bg_music_ducking_check",
            )

        st.markdown("---")
        st.markdown("### 📁 Output Destination")
        from config import DEFAULT_OUTPUT_DIR
        default_output_str = str(st.session_state.get("custom_output_dir", DEFAULT_OUTPUT_DIR))
        output_dir_input = st.text_input(
            "Destination Directory",
            value=default_output_str,
            help="Folder on your machine where processed shorts, cut clips, and SEO JSON files are saved automatically.",
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
            "Greek Shorts Engine · CPU-only<br>"
            "Keys loaded from <code>.env</code> — never exposed in UI."
            "</div>",
            unsafe_allow_html=True,
        )

    return Settings(
        whisper_model_size=str(model_size),
        whisper_beam_size=int(whisper_beam_size),
        whisper_context_hint=str(whisper_context_hint),

        enable_face_tracking=bool(enable_face_tracking),
        enable_vfx=bool(enable_vfx),
        broll_start_offset=float(broll_start),
        broll_overlay_duration=float(broll_duration),
        broll_ken_burns=bool(broll_ken_burns),
        broll_split_screen=bool(broll_split_screen),
        transition_type=str(transition_style),
        transition_duration=float(transition_duration),
        enable_bg_music=bool(enable_bg_music),
        bg_music_track=str(bg_music_track),
        bg_music_path=custom_music_path,
        bg_music_volume=float(bg_music_vol),
        bg_music_ducking=bool(bg_music_duck),
        min_clips=3,
        max_clips=int(max_clips),
        clip_min_duration=float(clip_min_dur),
        clip_max_duration=float(max(clip_max_dur, clip_min_dur + 5)),
        brand_voice=final_brand_voice,
        outro_path=outro_path,
        output_dir=resolved_output_dir,
    )


# ── Result Card Renderer ───────────────────────────────────────────────────────

def _get_download_filename(result: ProcessingResult, extension: str = "mp4") -> str:
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


def _render_result_card(result: ProcessingResult, index: int) -> None:
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
                    dl_filename = _get_download_filename(result, extension="mp4")
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
                        default_dest = st.session_state.get("custom_output_dir", str(Path.home() / "Downloads"))
                        dest_folder_val = st.text_input(
                            "Destination Folder",
                            value=default_dest,
                            key=f"dest_folder_val_{index}",
                            help="Local directory path to save a copy of this short.",
                        )
                        if st.button("💾 Save Copy to Folder", key=f"btn_save_dest_{index}", use_container_width=True):
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
                        seo_dl_filename = _get_download_filename(result, extension="json")
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


# ── Progress Callback ──────────────────────────────────────────────────────────


def _make_progress_callback(
    progress_bar: st.delta_generator.DeltaGenerator,
    stage_text: st.delta_generator.DeltaGenerator,
    total: int,
) -> pipeline.ProgressCallback:
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
        "<div style='font-size:0.85rem;color:#71717a;margin-bottom:0.8rem;'>"
        "Review the AI-selected clips below. Uncheck any you do not want to render."
        "</div>",
        unsafe_allow_html=True,
    )

    # Column headers
    hdr_cols = st.columns([0.5, 0.5, 1.5, 2.5, 3])
    headers = ["", "#", "Duration", "Time Range", "AI SEO Title"]
    for col, hdr in zip(hdr_cols, headers):
        col.markdown(f"<span style='font-size:0.75rem;color:#71717a;font-weight:600;"
                     f"text-transform:uppercase;letter-spacing:0.05em'>{hdr}</span>",
                     unsafe_allow_html=True)

    st.markdown("<hr style='margin:0.4rem 0;border-color:#27272a'>",
                unsafe_allow_html=True)

    for clip in candidates:
        row_cols = st.columns([0.5, 0.5, 1.5, 2.5, 3])

        with row_cols[0]:
            checked = st.checkbox(
                label=f"Select Clip {clip.index}",
                value=True,
                key=f"clip_check_{clip.index}",
                label_visibility="collapsed",
            )

        with row_cols[1]:
            st.markdown(
                f"<span style='font-size:0.88rem;font-weight:600;color:#d4d4d8'>"
                f"#{clip.index}</span>",
                unsafe_allow_html=True,
            )

        with row_cols[2]:
            st.markdown(
                f"<span style='font-size:0.88rem;color:#f4f4f5'>"
                f"{clip.duration_display}</span>",
                unsafe_allow_html=True,
            )

        with row_cols[3]:
            st.markdown(
                f"<span style='font-size:0.8rem;color:#71717a'>"
                f"{clip.start_display} → {clip.end_display}</span>",
                unsafe_allow_html=True,
            )

        with row_cols[4]:
            # Truncate title for display
            display_title = clip.seo.title[:55] + "…" if len(clip.seo.title) > 55 else clip.seo.title
            st.markdown(
                f"<span style='font-size:0.88rem;color:#f4f4f5'>"
                f"<strong>{display_title}</strong></span>",
                unsafe_allow_html=True,
            )

        # Hook, SEO & tags in an expander below the row
        with st.expander(f"Hook, SEO & Tags — Clip #{clip.index}", expanded=False):
            st.markdown(f"**Hook:** {clip.hook_summary}")
            st.markdown(f"**B-Roll query:** `{clip.broll_query}`")
            st.markdown(f"**Primary Keyword:** `{clip.seo.primary_keyword}`")
            st.markdown(f"**AI Title:** {clip.seo.title}")
            st.markdown(f"**Curiosity Angle:** *{clip.seo.curiosity_title}*")
            st.markdown(f"**Authority Angle:** *{clip.seo.authority_title}*")
            st.markdown(f"**Contrarian Angle:** *{clip.seo.contrarian_title}*")
            st.markdown(f"**Description:** {clip.seo.description}")
            if clip.seo.pinned_comment:
                st.markdown("**Pinned Comment:**")
                st.info(clip.seo.pinned_comment)
            st.markdown("**YouTube Tags (Copy & Paste directly into YouTube Studio Tags box):**")
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

    st.markdown("### Shorts Generated")
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.markdown(
            f'<div class="metric-value" style="color:#22c55e">{n_success}</div>'
            f'<div class="metric-label">Succeeded</div>',
            unsafe_allow_html=True,
        )
    with summary_cols[1]:
        color = "#ef4444" if n_fail > 0 else "#71717a"
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

    if results:
        completion_ratio = n_success / max(len(results), 1)
        st.progress(
            value=completion_ratio,
            text=f"Batch Completion: {n_success}/{len(results)} clips generated ({int(completion_ratio * 100)}%)",
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
                f'<div style="margin:-0.4rem 0 0.8rem 0;padding:0.4rem 0.6rem;'
                f'background:#141416;border-radius:4px;'
                f'font-size:0.78rem;color:#71717a;">'
                f'<em>{matched_clip.hook_summary}</em>'
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

    # ── Tabs ───────────────────────────────────────────────────────────────────
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "File Upload"

    active_tab = st.segmented_control(
        "Navigation",
        ["File Upload", "Video URL", "📊 Niche Explorer", "📤 Upload to YouTube"],
        key="active_tab",
        label_visibility="collapsed"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — FILE UPLOAD (existing behaviour, untouched)
    # ══════════════════════════════════════════════════════════════════════════
    if active_tab == "File Upload":
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
                or st.session_state["is_processing"]
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

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — VIDEO URL
    # ══════════════════════════════════════════════════════════════════════════
    elif active_tab == "Video URL":
        _, main_col, _ = st.columns([1, 6, 1])
        with main_col:
            st.markdown("### Process from URL")
            st.markdown(
                "Paste a YouTube, TikTok, or Instagram link. The engine will download it, "
                "detect the most energetic speaking segments, and prepare candidates for you."
            )

            if "queued_url" in st.session_state:
                st.session_state["url_input"] = st.session_state.pop("queued_url")

            url_input = st.text_input(
                "Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                key="url_input",
            )

            settings_errors_url = settings.validate()
            if settings_errors_url:
                for err in settings_errors_url:
                    st.warning(err)

            if st.session_state.pop("queued_url_autostart", False) and url_input.strip():
                st.session_state["url_is_analyzing"] = True
                st.rerun()

            col_a, col_b = st.columns([1, 4])
            with col_a:
                analyze_clicked = st.button(
                    "🔍 Analyze Video",
                    key="analyze_url_btn",
                    use_container_width=True,
                    type="primary",
                    disabled=bool(settings_errors_url) or not url_input.strip()
                )

            if analyze_clicked and url_input.strip() and not settings_errors_url:
                st.session_state["url_is_analyzing"] = True

            if st.session_state.get("url_is_analyzing"):
                url_progress_bar = st.progress(0.0)
                url_stage_text = st.empty()

                def _url_analysis_cb(idx: int, total: int, stage: str):
                    # For a single video analysis, map stages to approximate percentages
                    val = 0.0
                    if "Downloading" in stage:
                        import re
                        m = re.search(r'\[(.*?)%\]', stage)
                        if m:
                            try:
                                pct_val = float(m.group(1)) / 100.0
                            except ValueError:
                                pct_val = 0.0
                            val = min(0.70, 0.20 + 0.50 * pct_val)
                        else:
                            val = 0.20
                    elif "Transcribing" in stage:
                        val = 0.20
                    elif "Correcting" in stage:
                        val = 0.75
                    elif "Selecting" in stage:
                        val = 0.85
                    pct = int(val * 100)
                    url_progress_bar.progress(val, text=f"Analyzing video ({pct}%): {stage}")
                    url_stage_text.markdown(f"**{stage}**")

                try:
                    with st.spinner("Analyzing video (this may take several minutes)..."):
                        candidates, _no_results = run_url_pipeline(
                            url=url_input.strip(),
                            settings=settings,
                            clip_indices=[],  # Empty = select clips but don't render yet
                            progress_cb=_url_analysis_cb,
                        )

                    url_progress_bar.progress(1.0, text="100% — Analysis complete.")
                    url_stage_text.markdown("**Analysis complete. Review clips below.**")
                    st.session_state["url_candidates"] = candidates
                    st.session_state["url_is_analyzing"] = False

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
                st.markdown(f"### {len(candidates)} Clip Candidates Found (Minimum: 3)")

                selected_indices = _render_url_candidate_table(candidates)

                st.session_state["url_selected_indices"] = selected_indices

                st.markdown("---")
                st.markdown("### Render Selected")
                
                n_selected = len(selected_indices)
                min_required = settings.min_clips
                
                render_col1, render_col2 = st.columns([1, 4])
                with render_col1:
                    render_clicked = st.button(
                        f"🚀 Render {n_selected} Clips",
                        key="render_url_btn",
                        use_container_width=True,
                        type="primary",
                        disabled=n_selected < min_required or not url_input.strip()
                    )
                with render_col2:
                    if n_selected < min_required:
                        st.warning(f"Please select at least {min_required} clips to proceed.")

                if render_clicked and n_selected >= min_required and url_input.strip():
                    st.session_state["url_is_rendering"] = True
                    st.session_state["url_results"] = []
                    
            if st.session_state.get("url_is_rendering"):
                render_progress = st.progress(0.0)
                render_stage_text = st.empty()

                def _url_render_cb(idx: int, total: int, stage: str):
                    pct = idx / total if total > 0 else 0.0
                    render_progress.progress(pct, text=f"Rendering {idx}/{total}: {stage}")
                    render_stage_text.markdown(f"**{stage}**")

                try:
                    with st.spinner("Rendering final Shorts..."):
                        _all_candidates, render_results = run_url_pipeline(
                            url=url_input.strip(),
                            settings=settings,
                            clip_indices=selected_indices,
                            progress_cb=_url_render_cb,
                        )

                    render_progress.progress(1.0, text="100% — All clips rendered.")
                    render_stage_text.markdown("**All clips rendered.**")
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
                <div style="text-align:center;padding:3rem 0;border:1px dashed #27272a;border-radius:6px;margin-top:1rem;">
                    <p style="font-size:0.88rem;color:#71717a;margin:0;">
                        Paste a video URL above and click Analyze Video to get started
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # ── URL Pipeline Results ───────────────────────────────────────────────
            url_results: list[ProcessingResult] = st.session_state.get("url_results", [])
            if url_results:
                st.markdown("---")
                _render_url_results(url_results, candidates)


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — NICHE EXPLORER
    # ══════════════════════════════════════════════════════════════════════════
    elif active_tab == "📊 Niche Explorer":
        _, main_col, _ = st.columns([1, 6, 1])
        with main_col:
            st.markdown("### 📊 My Channel Analytics")
            
            # --- AUTH CHECK ---
            try:
                from services.youtube_uploader import (
                    is_authenticated, authenticate, get_analytics_client,
                    fetch_channel_analytics, fetch_my_recent_videos,
                    get_channel_info
                )
                yt_connected = is_authenticated()
            except ImportError:
                yt_connected = False
            
            analyze_mine_clicked = False
            
            if yt_connected:
                st.success("✅ **YouTube API Connected:** Pulling deep real-time analytics for your channel.")
                max_videos_mine = st.number_input("Max recent videos to analyze", min_value=5, max_value=100, value=30, step=5, key="channel_max_mine")
                analyze_mine_clicked = st.button("🚀 Analyze My Channel", type="primary", use_container_width=True)
            else:
                st.warning("⚠️ **YouTube Account Not Connected**")
                st.markdown(
                    "Please connect your YouTube channel in the **Upload to YouTube** tab to unlock deep channel analytics and AI insights."
                )

            # Scan & analyze
            if analyze_mine_clicked:
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
                                    analytics_data=st.session_state.get("yt_analytics_30d")
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
                    Click 'Analyze My Channel' above to get started
                </p>
            </div>
            """, unsafe_allow_html=True)


    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4 — UPLOAD TO YOUTUBE
    # ═══════════════════════════════════════════════════════════════════════════
    elif active_tab == "📤 Upload to YouTube":
        _, main_col, _ = st.columns([1, 6, 1])
        with main_col:
            st.markdown("### 📤 Upload to YouTube")
            st.markdown(
                "Connect your YouTube channel to upload finished Shorts directly from the app. "
                "Set an immediate or scheduled publish time."
            )

            # Lazy-import uploader service so the rest of the app works even
            # if google-api-python-client is not yet installed.
            try:
                from services.youtube_uploader import (
                    YouTubeAuthError,
                    YouTubeUploadError,
                    authenticate,
                    get_channel_info,
                    is_authenticated,
                    revoke_token,
                    upload_short,
                )
                _yt_pkgs_ok = True
            except ImportError:
                _yt_pkgs_ok = False
                st.error(
                    "📦 **Missing packages.** Run the following in the terminal, then restart the app:\n\n"
                    "```bash\n"
                    "pip install google-api-python-client google-auth-oauthlib google-auth-httplib2\n"
                    "```"
                )

            if _yt_pkgs_ok:
                # ── Check for client_secrets.json ───────────────────────────────
                from pathlib import Path as _Path
                _secrets_path = _Path(__file__).parent / "client_secrets.json"
                if not _secrets_path.is_file():
                    st.warning(
                        "⚠️ **client_secrets.json not found.** "
                        "To connect your YouTube channel:\n\n"
                        "1. Go to [Google Cloud Console](https://console.cloud.google.com/).\n"
                        "2. Create a project, enable the **YouTube Data API v3**.\n"
                        "3. Create an **OAuth 2.0 Client ID** (type: Desktop App).\n"
                        "4. Download the JSON file and save it as `shorts_engine/client_secrets.json`.\n"
                        "5. Restart the app."
                    )

                st.markdown("---")

                # ── Auth Status ──────────────────────────────────────────────────
                connected = is_authenticated()

                auth_col, rev_col = st.columns([3, 1])
                with auth_col:
                    if connected:
                        st.success("✅ YouTube account connected")
                    else:
                        st.info("🔒 Not connected — click **Connect YouTube Account** to begin.")

                with rev_col:
                    if connected:
                        if st.button("🔓 Disconnect", key="yt_revoke", type="secondary"):
                            revoke_token()
                            st.success("Disconnected. Token deleted.")
                            st.rerun()

                if not connected:
                    if st.button("🔗 Connect YouTube Account", key="yt_connect", type="primary"):
                        try:
                            with st.spinner("Opening browser for Google OAuth2 login..."):
                                yt = authenticate()
                                info = get_channel_info(yt)
                                st.session_state["yt_channel_info"] = info
                            st.success(
                                f"✅ Connected as: **{info['title']}** "
                                f"({info['subscriber_count']:,} subscribers)"
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"❌ Authentication failed: {exc}")

                if connected:
                    # ── Channel Info Card ────────────────────────────────────────
                    try:
                        if "yt_channel_info" not in st.session_state:
                            with st.spinner("Loading channel info..."):
                                yt = authenticate()
                                st.session_state["yt_channel_info"] = get_channel_info(yt)
                        info = st.session_state["yt_channel_info"]

                        st.markdown("#### Your Channel")
                        ch_cols = st.columns(3)
                        with ch_cols[0]:
                            st.metric("📺 Channel", info["title"])
                        with ch_cols[1]:
                            st.metric("👥 Subscribers", f"{info['subscriber_count']:,}")
                        with ch_cols[2]:
                            st.metric("🎥 Videos", f"{info['video_count']:,}")

                    except Exception as exc:
                        st.warning(f"Could not load channel info: {exc}")

                    st.markdown("---")

                    # ── Processed Clips ──────────────────────────────────────────
                    st.markdown("#### Processed Clips Ready to Upload")

                    output_dir = settings.output_dir
                    all_clips: list[Path] = []
                    if output_dir.is_dir():
                        all_clips = sorted(
                            output_dir.rglob("*_short.mp4"),
                            key=lambda p: p.stat().st_mtime,
                            reverse=True,
                        )

                    if not all_clips:
                        st.info(
                            "No processed clips found in the output directory yet. "
                            "Use **File Upload** or **Video URL** tabs to generate Shorts first."
                        )
                    else:
                        st.caption(
                            f"Found {len(all_clips)} clip(s) in `{output_dir}`. "
                            "Select a clip to configure and upload."
                        )

                        for idx, clip_path in enumerate(all_clips):
                            # Attempt to load companion SEO JSON
                            seo_stem = clip_path.stem.replace("_short", "")
                            seo_json_path = clip_path.parent / f"seo_{seo_stem}.json"
                            seo_data: dict = {}
                            if seo_json_path.is_file():
                                try:
                                    import json as _json
                                    seo_data = _json.loads(seo_json_path.read_text(encoding="utf-8"))
                                except Exception:
                                    pass

                            with st.expander(f"🎥 {clip_path.name}", expanded=False):
                                upload_title = st.text_input(
                                    "Title",
                                    value=seo_data.get("title", clip_path.stem)[:100],
                                    max_chars=100,
                                    key=f"yt_title_{idx}_{clip_path.name}",
                                )
                                upload_desc = st.text_area(
                                    "Description",
                                    value=seo_data.get("description", "")[:5000],
                                    height=120,
                                    key=f"yt_desc_{idx}_{clip_path.name}",
                                )
                                raw_tags = seo_data.get("tags", [])
                                upload_tags_str = st.text_input(
                                    "Tags (comma-separated)",
                                    value=", ".join(raw_tags) if raw_tags else "",
                                    key=f"yt_tags_{idx}_{clip_path.name}",
                                    help="Paste YouTube tags separated by commas. Max 500 chars total.",
                                )

                                publish_mode = st.radio(
                                    "Publish Mode",
                                    ["Publish Now", "Schedule"],
                                    horizontal=True,
                                    key=f"yt_mode_{idx}_{clip_path.name}",
                                )

                                scheduled_dt = None
                                if publish_mode == "Schedule":
                                    from datetime import date as _date, time as _time
                                    sched_date = st.date_input(
                                        "Publish Date (UTC)",
                                        value=_date.today(),
                                        key=f"yt_date_{idx}_{clip_path.name}",
                                    )
                                    sched_time = st.time_input(
                                        "Publish Time (UTC)",
                                        value=_time(9, 0),
                                        key=f"yt_time_{idx}_{clip_path.name}",
                                    )
                                    scheduled_dt = datetime(
                                        sched_date.year, sched_date.month, sched_date.day,
                                        sched_time.hour, sched_time.minute,
                                        tzinfo=timezone.utc,
                                    )

                                if st.button(
                                    "🚀 Upload to YouTube" if publish_mode == "Publish Now" else "🗓️ Schedule Upload",
                                    key=f"yt_upload_{idx}_{clip_path.name}",
                                    type="primary",
                                ):
                                    try:
                                        tag_list = [
                                            t.strip() for t in upload_tags_str.split(",")
                                            if t.strip()
                                        ]
                                        with st.spinner(
                                            f"Uploading '{clip_path.name}' to YouTube..."
                                        ):
                                            yt = authenticate()
                                            video_id = upload_short(
                                                youtube_client=yt,
                                                video_path=clip_path,
                                                title=upload_title,
                                                description=upload_desc,
                                                tags=tag_list,
                                                publish_at=scheduled_dt,
                                            )
                                        watch_url = f"https://www.youtube.com/watch?v={video_id}"
                                        if scheduled_dt:
                                            st.success(
                                                f"✅ **Scheduled!** Will go live at "
                                                f"`{scheduled_dt.strftime('%Y-%m-%d %H:%M UTC')}`\n\n"
                                                f"🔗 [View in YouTube Studio]({watch_url})"
                                            )
                                        else:
                                            st.success(
                                                f"✅ **Published!** Your Short is now live:\n\n"
                                                f"🔗 [{watch_url}]({watch_url})"
                                            )
                                    except FileNotFoundError as exc:
                                        st.error(f"❌ File not found: {exc}")
                                    except YouTubeAuthError as exc:
                                        st.error(f"❌ Auth error: {exc}")
                                    except YouTubeUploadError as exc:
                                        st.error(f"❌ Upload failed: {exc}")
                                    except Exception as exc:
                                        st.error(f"❌ Unexpected error: {exc}")


if __name__ == "__main__":
    main()
