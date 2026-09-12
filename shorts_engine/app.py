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
            index=2,  # default: small
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

        enable_face_tracking=bool(enable_face_tracking),
        enable_vfx=bool(enable_vfx),
        broll_start_offset=float(broll_start),
        broll_overlay_duration=float(broll_duration),
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
    tab_upload, tab_url = st.tabs(["File Upload", "Video URL"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — FILE UPLOAD (existing behaviour, untouched)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_upload:
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

            progress_bar.progress(1.0)
            stage_text.markdown("**Batch complete.**")
            st.session_state["results"] = results
            st.session_state["is_processing"] = False

            shutil.rmtree(upload_tmp, ignore_errors=True)

            st.rerun()

        # ── Upload Results ─────────────────────────────────────────────────────
        results: list[ProcessingResult] = st.session_state.get("results", [])
        if results:
            n_success = sum(1 for r in results if r.success)
            n_fail = len(results) - n_success

            st.markdown("### Results")
            cols = st.columns(3)
            with cols[0]:
                st.markdown(
                    f'<div class="metric-value" style="color:#22c55e">{n_success}</div>'
                    f'<div class="metric-label">Succeeded</div>',
                    unsafe_allow_html=True,
                )
            with cols[1]:
                color = "#ef4444" if n_fail > 0 else "#71717a"
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

            completion_ratio = n_success / max(len(results), 1)
            st.progress(
                value=completion_ratio,
                text=f"Batch Completion: {n_success}/{len(results)} clips processed ({int(completion_ratio * 100)}%)",
            )

            st.markdown("")
            for idx, result in enumerate(results):
                _render_result_card(result, idx)

        elif not uploaded_files:
            st.markdown("""
            <div style="text-align:center;padding:3rem 0;border:1px dashed #27272a;border-radius:6px;margin-top:1rem;">
                <p style="font-size:0.88rem;color:#71717a;margin:0;">
                    Upload Greek video clips above to get started
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — VIDEO URL
    # ══════════════════════════════════════════════════════════════════════════
    with tab_url:
        _, url_col, _ = st.columns([1, 6, 1])
        with url_col:

            st.markdown("### Video URL")
            st.markdown(
                "<div style='font-size:0.85rem;color:#71717a;margin-bottom:0.8rem;'>"
                "Supports YouTube, Vimeo, and direct MP4 links. "
                "The engine transcribes the video and detects the most engaging clips."
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
                    st.warning(err)

            # ── Stage 1: Analyze Button ────────────────────────────────────────
            col_analyze, col_url_clear = st.columns([3, 1])
            with col_analyze:
                analyze_clicked = st.button(
                    "Analyze Video",
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
                if st.button("Clear", use_container_width=True, key="url_clear_button"):
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

            st.markdown("### Analyzing Video...")
            url_progress_bar = st.progress(0.0)
            url_stage_text = st.empty()


            def _url_analysis_cb(current: int, total: int, stage: str) -> None:
                val = 0.10
                if "Probing" in stage:
                    val = 0.05
                elif "Downloading" in stage:
                    val = 0.15
                elif "Transcribing audio:" in stage:
                    m = re.search(r"\((\d+)%\)", stage)
                    if m:
                        pct_val = int(m.group(1)) / 100.0
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
            st.markdown(f"### {len(candidates)} Clip Candidates Found (Minimum: 3)")

            selected_indices = _render_url_candidate_table(candidates)

            # Update session state with current checkbox state
            st.session_state["url_selected_indices"] = selected_indices

            st.markdown("")
            n_selected = len(selected_indices)
            min_required = min(3, len(candidates))

            # ── Stage 2: Render Button ─────────────────────────────────────────
            col_render, _ = st.columns([3, 1])
            with col_render:
                render_clicked = st.button(
                    f"Generate {n_selected} Short{'s' if n_selected != 1 else ''}",
                    type="primary",
                    disabled=(
                        n_selected < min_required
                        or st.session_state["url_is_rendering"]
                        or not url_input.strip()
                    ),
                    use_container_width=True,
                    key="render_button",
                )

            if n_selected < min_required:
                st.warning(f"⚠️ A minimum of {min_required} clips must be selected to render (currently selected: {n_selected}).")

            # ── Render Execution ───────────────────────────────────────────────
            if render_clicked and n_selected >= min_required and url_input.strip():
                st.session_state["url_is_rendering"] = True
                st.session_state["url_results"] = []

                st.markdown("### Rendering Clips...")
                render_progress = st.progress(0.0)
                render_stage_text = st.empty()

                def _render_cb(current: int, total: int, stage: str) -> None:
                    sub_stage = 0.50
                    stage_lower = stage.lower()
                    if "slicing" in stage_lower:
                        sub_stage = 0.10
                    elif "building subtitles" in stage_lower:
                        sub_stage = 0.25
                    elif "searching b-roll" in stage_lower:
                        sub_stage = 0.40
                    elif "downloading b-roll" in stage_lower:
                        sub_stage = 0.55
                    elif "crop" in stage_lower or "framing" in stage_lower:
                        sub_stage = 0.70
                    elif "overlay" in stage_lower or "b-roll" in stage_lower:
                        sub_stage = 0.85
                    elif "burning subtitles" in stage_lower:
                        sub_stage = 0.95

                    fraction = (current + sub_stage) / max(total, 1)
                    pct = int(min(fraction, 0.99) * 100)
                    render_progress.progress(
                        min(fraction, 0.99),
                        text=f"Rendering ({pct}%): [{current + 1}/{total}] {stage}",
                    )
                    render_stage_text.markdown(f"**[{current + 1}/{total}]** {stage}")

                try:
                    with st.spinner("Rendering selected clips..."):
                        _all_candidates, render_results = run_url_pipeline(
                            url=url_input.strip(),
                            settings=settings,
                            clip_indices=selected_indices,
                            progress_cb=_render_cb,
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


if __name__ == "__main__":
    main()

