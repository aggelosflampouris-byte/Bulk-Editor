"""
config.py — Settings, PATH injection, binary assertions, and style constants.

This module is the single source of truth for all configuration. It must be
imported before any service module is used so that PATH injection runs first.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# ── .env Auto-Loader (zero external dependencies) ─────────────────────────────

def _load_dotenv(env_file: Path | None = None) -> None:
    """
    Parse a .env file and inject its key=value pairs into os.environ.

    Only keys that are NOT already set in the environment are written,
    so real environment variables always take precedence over the file.

    Supported syntax:
      KEY=value
      KEY="quoted value"
      # comment lines (ignored)
      blank lines (ignored)

    Args:
        env_file: Path to the .env file.  Defaults to APP_ROOT / '.env'.
    """
    # APP_ROOT is not yet defined at call time, so compute it inline.
    root = Path(__file__).parent.resolve()
    target = env_file if env_file is not None else root / ".env"

    if not target.is_file():
        return  # No .env present — silently skip

    with target.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            # Skip blanks and comments
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Honour existing env vars — never overwrite them
            if key and key not in os.environ:
                os.environ[key] = value


# Load .env at import time so os.environ is populated before Settings is used.
_load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────

# Root of the shorts_engine package (directory containing this file)
APP_ROOT: Path = Path(__file__).parent.resolve()

# Bundled FFmpeg binary directory (populated by run.bat on Windows)
BUNDLED_BIN_DIR: Path = APP_ROOT / "bin"

# Default output directory
DEFAULT_OUTPUT_DIR: Path = APP_ROOT / "output"

# Default bundled music directory
DEFAULT_MUSIC_DIR: Path = APP_ROOT / "assets" / "music"


# ── Runtime PATH Injection ─────────────────────────────────────────────────────

def inject_ffmpeg_path() -> None:
    """
    Prepend the bundled ./bin directory to PATH so that subprocess calls to
    'ffmpeg'/'ffprobe' resolve to the locally downloaded static build.

    This is a no-op if ffmpeg is already available on the system PATH.
    """
    if shutil.which("ffmpeg"):
        # Already on PATH — nothing to do
        return

    if BUNDLED_BIN_DIR.is_dir() and (BUNDLED_BIN_DIR / _ffmpeg_exe_name()).exists():
        os.environ["PATH"] = str(BUNDLED_BIN_DIR) + os.pathsep + os.environ.get("PATH", "")


def _ffmpeg_exe_name() -> str:
    """Return the platform-specific ffmpeg binary name."""
    return "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"


# ── Binary Assertions ──────────────────────────────────────────────────────────

def assert_system_binaries() -> None:
    """
    Assert that 'ffmpeg' and 'ffprobe' are resolvable on PATH.

    Raises:
        RuntimeError: with actionable fix instructions if either binary is missing.
    """
    # Inject bundled path first so the check accounts for it
    inject_ffmpeg_path()

    missing: list[str] = []
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            missing.append(binary)

    if missing:
        system = platform.system()
        if system == "Windows":
            hint = (
                "On Windows: re-run run.bat which auto-downloads FFmpeg, "
                "or install manually and add to PATH."
            )
        elif system == "Darwin":
            hint = "On macOS: run `brew install ffmpeg`."
        else:
            hint = "On Linux: run `sudo apt install ffmpeg` or equivalent."

        raise RuntimeError(
            f"Missing required system binaries: {missing}.\n{hint}"
        )


# ── ASS Subtitle Styles ────────────────────────────────────────────────────────

# Default subtitle style — Greek-safe Arial.
#
# Key positioning parameters:
#   Alignment = 2  (bottom-centre in the Numpad layout)
#   MarginV   = 540  — pushes the baseline 540 px above the bottom edge of the
#               1920-px frame, placing captions in the speaker's lower-third
#               (roughly 28% up) rather than the dead-zone at the very bottom.
#
# Format order: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,
#   OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,
#   ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,
#   Alignment, MarginL, MarginR, MarginV, Encoding
ASS_STYLE_LINE: str = (
    "Style: Default,Impact,64,&H00000000,&H00000000,&H00000000,&H00FFFFFF,"
    "-1,0,0,0,100,100,0,0,3,4,0,2,80,80,540,1"
)

# Highlight style — SecondaryColour is unused here because we inject karaoke.
# PrimaryColour is Red (&H000000FF) with a White Box.
ASS_HIGHLIGHT_STYLE_LINE: str = (
    "Style: Highlight,Impact,64,&H000000FF,&H00000000,&H00000000,&H00FFFFFF,"
    "-1,0,0,0,100,100,0,0,3,4,0,2,80,80,540,1"
)

# Full ASS file header template.  {dialogue_lines} is replaced at generation time.
ASS_HEADER_TEMPLATE: str = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style_line}
{highlight_style_line}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
{dialogue_lines}
"""


# ── Settings Dataclass ─────────────────────────────────────────────────────────

@dataclass
class Settings:
    """
    Centralised, typed configuration container.

    All values have safe defaults. API keys must be provided at runtime
    (from the Streamlit sidebar or environment variables) and are never
    persisted to disk by this module.
    """

    # ── API Keys ───────────────────────────────────────────────
    # Defaults read from environment / .env so the sidebar can be left blank.
    pexels_api_key: str = field(
        default_factory=lambda: os.environ.get("PEXELS_API_KEY", "")
    )
    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "")
    )

    # ── Transcription ──────────────────────────────────────────
    # Whisper model size: tiny | base | small | medium | large-v3
    whisper_model_size: str = "large-v3"
    # Always CPU — no CUDA complexity
    whisper_device: str = "cpu"
    # Compute type appropriate for CPU
    whisper_compute_type: str = "int8"
    # Beam search size (5 = full beam search for highest accuracy)
    whisper_beam_size: int = 5



    # ── Active Speaker Framing (YOLO-face) ─────────────────────
    enable_face_tracking: bool = True

    # ── VFX / Colour Grading ──────────────────────────────────
    # Enable transcript- and YOLO-driven colour grading applied after subtitle burn.
    enable_vfx: bool = True
    # YOLO model weights used for scene analysis (same file as face tracker).
    vfx_yolo_model: str = "yolov8n.pt"

    # ── Output ────────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)

    # ── Outro ─────────────────────────────────────────────────
    # Path to an optional outro bumper video. None = skip concatenation.
    outro_path: Path | None = None

    # ── B-Roll & Transitions ──────────────────────────────────
    # Target duration (seconds) for each B-roll overlay
    broll_overlay_duration: float = 5.0
    # Timestamp offset (seconds from start) at which B-roll starts
    broll_start_offset: float = 3.0
    # Apply a Ken Burns zoom effect to B-roll
    broll_ken_burns: bool = True
    # Apply a Picture-in-Picture / Split-Screen effect to B-roll
    broll_split_screen: bool = False
    # Transition style between main clip, B-roll, and outro: "fade" | "flash" | "none"
    transition_type: str = "fade"
    # Transition duration in seconds (default 0.35s for snappy short-form pacing)
    transition_duration: float = 0.35

    # ── URL / Clip Selection ───────────────────────────────────
    # Minimum number of clips to extract per source video (default: 3)
    min_clips: int = 3
    # Maximum number of clips to extract per source video (1–10)
    max_clips: int = 10
    # Minimum clip duration in seconds
    clip_min_duration: float = 35.0
    # Maximum clip duration in seconds
    clip_max_duration: float = 50.0
    # Reject source videos longer than this many seconds (2 hours default)
    max_source_duration_seconds: int = 7200
    
    # ── SEO Engine ─────────────────────────────────────────────
    # Optional Brand Voice injection to tailor Gemini's writing tone
    brand_voice: str = ""

    # ── Background Music ──────────────────────────────────────
    # Enable subtle background audio bed
    enable_bg_music: bool = True
    # Track preset: "ambient_calm" | "dramatic_pulse" | "upbeat_groove" | "custom" | "none"
    bg_music_track: str = "ambient_calm"
    # Optional path to custom uploaded audio file (used when bg_music_track == "custom")
    bg_music_path: Path | None = None
    # Music volume level (0.01 to 0.50, default 0.20 = 20% volume)
    bg_music_volume: float = 0.20
    # Dynamic speech ducking (sidechain compression to dip music during speech)
    bg_music_ducking: bool = True

    # ── Target dimensions ─────────────────────────────────────
    target_width: int = 1080
    target_height: int = 1920

    def resolve_bg_music_path(self) -> Path | None:
        """
        Resolve the Path to the active background music file, or None if disabled.
        """
        if not self.enable_bg_music or self.bg_music_track in ("none", ""):
            return None

        if self.bg_music_track == "custom":
            if self.bg_music_path and self.bg_music_path.is_file():
                return self.bg_music_path
            return None

        # Check default music directory for preset files
        for ext in (".m4a", ".mp3", ".wav", ".aac"):
            preset_file = DEFAULT_MUSIC_DIR / f"{self.bg_music_track}{ext}"
            if preset_file.is_file():
                return preset_file

        return None

    def validate(self) -> list[str]:
        """
        Return a list of human-readable validation error strings.
        An empty list means the settings are valid.
        """
        errors: list[str] = []

        if self.outro_path is not None and not self.outro_path.is_file():
            errors.append(f"Outro file not found: {self.outro_path}")

        if self.whisper_model_size not in {
            "tiny", "base", "small", "medium", "large-v3"
        }:
            errors.append(
                f"Invalid whisper_model_size '{self.whisper_model_size}'. "
                "Choose from: tiny, base, small, medium, large-v3."
            )

        if self.broll_overlay_duration <= 0:
            errors.append("broll_overlay_duration must be positive.")

        if self.broll_start_offset < 0:
            errors.append("broll_start_offset must be >= 0.")

        if self.transition_type not in {"fade", "flash", "none"}:
            errors.append(
                f"Invalid transition_type '{self.transition_type}'. "
                "Choose from: fade, flash, none."
            )

        if not (0.0 <= self.transition_duration <= 2.0):
            errors.append("transition_duration must be between 0.0 and 2.0 seconds.")

        if self.enable_bg_music:
            if self.bg_music_track not in {
                "ambient_calm", "dramatic_pulse", "upbeat_groove", "custom", "none"
            }:
                errors.append(
                    f"Invalid bg_music_track '{self.bg_music_track}'. "
                    "Choose from: ambient_calm, dramatic_pulse, upbeat_groove, custom, none."
                )

            if not (0.01 <= self.bg_music_volume <= 0.50):
                errors.append("bg_music_volume must be between 0.01 and 0.50.")

            if self.bg_music_track == "custom":
                if self.bg_music_path is None or not self.bg_music_path.is_file():
                    errors.append(f"Custom background music file not found: {self.bg_music_path}")

        if not (1 <= self.min_clips <= 10):
            errors.append("min_clips must be between 1 and 10.")

        if not (1 <= self.max_clips <= 10):
            errors.append("max_clips must be between 1 and 10.")

        if self.min_clips > self.max_clips:
            errors.append("min_clips must be less than or equal to max_clips.")

        if self.clip_min_duration < 5:
            errors.append("clip_min_duration must be at least 5 seconds.")

        if self.clip_max_duration > 60:
            errors.append("clip_max_duration must be at most 60 seconds.")

        if self.clip_min_duration >= self.clip_max_duration:
            errors.append("clip_min_duration must be less than clip_max_duration.")

        if self.max_source_duration_seconds <= 0:
            errors.append("max_source_duration_seconds must be positive.")

        return errors
