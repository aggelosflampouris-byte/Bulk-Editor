"""
services/asset_library.py — Local Media & Asset Library System.

Provides an organized local filesystem repository for media used by the AI engine:
  - assets/music/        (Background score tracks & ambient beds)
  - assets/sfx/          (Sound effects: whooshes, dings, bass drops, clicks, revs)
  - assets/effects/      (Visual overlays: light leaks, dust, glitch, VHS)
  - assets/green_screen/ (Transparent green screen / alpha overlays)

Includes automatic zero-dependency synthesis of basic royalty-free sound effects
via pure Python `wave` + `math`, so SFX are immediately usable out of the box.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Root asset directory under shorts_engine/assets
_ROOT_DIR = Path(__file__).parent.parent.resolve()
ASSETS_DIR = _ROOT_DIR / "assets"
MUSIC_DIR = ASSETS_DIR / "music"
SFX_DIR = ASSETS_DIR / "sfx"
EFFECTS_DIR = ASSETS_DIR / "effects"
GREEN_SCREEN_DIR = ASSETS_DIR / "green_screen"

CategoryType = Literal["music", "sfx", "effects", "green_screen"]


@dataclass(frozen=True)
class AssetItem:
    """
    Representation of an individual media asset in the local library.
    """

    id: str
    name: str
    category: CategoryType
    path: Path
    file_extension: str
    tags: tuple[str, ...]
    duration_seconds: float = 0.0

    @property
    def exists(self) -> bool:
        return self.path.is_file()


def initialize_asset_library() -> None:
    """
    Ensure all asset subdirectories exist and seed default synthetic sound effects
    if the SFX directory is empty.
    """
    for dir_path in (ASSETS_DIR, MUSIC_DIR, SFX_DIR, EFFECTS_DIR, GREEN_SCREEN_DIR):
        dir_path.mkdir(parents=True, exist_ok=True)

    _seed_default_sfx_if_needed()


def _generate_wav(
    filepath: Path,
    duration: float,
    sample_rate: int,
    sample_generator,
) -> None:
    """Generate a 16-bit mono PCM WAV file using a mathematical generator function."""
    num_samples = int(duration * sample_rate)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)        # Mono
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            sample_val = sample_generator(t, duration)
            clamped = max(-1.0, min(1.0, sample_val))
            int_val = int(clamped * 32767.0)
            frames.extend(struct.pack("<h", int_val))
        
        wf.writeframes(frames)


def _seed_default_sfx_if_needed() -> None:
    """
    Synthesize baseline royalty-free sound effects if no SFX are present.
    Creates: whoosh, ding, pop, bass_drop, click, impact.
    """
    sr = 44100

    # 1. Whoosh transition (frequency modulated filtered sweep)
    whoosh_path = SFX_DIR / "whoosh_fast.wav"
    if not whoosh_path.exists():
        def _whoosh_gen(t: float, dur: float) -> float:
            env = math.sin((t / dur) * math.pi) ** 2
            # Frequency sweeps from 180Hz up to 900Hz and back down
            freq = 200.0 + 700.0 * math.sin((t / dur) * math.pi)
            noise = ((math.sin(t * 12345.67) + math.sin(t * 43210.98)) / 2.0) * 0.3
            tone = math.sin(2.0 * math.pi * freq * t) * 0.7
            return (tone + noise) * env * 0.8
        _generate_wav(whoosh_path, duration=0.35, sample_rate=sr, sample_generator=_whoosh_gen)

    # 2. Ding / Bell accent (clear high chime with exponential decay)
    ding_path = SFX_DIR / "ding_bell.wav"
    if not ding_path.exists():
        def _ding_gen(t: float, dur: float) -> float:
            decay = math.exp(-t * 6.0)
            tone1 = math.sin(2.0 * math.pi * 1760.0 * t) * 0.6  # A6
            tone2 = math.sin(2.0 * math.pi * 3520.0 * t) * 0.3  # A7 harmonic
            return (tone1 + tone2) * decay * 0.7
        _generate_wav(ding_path, duration=0.6, sample_rate=sr, sample_generator=_ding_gen)

    # 3. Pop / Bubble sound effect (rapid upward pitch blip)
    pop_path = SFX_DIR / "pop_subtle.wav"
    if not pop_path.exists():
        def _pop_gen(t: float, dur: float) -> float:
            decay = math.exp(-t * 35.0)
            freq = 300.0 + (t / dur) * 1200.0
            return math.sin(2.0 * math.pi * freq * t) * decay * 0.85
        _generate_wav(pop_path, duration=0.12, sample_rate=sr, sample_generator=_pop_gen)

    # 4. Bass Drop / Sub impact (deep sub-bass sweep down)
    bass_path = SFX_DIR / "bass_drop.wav"
    if not bass_path.exists():
        def _bass_gen(t: float, dur: float) -> float:
            decay = math.exp(-t * 3.5)
            freq = max(38.0, 140.0 - (t / dur) * 95.0)
            return math.sin(2.0 * math.pi * freq * t) * decay * 0.95
        _generate_wav(bass_path, duration=0.8, sample_rate=sr, sample_generator=_bass_gen)

    # 5. Mechanical / Tool Click (industrial wrench/switch feel)
    click_path = SFX_DIR / "mech_click.wav"
    if not click_path.exists():
        def _click_gen(t: float, dur: float) -> float:
            decay = math.exp(-t * 70.0)
            freq = 800.0 if t < 0.015 else 450.0
            return math.sin(2.0 * math.pi * freq * t) * decay * 0.9
        _generate_wav(click_path, duration=0.08, sample_rate=sr, sample_generator=_click_gen)

    # 6. Heavy Impact / Stinger (cinematic hit)
    impact_path = SFX_DIR / "impact_heavy.wav"
    if not impact_path.exists():
        def _impact_gen(t: float, dur: float) -> float:
            decay = math.exp(-t * 4.5)
            f_low = 65.0 * math.exp(-t * 3.0)
            tone = math.sin(2.0 * math.pi * f_low * t) * 0.8
            burst = (math.sin(t * 98765.43) * 0.4) if t < 0.05 else 0.0
            return (tone + burst) * decay * 0.9
        _generate_wav(impact_path, duration=0.9, sample_rate=sr, sample_generator=_impact_gen)


def _get_category_dir(category: CategoryType) -> Path:
    mapping = {
        "music": MUSIC_DIR,
        "sfx": SFX_DIR,
        "effects": EFFECTS_DIR,
        "green_screen": GREEN_SCREEN_DIR,
    }
    return mapping.get(category, ASSETS_DIR)


def list_assets(category: CategoryType | None = None) -> list[AssetItem]:
    """
    List all assets in the local library, optionally filtered by category.
    """
    initialize_asset_library()
    categories: list[CategoryType] = [category] if category else ["music", "sfx", "effects", "green_screen"]
    results: list[AssetItem] = []

    valid_audio_exts = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}
    valid_video_exts = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
    valid_image_exts = {".png", ".jpg", ".jpeg", ".webp"}

    for cat in categories:
        target_dir = _get_category_dir(cat)
        if not target_dir.is_dir():
            continue

        for entry in sorted(target_dir.iterdir()):
            if entry.name.startswith(".") or not entry.is_file():
                continue

            ext = entry.suffix.lower()
            allowed = (
                valid_audio_exts if cat in ("music", "sfx")
                else (valid_video_exts | valid_image_exts)
            )
            if ext not in allowed:
                continue

            # Derive tags from file stem
            stem_clean = entry.stem.lower().replace("-", "_")
            tags = tuple(filter(None, stem_clean.split("_")))

            results.append(
                AssetItem(
                    id=f"{cat}_{entry.stem}",
                    name=entry.stem.replace("_", " ").title(),
                    category=cat,
                    path=entry.resolve(),
                    file_extension=ext,
                    tags=tags,
                )
            )

    return results


def find_assets_by_tags(tags: list[str], category: CategoryType | None = None) -> list[AssetItem]:
    """
    Find media assets matching one or more semantic tags.
    """
    normalized_tags = [t.lower().strip() for t in tags if t.strip()]
    candidates = list_assets(category)
    if not normalized_tags:
        return candidates

    scored: list[tuple[int, AssetItem]] = []
    for item in candidates:
        match_count = sum(1 for t in normalized_tags if any(t in item_tag for item_tag in item.tags))
        if match_count > 0:
            scored.append((match_count, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored]


def import_user_asset(
    source_file: Path,
    category: CategoryType,
    custom_name: str | None = None,
) -> AssetItem:
    """
    Copy a user-provided media asset into the local library folder.
    """
    initialize_asset_library()
    if not source_file.is_file():
        raise FileNotFoundError(f"Source asset file does not exist: {source_file}")

    target_dir = _get_category_dir(category)
    clean_stem = (custom_name or source_file.stem).replace(" ", "_").lower()
    dest_path = target_dir / f"{clean_stem}{source_file.suffix.lower()}"

    # Counter to avoid unintentional overwrites
    counter = 1
    while dest_path.exists() and dest_path != source_file.resolve():
        dest_path = target_dir / f"{clean_stem}_{counter}{source_file.suffix.lower()}"
        counter += 1

    shutil.copy2(source_file, dest_path)
    logger.info("Imported media asset into local library: %s -> %s", source_file.name, dest_path.name)

    tags = tuple(filter(None, dest_path.stem.lower().split("_")))
    return AssetItem(
        id=f"{category}_{dest_path.stem}",
        name=dest_path.stem.replace("_", " ").title(),
        category=category,
        path=dest_path.resolve(),
        file_extension=dest_path.suffix.lower(),
        tags=tags,
    )
