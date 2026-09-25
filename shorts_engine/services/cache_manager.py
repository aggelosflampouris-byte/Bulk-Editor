"""
services/cache_manager.py — Centralized caching and history logging for the shorts pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from config import APP_ROOT, DEFAULT_OUTPUT_DIR
except ImportError:
    from shorts_engine.config import APP_ROOT, DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = APP_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_HISTORY_FILE: Path = DEFAULT_OUTPUT_DIR / "project_history.json"
PROCESSED_REGISTRY_FILE: Path = CACHE_DIR / "processed_videos_registry.json"


def _get_cache_path(domain: str, key: str) -> Path:
    """Generate a stable, hashed path for a cache file."""
    domain_dir = CACHE_DIR / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    # Hash the key to prevent filesystem issues with long/invalid characters
    hashed = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return domain_dir / f"{hashed}.pkl"


def load_cache_pickle(domain: str, key: str) -> Any | None:
    """Load a cached python object if it exists."""
    path = _get_cache_path(domain, key)
    if path.exists():
        try:
            with open(path, "rb") as f:
                logger.info("Cache hit for domain '%s', key '%s'", domain, key)
                return pickle.load(f)
        except (OSError, pickle.PickleError, EOFError, AttributeError) as e:
            logger.warning("Failed to load cache for %s/%s: %s", domain, key, e)
    return None


def save_cache_pickle(domain: str, key: str, obj: Any) -> None:
    """Save a python object to the disk cache."""
    path = _get_cache_path(domain, key)
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
            logger.debug("Saved cache for domain '%s', key '%s'", domain, key)
    except (OSError, pickle.PickleError) as e:
        logger.warning("Failed to save cache for %s/%s: %s", domain, key, e)


def _append_entry_to_history(history_file: Path, entry: dict[str, Any]) -> None:
    """Helper to safely read, deduplicate, and append an entry to a history JSON file."""
    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, Any]] = []
        if history_file.is_file():
            try:
                data = json.loads(history_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    history = data
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as read_err:
                logger.debug("Failed reading existing history file %s: %s", history_file, read_err)
                history = []

        vid_id = entry.get("video_id") or entry.get("source_video_id")
        already_has = any(
            isinstance(e, dict) and vid_id and (e.get("video_id") == vid_id or e.get("source_video_id") == vid_id)
            for e in history
        )
        if not already_has:
            history.append(entry)
            history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Failed appending to history file %s: %s", history_file, exc)


def record_processed_video(
    video_id: str,
    url: str = "",
    title: str = "",
    output_dir: Path | None = None,
    mode: str = "autopilot",
    seo: Any | None = None,
    output_file: str | Path | None = None,
) -> None:
    """
    Persistently record a processed video ID, URL, and metadata across sessions.

    Guarantees cross-session deduplication so Autopilot never re-sources
    the same video repeatedly across runs or directory changes.
    """
    clean_id = (video_id or "").strip()

    # Extract clean 11-char ID if a full URL or composite string was passed
    try:
        from services.youtube_transcript_fetcher import extract_youtube_id
    except ImportError:
        try:
            from shorts_engine.services.youtube_transcript_fetcher import (
                extract_youtube_id,
            )
        except ImportError:
            extract_youtube_id = None

    if extract_youtube_id is not None:
        parsed_id = extract_youtube_id(clean_id)
        if parsed_id:
            clean_id = parsed_id

    if not clean_id and url and extract_youtube_id is not None:
        clean_id = extract_youtube_id(url) or ""

    target_url = url.strip() if url else (f"https://www.youtube.com/watch?v={clean_id}" if clean_id else "")

    seo_data = None
    if seo:
        seo_data = {
            "title": getattr(seo, "title", ""),
            "description": getattr(seo, "description", ""),
            "tags": list(getattr(seo, "tags", [])),
            "primary_keyword": getattr(seo, "primary_keyword", ""),
            "pinned_comment": getattr(seo, "pinned_comment", ""),
            "alt_titles": list(getattr(seo, "alt_titles", [])),
        }

    now_iso = datetime.now(timezone.utc).isoformat()
    entry_id = hashlib.md5(f"{clean_id}_{now_iso}".encode()).hexdigest()[:12]
    input_filename = f"{title} [{clean_id}].mp4" if title and clean_id else (f"[{clean_id}].mp4" if clean_id else "")

    entry: dict[str, Any] = {
        "id": entry_id,
        "timestamp": now_iso,
        "video_id": clean_id,
        "source_video_id": clean_id,
        "url": target_url,
        "title": title,
        "mode": mode,
        "input_file": input_filename,
        "output_file": str(output_file) if output_file else "",
        "seo": seo_data,
    }

    # 1. Update persistent cache registry (.cache/processed_videos_registry.json)
    try:
        registry: dict[str, Any] = {}
        if PROCESSED_REGISTRY_FILE.is_file():
            try:
                loaded = json.loads(PROCESSED_REGISTRY_FILE.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    registry = loaded
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                registry = {}
        if clean_id:
            registry[clean_id] = entry
        if target_url:
            registry[target_url] = entry
        PROCESSED_REGISTRY_FILE.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Failed updating persistent cache registry: %s", exc)

    # 2. Append to central project_history.json
    _append_entry_to_history(GLOBAL_HISTORY_FILE, entry)

    # 3. Append to output_dir project_history.json if specified and distinct
    if output_dir is not None:
        try:
            resolved_out = Path(output_dir).resolve()
            target_file = resolved_out if resolved_out.is_file() else (resolved_out / "project_history.json")
            if target_file.resolve() != GLOBAL_HISTORY_FILE.resolve():
                _append_entry_to_history(target_file, entry)
        except (OSError, ValueError, KeyError) as exc:
            logger.debug("Failed appending to custom output_dir history: %s", exc)

    logger.info("Persistently recorded processed video [%s] '%s' (mode: %s)", clean_id, title[:40], mode)


def log_project_history(result: Any, output_dir: Path | None = None) -> None:
    """
    Append successful project details to persistent history files.
    Expects a ProcessingResult object. Maintained for pipeline.py backward compatibility.
    """
    if not getattr(result, "success", False) or not getattr(result, "output_file", None):
        return

    source_vid = getattr(result, "source_video_id", None)
    if not source_vid and getattr(result, "input_file", None):
        input_str = str(result.input_file.name)
        if "[" in input_str and "]" in input_str:
            sub = input_str.split("[")[-1].split("]")[0].strip()
            if len(sub) == 11:
                source_vid = sub

    clean_title = ""
    if getattr(result, "seo", None) and getattr(result.seo, "title", None):
        clean_title = str(result.seo.title)
    elif getattr(result, "input_file", None):
        clean_title = result.input_file.stem

    record_processed_video(
        video_id=source_vid or "",
        url=f"https://www.youtube.com/watch?v={source_vid}" if source_vid else "",
        title=clean_title,
        output_dir=output_dir,
        mode="clip",
        seo=getattr(result, "seo", None),
        output_file=result.output_file,
    )


def get_processed_video_ids(output_dir: Path | None = None) -> set[str]:
    """
    Consolidate processed YouTube video IDs, URLs, and source stems from:
      1. Persistent cache registry (.cache/processed_videos_registry.json)
      2. Central project_history.json (output/project_history.json)
      3. Active run's output_dir (if provided)
      4. Subdirectories in DEFAULT_OUTPUT_DIR
    """
    processed: set[str] = set()

    def _extract_from_list(entries: list[Any]) -> None:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            vid_id = entry.get("video_id") or entry.get("source_video_id")
            if vid_id:
                processed.add(str(vid_id))
            url = entry.get("url")
            if url:
                processed.add(str(url))
            input_file = entry.get("input_file", "")
            if "[" in input_file and "]" in input_file:
                sub = input_file.split("[")[-1].split("]")[0].strip()
                if len(sub) == 11:
                    processed.add(sub)
            if input_file:
                processed.add(Path(input_file).stem)

    # 1. Read persistent cache registry
    if PROCESSED_REGISTRY_FILE.is_file():
        try:
            reg_data = json.loads(PROCESSED_REGISTRY_FILE.read_text(encoding="utf-8"))
            if isinstance(reg_data, dict):
                for k, v in reg_data.items():
                    processed.add(str(k))
                    if isinstance(v, dict):
                        if v.get("video_id"):
                            processed.add(str(v["video_id"]))
                        if v.get("url"):
                            processed.add(str(v["url"]))
            elif isinstance(reg_data, list):
                _extract_from_list(reg_data)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.debug("Failed reading processed cache registry: %s", exc)

    # 2. Read central project_history.json
    if GLOBAL_HISTORY_FILE.is_file():
        try:
            gh_data = json.loads(GLOBAL_HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(gh_data, list):
                _extract_from_list(gh_data)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.debug("Failed reading central project_history: %s", exc)

    # 3. Read specific output_dir if provided
    if output_dir is not None:
        try:
            out_p = Path(output_dir).resolve()
            h_file = out_p if out_p.is_file() else (out_p / "project_history.json")
            if h_file.is_file():
                out_data = json.loads(h_file.read_text(encoding="utf-8"))
                if isinstance(out_data, list):
                    _extract_from_list(out_data)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.debug("Failed reading output_dir history: %s", exc)

    # 4. Quick scan of recent subdirectories in DEFAULT_OUTPUT_DIR
    if DEFAULT_OUTPUT_DIR.is_dir():
        try:
            for sub_hist in DEFAULT_OUTPUT_DIR.glob("*/project_history.json"):
                try:
                    sub_data = json.loads(sub_hist.read_text(encoding="utf-8"))
                    if isinstance(sub_data, list):
                        _extract_from_list(sub_data)
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as scan_err:
                    logger.debug("Skipping sub-history file %s: %s", sub_hist, scan_err)
                    continue
        except (OSError, ValueError) as scan_dir_err:
            logger.debug("Failed scanning subdirectories for history: %s", scan_dir_err)

    return processed


def is_video_already_processed(video_id_or_url: str, output_dir: Path | None = None) -> bool:
    """
    Check if a YouTube video ID or URL has already been processed and logged.
    Guarantees cross-session deduplication.
    """
    if not video_id_or_url or not isinstance(video_id_or_url, str):
        return False

    val = video_id_or_url.strip()
    if not val:
        return False

    processed = get_processed_video_ids(output_dir)
    if not processed:
        return False

    if val in processed:
        return True

    try:
        from services.youtube_transcript_fetcher import extract_youtube_id
    except ImportError:
        try:
            from shorts_engine.services.youtube_transcript_fetcher import (
                extract_youtube_id,
            )
        except ImportError:
            extract_youtube_id = None

    if extract_youtube_id is not None:
        try:
            extracted = extract_youtube_id(val)
            if extracted and (
                extracted in processed
                or f"https://www.youtube.com/watch?v={extracted}" in processed
                or f"https://youtu.be/{extracted}" in processed
            ):
                return True
        except (OSError, ValueError, KeyError) as ext_err:
            logger.debug("Failed extracting YouTube ID: %s", ext_err)

    # Resilient URL component extraction
    if "youtu.be/" in val:
        short_id = val.split("youtu.be/")[-1].split("?")[0].split("/")[0].strip()
        if short_id and short_id in processed:
            return True
    if "watch?v=" in val:
        watch_id = val.split("watch?v=")[-1].split("&")[0].split("/")[0].strip()
        if watch_id and watch_id in processed:
            return True
    if "/shorts/" in val:
        shorts_id = val.split("/shorts/")[-1].split("?")[0].split("/")[0].strip()
        if shorts_id and shorts_id in processed:
            return True

    return False
