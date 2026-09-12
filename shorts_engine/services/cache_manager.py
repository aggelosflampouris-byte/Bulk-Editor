"""
services/cache_manager.py — Centralized caching and history logging for the shorts pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from config import APP_ROOT
except ImportError:
    from shorts_engine.config import APP_ROOT

logger = logging.getLogger(__name__)

CACHE_DIR = APP_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
        except Exception as e:
            logger.warning("Failed to load cache for %s/%s: %s", domain, key, e)
    return None


def save_cache_pickle(domain: str, key: str, obj: Any) -> None:
    """Save a python object to the disk cache."""
    path = _get_cache_path(domain, key)
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
            logger.debug("Saved cache for domain '%s', key '%s'", domain, key)
    except Exception as e:
        logger.warning("Failed to save cache for %s/%s: %s", domain, key, e)


def log_project_history(result: Any, output_dir: Path) -> None:
    """
    Append the successful project details to a persistent history.json file.
    Expects a ProcessingResult object.
    """
    if not result.success or not result.output_file:
        return
        
    history_file = output_dir / "project_history.json"
    
    # Prepare entry
    seo_data = None
    if result.seo:
        seo_data = {
            "title": result.seo.title,
            "description": result.seo.description,
            "tags": list(result.seo.tags),
            "primary_keyword": result.seo.primary_keyword,
            "pinned_comment": result.seo.pinned_comment,
            "alt_titles": list(result.seo.alt_titles),
        }
        
    entry = {
        "id": hashlib.md5(f"{result.input_file.name}_{datetime.now().isoformat()}".encode()).hexdigest()[:12],
        "timestamp": datetime.now().isoformat(),
        "input_file": str(result.input_file.name),
        "output_file": str(result.output_file.name),
        "seo": seo_data,
        "broll_query": result.broll_query,
        "hook_text": result.hook_text,
        "virality_score": result.virality_score,
    }
    
    # Load existing history
    history = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as e:
            logger.warning("Failed to read project history file: %s", e)
            
    history.append(entry)
    
    # Save back
    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
            logger.info("Logged project history for '%s' to %s", result.input_file.name, history_file.name)
    except Exception as e:
        logger.warning("Failed to write project history: %s", e)
