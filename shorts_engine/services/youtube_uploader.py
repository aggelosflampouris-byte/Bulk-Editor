"""
services/youtube_uploader.py — YouTube Data API v3 upload and scheduling.

Responsibilities:
  1. Manage OAuth2 authentication via the installed-application flow.
  2. Upload a video file to YouTube with full metadata.
  3. Support immediate ('public') and scheduled ('private' + publishAt) publishing.
  4. Fetch basic connected-channel statistics.

This module is the single source of truth for all YouTube API interactions.
It has zero FFmpeg or transcription dependencies.

Prerequisites:
  - A `client_secrets.json` file from Google Cloud Console with the
    YouTube Data API v3 enabled. Place it alongside this file or at the
    project root. The path is configurable via the YOUTUBE_CLIENT_SECRETS env var.
  - On first run, a browser window will open for OAuth2 consent. The token
    is persisted to `yt_token.json` in the project root for subsequent runs.

Required packages (add to requirements.txt):
  google-api-python-client>=2.100.0
  google-auth-oauthlib>=1.0.0
  google-auth-httplib2>=0.2.0
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_SCOPES: list[str] = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

_PKG_DIR = Path(__file__).parent.parent.resolve()
_DEFAULT_SECRETS_PATH = _PKG_DIR / "client_secrets.json"
_DEFAULT_TOKEN_PATH = _PKG_DIR / "yt_token.json"

# 8 MB upload chunks (must be a multiple of 256 KB)
_UPLOAD_CHUNK_SIZE: int = 8 * 1024 * 1024


# ── Public Exception Types ─────────────────────────────────────────────────────


class YouTubeAuthError(RuntimeError):
    """Raised when OAuth2 authentication cannot be completed."""


class YouTubeUploadError(RuntimeError):
    """Raised when a video upload fails."""


# ── Path Helpers ───────────────────────────────────────────────────────────────


def _get_secrets_path() -> Path:
    env_path = os.environ.get("YOUTUBE_CLIENT_SECRETS", "").strip()
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
    if _DEFAULT_SECRETS_PATH.is_file():
        return _DEFAULT_SECRETS_PATH
    raise YouTubeAuthError(
        f"client_secrets.json not found.\n"
        f"Expected at: {_DEFAULT_SECRETS_PATH}\n"
        "Download it from Google Cloud Console → APIs & Services → Credentials → "
        "OAuth 2.0 Client IDs, then place it in the shorts_engine directory."
    )


def _get_token_path() -> Path:
    env_path = os.environ.get("YOUTUBE_TOKEN_PATH", "").strip()
    return Path(env_path) if env_path else _DEFAULT_TOKEN_PATH


# ── Authentication ─────────────────────────────────────────────────────────────


def authenticate():
    """
    Build and return an authenticated YouTube API client resource.

    On first run, triggers the browser-based OAuth2 consent flow and saves
    the token to yt_token.json. Subsequent calls load the cached token and
    refresh it automatically when expired.

    Returns:
        A googleapiclient.discovery.Resource for the YouTube Data API v3.

    Raises:
        YouTubeAuthError: If client_secrets.json is missing or auth fails.
        ImportError:      If required packages are not installed.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError(
            "YouTube upload requires: pip install google-api-python-client "
            "google-auth-oauthlib google-auth-httplib2"
        ) from exc

    token_path = _get_token_path()
    creds = None

    if token_path.is_file():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)
        except Exception as exc:
            logger.warning("Cached YouTube token invalid (%s) — re-authenticating.", exc)
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.info("YouTube OAuth2 token refreshed.")
            except Exception as exc:
                logger.warning("Token refresh failed (%s) — re-running auth flow.", exc)
                creds = None

        if not creds:
            secrets_path = _get_secrets_path()
            try:
                flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), _SCOPES)
                creds = flow.run_local_server(port=0)
                logger.info("YouTube OAuth2 authentication successful.")
            except Exception as exc:
                raise YouTubeAuthError(
                    f"OAuth2 flow failed: {exc}\n"
                    "Ensure client_secrets.json is valid and YouTube Data API v3 is enabled."
                ) from exc

        try:
            token_path.write_text(creds.to_json(), encoding="utf-8")
            logger.info("YouTube token saved to: %s", token_path)
        except OSError as exc:
            logger.warning("Could not save YouTube token: %s", exc)

    return build("youtube", "v3", credentials=creds)


def is_authenticated() -> bool:
    """
    Check whether a valid or refreshable cached OAuth2 token exists without
    triggering any browser flow.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        return False

    token_path = _get_token_path()
    if not token_path.is_file():
        return False

    try:
        creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)
        if creds.valid:
            return True
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")
            return True
    except Exception:
        pass

    return False


def revoke_token() -> None:
    """Delete the locally cached OAuth2 token, forcing re-authentication."""
    token_path = _get_token_path()
    if token_path.is_file():
        token_path.unlink()
        logger.info("YouTube token revoked: %s", token_path)


# ── Channel Info ───────────────────────────────────────────────────────────────


def get_channel_info(youtube_client) -> dict[str, Any]:
    """
    Fetch basic statistics for the authenticated channel.

    Returns:
        Dict with keys: channel_id, title, description, subscriber_count,
        video_count, view_count, thumbnail_url.
    """
    try:
        response = youtube_client.channels().list(
            part="snippet,statistics",
            mine=True,
        ).execute()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch channel info: {exc}") from exc

    items = response.get("items", [])
    if not items:
        raise RuntimeError("No channel found for the authenticated account.")

    item = items[0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    thumbnails = snippet.get("thumbnails", {})
    thumb = (
        thumbnails.get("medium", {}).get("url")
        or thumbnails.get("default", {}).get("url")
        or ""
    )

    return {
        "channel_id": item.get("id", ""),
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "subscriber_count": int(stats.get("subscriberCount", 0)),
        "video_count": int(stats.get("videoCount", 0)),
        "view_count": int(stats.get("viewCount", 0)),
        "thumbnail_url": thumb,
    }


# ── Upload ─────────────────────────────────────────────────────────────────────


def upload_short(
    youtube_client,
    video_path: Path,
    title: str,
    description: str,
    tags: list[str],
    publish_at: datetime | None = None,
    category_id: str = "25",
    progress_cb=None,
) -> str:
    """
    Upload a video to YouTube.

    When publish_at is provided, the video is set to Private and scheduled
    to go Public at the specified UTC datetime. When None, it is published
    immediately as Public.

    Args:
        youtube_client: Authenticated YouTube API resource.
        video_path:     Absolute path to the .mp4 file.
        title:          Video title (truncated to 100 chars).
        description:    Video description (truncated to 5000 chars).
        tags:           Keyword tags (total truncated to 500 chars).
        publish_at:     Optional UTC datetime for scheduled publishing.
        category_id:    YouTube category ID (default '25' = News & Politics).
        progress_cb:    Optional callable(bytes_uploaded, total_bytes).

    Returns:
        YouTube video ID string (e.g. 'dQw4w9WgXcQ').

    Raises:
        FileNotFoundError:  If video_path does not exist.
        YouTubeUploadError: If the upload fails.
    """
    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        raise ImportError(
            "youtube_uploader requires google-api-python-client. "
            "Run: pip install google-api-python-client"
        ) from exc

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Resolve privacy and publish schedule
    if publish_at is not None:
        if publish_at.tzinfo is None:
            publish_at = publish_at.replace(tzinfo=timezone.utc)
        privacy_status = "private"
        publish_at_str: str | None = publish_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        privacy_status = "public"
        publish_at_str = None

    # Enforce YouTube field limits
    safe_title = title[:100]
    safe_description = description[:5000]
    safe_tags: list[str] = []
    total_tag_chars = 0
    for tag in tags:
        if total_tag_chars + len(tag) + 1 > 500:
            break
        safe_tags.append(tag)
        total_tag_chars += len(tag) + 1

    body: dict[str, Any] = {
        "snippet": {
            "title": safe_title,
            "description": safe_description,
            "tags": safe_tags,
            "categoryId": category_id,
            "defaultLanguage": "el",
            "defaultAudioLanguage": "el",
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
            "madeForKids": False,
        },
    }

    if publish_at_str:
        body["status"]["publishAt"] = publish_at_str

    media = MediaFileUpload(
        str(video_path),
        mimetype="video/mp4",
        chunksize=_UPLOAD_CHUNK_SIZE,
        resumable=True,
    )

    logger.info(
        "Uploading '%s' (privacy=%s, scheduled=%s)...",
        video_path.name, privacy_status, publish_at_str or "immediate",
    )

    try:
        request = youtube_client.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media,
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status and progress_cb:
                try:
                    progress_cb(status.resumable_progress, status.total_size)
                except Exception:
                    pass

        video_id: str = response.get("id", "")
        if not video_id:
            raise YouTubeUploadError("Upload succeeded but no video ID was returned.")

        logger.info(
            "Upload complete: %s → https://www.youtube.com/watch?v=%s",
            video_path.name, video_id,
        )
        return video_id

    except YouTubeUploadError:
        raise
    except Exception as exc:
        raise YouTubeUploadError(f"Upload failed for '{video_path.name}': {exc}") from exc
