"""
ui/youtube_upload_tab.py — YouTube upload and scheduling UI tab.

Extracted from app.py to keep the main entry-point lean.
Contains zero business logic; delegates all uploads to services/youtube_uploader.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from config import Settings


def render_youtube_upload_tab(settings: Settings) -> None:
    """Render the 'Upload to YouTube' tab content."""
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

        if not _yt_pkgs_ok:
            return

        # ── Check for client_secrets.json ───────────────────────────────────
        _secrets_path = Path(__file__).parent.parent / "client_secrets.json"
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

        # ── Auth Status ──────────────────────────────────────────────────────
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
            return

        # ── Channel Info Card ────────────────────────────────────────────────
        _render_channel_info(authenticate, get_channel_info)

        st.markdown("---")

        # ── Processed Clips ──────────────────────────────────────────────────
        st.markdown("#### Processed Clips Ready to Upload")
        _render_clip_list(settings, authenticate, upload_short, YouTubeAuthError, YouTubeUploadError)


def _render_channel_info(authenticate, get_channel_info) -> None:
    """Render the connected channel info metrics."""
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


def _render_clip_list(settings: Settings, authenticate, upload_short, YouTubeAuthError, YouTubeUploadError) -> None:
    """Render the list of processed clips available for upload."""
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
        return

    st.caption(
        f"Found {len(all_clips)} clip(s) in `{output_dir}`. "
        "Select a clip to configure and upload."
    )

    for idx, clip_path in enumerate(all_clips):
        _render_clip_upload_card(
            idx=idx,
            clip_path=clip_path,
            authenticate=authenticate,
            upload_short=upload_short,
            YouTubeAuthError=YouTubeAuthError,
            YouTubeUploadError=YouTubeUploadError,
        )


def _render_clip_upload_card(
    idx: int,
    clip_path: Path,
    authenticate,
    upload_short,
    YouTubeAuthError,
    YouTubeUploadError,
) -> None:
    """Render a single clip's upload configuration expander."""
    # Attempt to load companion SEO JSON
    seo_stem = clip_path.stem.replace("_short", "")
    seo_json_path = clip_path.parent / f"seo_{seo_stem}.json"
    seo_data: dict = {}
    if seo_json_path.is_file():
        try:
            seo_data = json.loads(seo_json_path.read_text(encoding="utf-8"))
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

        scheduled_dt: datetime | None = None
        if publish_mode == "Schedule":
            from datetime import time as _time
            sched_date = st.date_input(
                "Publish Date (UTC)",
                value=datetime.now(timezone.utc).date(),
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
                tag_list = [t.strip() for t in upload_tags_str.split(",") if t.strip()]
                with st.spinner(f"Uploading '{clip_path.name}' to YouTube..."):
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
