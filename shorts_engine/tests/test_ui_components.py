"""
tests/test_ui_components.py — Unit tests for modular UI tabs and components.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from shorts_engine.config import Settings
from shorts_engine.pipeline import ProcessingResult
from shorts_engine.services.seo_generator import SeoMetadata
from shorts_engine.ui.autopilot_tab import render_autopilot_tab
from shorts_engine.ui.components.result_card import (
    get_result_download_filename,
    render_result_card,
)
from shorts_engine.ui.components.review_card import render_review_and_approve_list
from shorts_engine.ui.file_upload_tab import render_file_upload_tab
from shorts_engine.ui.url_tab import render_video_url_tab


def _mock_columns(spec, **_kw):
    count = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
    return [MagicMock() for _ in range(count)]


def test_get_result_download_filename():
    # When SEO has a title, it sanitizes it
    res = ProcessingResult(
        input_file=Path("/tmp/input.mp4"),
        output_file=Path("/tmp/output_short.mp4"),
        seo=SeoMetadata(
            title="Το Μυστικό του Σύμπαντος!",
            description="Desc",
            tags=("tag1",),
            primary_keyword="μυστικό",
            pinned_comment="comment",
            curiosity_title="Curiosity",
            authority_title="Authority",
            contrarian_title="Contrarian",
        ),
        success=True,
    )
    filename = get_result_download_filename(res, extension="mp4")
    assert filename.endswith(".mp4")
    assert "To_Mystiko_tou_Sympantos" in filename or "mp4" in filename

    # When SEO is None, falls back to output file name
    res_no_seo = ProcessingResult(
        input_file=Path("/tmp/input.mp4"),
        output_file=Path("/tmp/clip_01_short.mp4"),
        seo=None,
        success=True,
    )
    fallback_fn = get_result_download_filename(res_no_seo, extension="mp4")
    assert fallback_fn == "clip_01_short.mp4"


def test_render_result_card_smoke(tmp_path: Path):
    dummy_out = tmp_path / "out.mp4"
    dummy_out.write_bytes(b"dummy")

    res = ProcessingResult(
        input_file=tmp_path / "in.mp4",
        output_file=dummy_out,
        seo=SeoMetadata(
            title="Test Short",
            description="Desc",
            tags=("test",),
            primary_keyword="test",
            pinned_comment="pinned",
            curiosity_title="Curiosity",
            authority_title="Authority",
            contrarian_title="Contrarian",
        ),
        success=True,
        hook_text="Hook line",
        virality_score=8.5,
        broll_query="nature footage",
    )

    with patch("streamlit.container"), \
         patch("streamlit.columns", side_effect=_mock_columns), \
         patch("streamlit.markdown"), \
         patch("streamlit.video"), \
         patch("streamlit.download_button"), \
         patch("streamlit.expander"), \
         patch("streamlit.text"), \
         patch("streamlit.code"):
        render_result_card(res, index=0)


def test_render_review_and_approve_list_empty():
    with patch("streamlit.markdown") as mock_md:
        render_review_and_approve_list([], key_prefix="test")
        mock_md.assert_not_called()


def test_render_file_upload_tab_smoke():
    settings = Settings()
    with patch("streamlit.columns", side_effect=_mock_columns), \
         patch("streamlit.markdown"), \
         patch("streamlit.file_uploader", return_value=[]), \
         patch("streamlit.session_state", {"is_processing": False, "results": []}), \
         patch("streamlit.button", return_value=False):
        render_file_upload_tab(settings)


def test_render_video_url_tab_smoke():
    settings = Settings()
    with patch("streamlit.columns", side_effect=_mock_columns), \
         patch("streamlit.markdown"), \
         patch("streamlit.text_input", return_value=""), \
         patch("streamlit.session_state", {}), \
         patch("streamlit.button", return_value=False):
        render_video_url_tab(settings)


def test_render_autopilot_tab_smoke():
    settings = Settings()
    with patch("streamlit.columns", side_effect=_mock_columns), \
         patch("streamlit.markdown"), \
         patch("streamlit.radio", return_value="🌐 Niche Trend Discovery (Smart Outlier & Velocity Scanner — Recommended)"), \
         patch("streamlit.text_input", return_value=""), \
         patch("streamlit.selectbox", return_value="🔥 Hybrid: Speaker Clip + AI Breakdown (Recommended)"), \
         patch("streamlit.slider", return_value=0.15), \
         patch("streamlit.number_input", return_value=1), \
         patch("streamlit.expander"), \
         patch("streamlit.session_state", {}), \
         patch("streamlit.button", return_value=False):
        render_autopilot_tab(settings)

