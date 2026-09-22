"""
ui/url_tab.py — Tab component for processing videos directly from URLs (YouTube, TikTok, IG).
"""

from __future__ import annotations

import streamlit as st

try:
    from config import Settings
    from pipeline import ProcessingResult, run_url_pipeline
    from services.clip_selector import ClipCandidate
    from ui.components.result_card import render_result_card
except ImportError:
    from shorts_engine.config import Settings
    from shorts_engine.pipeline import ProcessingResult, run_url_pipeline
    from shorts_engine.services.clip_selector import ClipCandidate
    from shorts_engine.ui.components.result_card import render_result_card


def _render_url_candidate_table(candidates: list[ClipCandidate]) -> list[int]:
    """
    Render an interactive table of AI-selected clip candidates.
    Each row has a checkbox, clip index, duration, time range, SEO title,
    and hook summary. Returns the list of 1-based clip indices that remain checked.
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
        col.markdown(
            f"<span style='font-size:0.75rem;color:#71717a;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.05em'>{hdr}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin:0.4rem 0;border-color:#27272a'>", unsafe_allow_html=True)

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
            display_title = clip.seo.title[:55] + "…" if len(clip.seo.title) > 55 else clip.seo.title
            st.markdown(
                f"<span style='font-size:0.88rem;color:#f4f4f5'>"
                f"<strong>{display_title}</strong></span>",
                unsafe_allow_html=True,
            )

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
    """Render assembled URL pipeline results alongside their clip-level metadata."""
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
        stem = result.output_file.stem.replace("_short", "") if result.output_file else ""
        matched_clip = candidate_map.get(stem)

        render_result_card(result, idx)

        if matched_clip and result.success:
            st.markdown(
                f'<div style="margin:-0.4rem 0 0.8rem 0;padding:0.4rem 0.6rem;'
                f'background:#141416;border-radius:4px;'
                f'font-size:0.78rem;color:#71717a;">'
                f'<em>{matched_clip.hook_summary}</em>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_video_url_tab(settings: Settings) -> None:
    """Render the Video URL analysis and clip extraction tab."""
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

        col_a, _ = st.columns([1, 4])
        with col_a:
            analyze_clicked = st.button(
                "🔍 Analyze Video",
                key="analyze_url_btn",
                use_container_width=True,
                type="primary",
                disabled=bool(settings_errors_url) or not url_input.strip(),
            )

        if analyze_clicked and url_input.strip() and not settings_errors_url:
            st.session_state["url_is_analyzing"] = True

        if st.session_state.get("url_is_analyzing"):
            url_progress_bar = st.progress(0.0)
            url_stage_text = st.empty()

            def _url_analysis_cb(idx: int, total: int, stage: str):
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
                        clip_indices=[],
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
        selected_indices: list[int] = st.session_state.get("url_selected_indices", [])
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
                    disabled=n_selected < min_required or not url_input.strip(),
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
