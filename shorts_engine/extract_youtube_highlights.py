"""
extract_youtube_highlights.py — 100% Cloud-driven YouTube transcript extraction and Gemini analysis.

Zero local model weights. Zero local GPU/RAM overhead.
100% Cloud-powered via youtube-transcript-api and Google Gemini Cloud API.

Workflow:
  1. Fetch YouTube transcript directly via youtube-transcript-api (cloud fetch, zero download).
  2. If captions are disabled/unavailable, fallback to Gemini Cloud Audio API transcription.
  3. Send transcript to Google Gemini Flash Cloud API to extract:
     - Target SEO Tags (comma-separated, keyword-focused)
     - Video Chapters (with precise start/end timestamps based on topic changes)
     - High-retention hooks for vertical clips/Shorts (minimum 3 clips guaranteed)
  4. Output structured JSON.

Usage:
  python shorts_engine/extract_youtube_highlights.py <YOUTUBE_URL> [--output results.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Ensure shorts_engine package is in sys.path
_PKG_DIR = Path(__file__).parent.resolve()
if str(_PKG_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR.parent))
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from services.cloud_analyzer import (
    CloudAnalyzerError,
    analyze_transcript_cloud,
    transcribe_audio_cloud,
)
from services.downloader import download_video
from services.transcriber import TranscriptionSegment
from services.youtube_transcript_fetcher import (
    YouTubeTranscriptError,
    fetch_youtube_transcript,
    format_transcript_for_llm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract_youtube_highlights")


def process_youtube_highlights(
    url: str,
    gemini_api_key: str = "",
    output_file: Path | None = None,
) -> dict:
    """
    Execute the 100% cloud-driven extraction and analysis pipeline.

    Args:
        url:            YouTube video URL or video ID.
        gemini_api_key: Google Gemini API key (defaults to GEMINI_API_KEY environment variable).
        output_file:    Optional file path to write results JSON.

    Returns:
        Dictionary containing target_seo_tags, video_chapters, and high_retention_hooks.
    """
    resolved_key = (gemini_api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not resolved_key:
        raise CloudAnalyzerError("GEMINI_API_KEY is required. Please set GEMINI_API_KEY environment variable or pass --api-key.")

    print(f"\n{'='*70}\n☁️ 100% CLOUD YOUTUBE HIGHLIGHTS & SEO PIPELINE (ZERO LOCAL COMPUTE)\n{'='*70}")
    print(f"URL: {url}\n")

    # ── Step 1: Attempt youtube-transcript-api ─────────────────────────
    segments: list[TranscriptionSegment] = []
    print("⏳ [1/3] Fetching transcript directly via youtube-transcript-api (HTTP cloud fetch)...")
    try:
        segments = fetch_youtube_transcript(url, languages=("el", "en"))
        print(f"✅ Retrieved {len(segments)} transcript segments from YouTube (zero download, zero local compute).\n")
    except YouTubeTranscriptError as exc:
        print(f"⚠️ YouTube captions unavailable ({exc}).")
        print("⏳ [1/3 Cloud Fallback] Transcribing via Google Gemini Cloud Audio API...")
        with tempfile.TemporaryDirectory(prefix="yt_audio_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_path = download_video(url, dest_dir=tmp_path)
            segments = transcribe_audio_cloud(video_path, gemini_api_key=resolved_key)
        print(f"✅ Transcribed {len(segments)} segments in the cloud via Gemini.\n")

    if not segments:
        raise RuntimeError("No transcript segments could be obtained.")

    # ── Step 2: Format Transcript for LLM ──────────────────────────────
    print(f"⏳ [2/3] Preparing transcript ({len(segments)} segments) for Cloud LLM ingestion...")
    transcript_text = format_transcript_for_llm(segments)

    # ── Step 3: Analyze with Gemini Cloud API ──────────────────────────
    print("⏳ [3/3] Analyzing with Google Gemini Cloud API for SEO tags, chapters, and clips...")
    analysis = analyze_transcript_cloud(transcript_text, gemini_api_key=resolved_key)

    # ── Format Results ─────────────────────────────────────────────────
    result_payload = {
        "url": url,
        "engine": "100% Cloud (Google Gemini 2.5 Flash)",
        "target_seo_tags": analysis.target_seo_tags,
        "video_chapters": [
            {
                "title": c.title,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "summary": c.summary,
            }
            for c in analysis.video_chapters
        ],
        "high_retention_hooks": [
            {
                "title": h.title,
                "start_time": h.start_time,
                "end_time": h.end_time,
                "duration": h.duration,
                "hook_summary": h.hook_summary,
                "virality_reason": h.virality_reason,
            }
            for h in analysis.high_retention_hooks
        ],
    }

    # Print summary to console
    print(f"\n{'='*70}\n🎯 TARGET SEO TAGS ({len(analysis.target_seo_tags)} tags):\n{'='*70}")
    print(", ".join(analysis.target_seo_tags))

    print(f"\n{'='*70}\n📖 VIDEO CHAPTERS ({len(analysis.video_chapters)} chapters):\n{'='*70}")
    for i, chap in enumerate(analysis.video_chapters, 1):
        s_m, s_s = divmod(int(chap.start_time), 60)
        e_m, e_s = divmod(int(chap.end_time), 60)
        print(f"  {i}. [{s_m:02d}:{s_s:02d} - {e_m:02d}:{e_s:02d}] {chap.title}")
        if chap.summary:
            print(f"     └─ {chap.summary}")

    print(f"\n{'='*70}\n🎬 HIGH-RETENTION HOOKS / SHORTS (MINIMUM: 3, Total: {len(analysis.high_retention_hooks)}):\n{'='*70}")
    for i, hook in enumerate(analysis.high_retention_hooks, 1):
        s_m, s_s = divmod(int(hook.start_time), 60)
        e_m, e_s = divmod(int(hook.end_time), 60)
        print(f"  Clip #{i}: [{s_m:02d}:{s_s:02d} - {e_m:02d}:{e_s:02d}] ({hook.duration:.1f}s) — \"{hook.title}\"")
        print(f"     ├─ Hook:   {hook.hook_summary}")
        print(f"     └─ Viral:  {hook.virality_reason}")
    print(f"{'='*70}\n")

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)
        print(f"💾 Structured output saved to: {output_file.resolve()}\n")

    return result_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract YouTube transcripts and analyze via 100% Cloud Gemini API (zero local models)."
    )
    parser.add_argument("url", help="YouTube video URL or video ID")
    parser.add_argument("--api-key", default="", help="Google Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")

    args = parser.parse_args()

    try:
        process_youtube_highlights(
            url=args.url,
            gemini_api_key=args.api_key,
            output_file=args.output,
        )
    except Exception as exc:
        print(f"\n❌ Error processing highlights: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
