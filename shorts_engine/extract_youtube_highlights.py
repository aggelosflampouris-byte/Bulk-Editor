"""
extract_youtube_highlights.py — YouTube transcript extraction and local Ollama analysis.

Workflow:
  1. Fetch YouTube transcript directly via youtube-transcript-api (zero video download).
  2. If captions are disabled/unavailable, fallback to downloading audio and local faster-whisper.
  3. Send transcript to local Ollama LLM (Qwen 2.5 / Mistral Small) to extract:
     - Target SEO Tags (comma-separated, keyword-focused)
     - Video Chapters (with precise start/end timestamps)
     - High-retention hooks for vertical clips/Shorts (minimum 3 clips)
  4. Output structured JSON.

Usage:
  python shorts_engine/extract_youtube_highlights.py <YOUTUBE_URL> [--model qwen2.5:32b] [--output results.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

# Ensure shorts_engine package is in sys.path
_PKG_DIR = Path(__file__).parent.resolve()
if str(_PKG_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR.parent))
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from services.youtube_transcript_fetcher import (
    fetch_youtube_transcript,
    format_transcript_for_llm,
    YouTubeTranscriptError,
)
from services.ollama_analyzer import (
    analyze_transcript_with_ollama,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_HOST,
)
from services.transcriber import transcribe, TranscriptionSegment
from services.downloader import download_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract_youtube_highlights")


def process_youtube_highlights(
    url: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    whisper_model: str = "large-v3",
    output_file: Path | None = None,
) -> dict:
    """
    Execute the end-to-end extraction and analysis pipeline.

    Args:
        url:           YouTube video URL.
        model:         Ollama model tag (e.g. 'qwen2.5:32b', 'mistral').
        host:          Ollama host URL.
        whisper_model: Whisper model size for local fallback transcription.
        output_file:   Optional file path to write results JSON.

    Returns:
        Dictionary containing target_seo_tags, video_chapters, and high_retention_hooks.
    """
    print(f"\n{'='*70}\n🚀 YOUTUBE HIGHLIGHTS & SEO EXTRACTION PIPELINE\n{'='*70}")
    print(f"URL:   {url}")
    print(f"Model: {model} (Ollama @ {host})\n")

    # ── Step 1: Attempt youtube-transcript-api ─────────────────────────
    segments: list[TranscriptionSegment] = []
    print("⏳ [1/3] Fetching transcript directly via youtube-transcript-api...")
    try:
        segments = fetch_youtube_transcript(url, languages=("el", "en"))
        print(f"✅ Retrieved {len(segments)} transcript segments directly from YouTube (no download needed).\n")
    except YouTubeTranscriptError as exc:
        print(f"⚠️ YouTube captions unavailable ({exc}).")
        print("⏳ [1/3 Fallback] Downloading audio and transcribing locally with faster-whisper...")
        with tempfile.TemporaryDirectory(prefix="yt_audio_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_path = download_video(url, dest_dir=tmp_path)
            segments = transcribe(video_path, model_size=whisper_model)
        print(f"✅ Transcribed {len(segments)} segments locally with faster-whisper.\n")

    if not segments:
        raise RuntimeError("No transcript segments could be obtained.")

    # ── Step 2: Format Transcript for LLM ──────────────────────────────
    print(f"⏳ [2/3] Preparing transcript ({len(segments)} segments) for LLM ingestion...")
    transcript_text = format_transcript_for_llm(segments)

    # ── Step 3: Analyze with Ollama ────────────────────────────────────
    print(f"⏳ [3/3] Sending to Ollama ({model}) for SEO tags, chapters, and clip extraction...")
    analysis = analyze_transcript_with_ollama(transcript_text, model=model, host=host)

    # ── Format Results ─────────────────────────────────────────────────
    result_payload = {
        "url": url,
        "model": model,
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

    print(f"\n{'='*70}\n🎬 HIGH-RETENTION HOOKS / SHORTS ({len(analysis.high_retention_hooks)} clips):\n{'='*70}")
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
        description="Extract YouTube transcripts and analyze with Ollama for SEO tags, chapters, and viral Shorts."
    )
    parser.add_argument("url", help="YouTube video URL or video ID")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help=f"Ollama model tag (default: {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help=f"Ollama server host (default: {DEFAULT_OLLAMA_HOST})")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model size for fallback transcription (default: large-v3)")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")

    args = parser.parse_args()

    try:
        process_youtube_highlights(
            url=args.url,
            model=args.model,
            host=args.host,
            whisper_model=args.whisper_model,
            output_file=args.output,
        )
    except Exception as exc:
        print(f"\n❌ Error processing highlights: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
