"""
services/transcriber.py — Greek speech transcription via faster-whisper.

Responsibilities:
  1. Transcribe a video file using the faster-whisper WhisperModel (CPU, int8).
  2. Return typed TranscriptionSegment objects.
  3. Render an ASS subtitle file with Greek-safe font styling.

This module has zero FFmpeg or HTTP dependencies.
"""

from __future__ import annotations

import logging
import string
from collections.abc import Callable
from pathlib import Path

from faster_whisper import WhisperModel

try:
    from config import ASS_HEADER_TEMPLATE, ASS_HIGHLIGHT_STYLE_LINE, ASS_STYLE_LINE
except ImportError:
    from shorts_engine.config import (
        ASS_HEADER_TEMPLATE,
        ASS_HIGHLIGHT_STYLE_LINE,
        ASS_STYLE_LINE,
    )

logger = logging.getLogger(__name__)


# ── Public Types ───────────────────────────────────────────────────────────────

class TranscriptionSegment:
    """
    Immutable typed container for a single whisper transcript segment.
    Uses __slots__ for memory efficiency when processing many segments.
    """

    __slots__ = ("end", "start", "text", "words")

    def __init__(
        self,
        start: float,
        end: float,
        text: str,
        words: list[tuple[float, float, str]] | None = None,
    ) -> None:
        self.start: float = start
        self.end: float = end
        self.text: str = text.strip()
        # Each entry: (word_start, word_end, word_text)
        self.words: list[tuple[float, float, str]] | None = words

    def __repr__(self) -> str:
        return f"TranscriptionSegment(start={self.start:.2f}, end={self.end:.2f}, text={self.text!r})"


# Domain-specific Greek vocabulary injected into the Whisper initial_prompt
# to anchor the decoder to the correct vocabulary before transcription starts.
_DOMAIN_PROMPTS: dict[str, str] = {
    "politics": (
        "Πολιτική, κυβέρνηση, βουλή, πρωθυπουργός, υπουργός, εκλογές, κόμμα, ψηφοφορία, "
        "οικονομία, προϋπολογισμός, ΕΕ, ΝΑΤΟ, διπλωματία, νόμος, ψήφος."
    ),
    "society": (
        "Κοινωνία, άνθρωποι, ζωή, οικογένεια, νέοι, εκπαίδευση, υγεία, εργασία, "
        "δικαιώματα, ισότητα, φτώχεια, μετανάστευση, πολιτισμός, παράδοση."
    ),
    "science": (
        "Επιστήμη, έρευνα, τεχνολογία, εφεύρεση, ανακάλυψη, φυσική, χημεία, βιολογία, "
        "διάστημα, κλίμα, περιβάλλον, ΑΙ, αλγόριθμος, δεδομένα."
    ),
    "technology": (
        "Τεχνολογία, ψηφιακός, AI, τεχνητή νοημοσύνη, software, hardware, startup, "
        "blockchain, crypto, metaverse, app, platform, data, cloud."
    ),
    "entertainment": (
        "Ψυχαγωγία, σινεμά, μουσική, τηλεόραση, σειρά, Netflix, celebrity, αστέρι, "
        "reality show, Survivor, τραγούδι, άλμπουμ, ηθοποιός, σκηνοθέτης."
    ),
    "business": (
        "Επιχείρηση, εταιρεία, startup, CEO, επενδυτής, χρηματοδότηση, κέρδος, "
        "αγορά, στρατηγική, marketing, brand, προϊόν, πελάτης, ανάπτυξη."
    ),
    "education": (
        "Εκπαίδευση, σχολείο, πανεπιστήμιο, μάθηση, ιστορία, αρχαία Ελλάδα, "
        "φιλοσοφία, Σωκράτης, Πλάτωνας, Αριστοτέλης, επανάσταση, πολιτισμός."
    ),
    "lifestyle": (
        "Τρόπος ζωής, ταξίδι, διακοπές, φαγητό, μαγειρική, γυμναστική, "
        "υγεία, ευεξία, μόδα, στυλ, σπίτι, διακόσμηση, vlog, εμπειρία."
    ),
    "gaming": (
        "Gaming, παιχνίδι, gamer, PlayStation, Xbox, PC, esports, streamer, "
        "Twitch, YouTube Gaming, update, patch, multiplayer, ranked, tournament."
    ),
}


_DEFAULT_WHISPER_PROMPT: str = (
    "Ελληνικά, ορθογραφία με τόνους, σωστή στίξη (κόμματα, τελείες, ερωτηματικά), "
    "κεφαλαία, ακρωνύμια, καθαρή αποτύπωση ομιλίας χωρίς παραλείψεις."
)


def _build_whisper_prompt(context_hint: str | None, source_title: str | None) -> str:
    """
    Build a domain-enriched Whisper initial_prompt that anchors the decoder
    to the correct Greek vocabulary before transcription starts.

    The hint is matched against known domain keys (politics, society, science,
    technology). If no domain is matched, a high-quality generic Greek prompt
    is returned.
    """
    parts: list[str] = []
    if source_title:
        # Lead with the video title to anchor proper nouns and names
        parts.append(f"Θέμα: {source_title.strip()}.")

    if context_hint:
        key = context_hint.lower().strip()
        domain_vocab = _DOMAIN_PROMPTS.get(key)
        if domain_vocab:
            parts.append(domain_vocab)

    parts.append(_DEFAULT_WHISPER_PROMPT)
    return " ".join(parts)


def extract_speech_audio(video_path: Path, output_wav: Path) -> Path:
    """
    Extract a 16kHz mono WAV from video_path with voice normalization filters.

    Uses highpass filter (80Hz) to eliminate rumble and dynaudnorm
    (dynamic audio normalizer) to elevate quiet speech and balance levels
    for optimal Whisper speech-to-text accuracy.
    """
    import subprocess
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-af", "highpass=f=80,dynaudnorm=f=150:g=15:m=10.0",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ]
    logger.debug("Extracting normalized speech audio: %s", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0:
        logger.warning(
            "Speech audio pre-extraction failed (code %d): %s — falling back to direct video read.",
            res.returncode,
            res.stderr.decode("utf-8", errors="replace").strip()[:200],
        )
        return video_path
    return output_wav


# ── Core API ───────────────────────────────────────────────────────────────────

def transcribe(
    video_path: Path,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 1,
    progress_cb: Callable[[float, str], None] | None = None,
    context_hint: str | None = None,
    source_title: str | None = None,
) -> list[TranscriptionSegment]:
    """
    Transcribe Greek speech from *video_path* using faster-whisper.

    The language is hard-forced to Greek ("el") to avoid the overhead of
    language detection on short clips and to maximise accuracy.

    Args:
        video_path:    Absolute path to the input video file.
        model_size:    Size of the whisper model to use (e.g. 'base', 'large-v3').
        device:        Device to run on ('cpu' or 'cuda').
        compute_type:  Quantization type (e.g. 'int8', 'float16').
        beam_size:     Beam search size.
        progress_cb:   Optional callback invoked per decoded segment: (fraction, msg).
        context_hint:  Optional domain key ('politics', 'society', 'science', 'technology')
                       used to inject domain-specific vocabulary into the Whisper prompt
                       for improved accuracy on specialised Greek speech.
        source_title:  Optional video title string prepended to the Whisper prompt
                       to anchor proper nouns and channel-specific terminology.

    Returns:
        Ordered list of TranscriptionSegment objects.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError:      If the model fails to load or transcription fails.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Memory Optimization: Check cache first
    try:
        from services.cache_manager import load_cache_pickle, save_cache_pickle
    except ImportError:
        from shorts_engine.services.cache_manager import (
            load_cache_pickle,
            save_cache_pickle,
        )

    cache_key = f"{video_path.name}_{model_size}_{device}_{compute_type}_{beam_size}"
    cached_segments = load_cache_pickle("transcription", cache_key)
    if cached_segments is not None:
        return cached_segments

    logger.info("Loading WhisperModel (size=%s, device=%s)", model_size, device)
    model_kwargs: dict[str, object] = {
        "device": device,
        "compute_type": compute_type,
    }
    if device == "cpu":
        import os
        cores = os.cpu_count() or 4
        # Cap Whisper CPU threads to prevent thermal saturation on ThinkPad/laptop CPUs
        optimal_whisper_threads = max(1, min(4, cores // 2))
        model_kwargs["cpu_threads"] = optimal_whisper_threads
        logger.debug("Configured WhisperModel CPU threads=%d", optimal_whisper_threads)

    try:
        model = WhisperModel(model_size, **model_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load WhisperModel '{model_size}': {exc}"
        ) from exc

    logger.info("Transcribing '%s' (language=el, beam_size=%d)...", video_path.name, beam_size)
    initial_prompt = _build_whisper_prompt(context_hint, source_title)
    logger.debug("Whisper initial_prompt: %s", initial_prompt[:120])

    import tempfile
    temp_wav = Path(tempfile.mktemp(suffix="_speech16k.wav"))
    audio_target = extract_speech_audio(video_path, temp_wav)

    try:
        raw_segments, _info = model.transcribe(
            str(audio_target),
            language="el",
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 400,
                "speech_pad_ms": 400,
            },
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            # temperature=0 forces greedy decoding — most deterministic and accurate
            temperature=0.0,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            hallucination_silence_threshold=2.0,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Transcription failed for '{video_path.name}': {exc}"
        ) from exc
    finally:
        if temp_wav.exists():
            try:
                temp_wav.unlink()
            except OSError as err:
                logger.debug("Failed to delete temp wav %s: %s", temp_wav, err)

    total_duration = getattr(_info, "duration", 0.0) or 0.0

    segments: list[TranscriptionSegment] = []
    for seg in raw_segments:
        if not seg.text.strip():
            continue
        # Extract word-level timing when available
        word_data: list[tuple[float, float, str]] | None = None
        if seg.words:
            word_data = [
                (w.start, w.end, w.word)
                for w in seg.words
                if w.word.strip()
            ]
        segments.append(
            TranscriptionSegment(seg.start, seg.end, seg.text, word_data)
        )

        # Stream real-time progress per decoded segment
        if total_duration > 0:
            pct = min(1.0, seg.end / total_duration)
            cur_m, cur_s = int(seg.end // 60), int(seg.end % 60)
            tot_m, tot_s = int(total_duration // 60), int(total_duration % 60)
            status_msg = (
                f"Transcribing audio: {cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d} ({int(pct * 100)}%)"
            )
            if progress_cb:
                progress_cb(pct, status_msg)
            logger.info(
                "[%02d:%02d / %02d:%02d] (%2d%%): %s",
                cur_m, cur_s, tot_m, tot_s, int(pct * 100), seg.text.strip()[:60],
            )

    logger.info("Transcription complete — %d segments extracted.", len(segments))
    
    # Save to cache
    save_cache_pickle("transcription", cache_key, segments)

    # Reclaim model memory on unified memory laptop architectures (ThinkPad)
    try:
        del model
        import gc
        gc.collect()
    except (RuntimeError, AttributeError) as exc:
        logger.debug("Model memory reclamation failed (non-fatal): %s", exc)

    return segments


def full_transcript_text(segments: list[TranscriptionSegment]) -> str:
    """
    Concatenate all segment texts into a single prose string.

    Used as input to the SEO generator and B-roll keyword extraction.
    """
    return " ".join(seg.text for seg in segments)


# ── ASS Subtitle Generation ────────────────────────────────────────────────────

def _seconds_to_ass_time(seconds: float) -> str:
    """
    Convert a float timestamp (seconds) to ASS time format: H:MM:SS.cc

    ASS centiseconds are two digits (hundredths of a second).
    Uses integer division on centiseconds to eliminate floating-point rounding
    and prevent three-digit centisecond overflows (e.g. 59.100).
    """
    total_cs = max(0, round(seconds * 100))
    hours, rem = divmod(total_cs, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """
    Escape characters that have special meaning in ASS dialogue lines.

    The ASS spec treats '{' as the start of an override tag block, and
    '\\n' / '\\N' as soft/hard line breaks. We escape bare braces to prevent
    accidental tag injection from transcript text.
    """
    # Replace curly braces with their ASS literal equivalents
    text = text.replace("{", r"\{").replace("}", r"\}")
    return text


# Mapping from position names to ASS MarginV values.
# MarginV is the distance in pixels from the bottom of the 1920px frame.
_SUBTITLE_MARGIN_V: dict[str, int] = {
    "lower_third": 540,   # ~28% up from bottom (default for Shorts)
    "center": 960,        # True vertical centre of the frame
    "top": 1600,          # Near the top, leaving room for platform UI
}


def _build_style_line(base: str, margin_v: int) -> str:
    """
    Clone a standard ASS style line with a custom MarginV.

    The MarginV field is the 22nd comma-separated token (0-indexed: 21).
    """
    parts = base.split(",")
    # Field index 21 = MarginV in the Format order defined in ASS_HEADER_TEMPLATE
    if len(parts) > 21:
        parts[21] = str(margin_v)
    return ",".join(parts)


def segments_to_ass(
    segments: list[TranscriptionSegment],
    style_line: str | None = None,
    highlight_style_line: str | None = None,
    primary_keyword: str | None = None,
    subtitle_position: str = "lower_third",
    subtitle_mode: str = "dynamic",
) -> str:
    """
    Generate an ASS subtitle payload from a list of transcription segments.
    Supports animated dynamic phrase karaoke, natural phrase chunks, or 1-word pop.

    Args:
        segments:            Ordered transcript segments.
        style_line:          Optional raw ASS style string override.
        highlight_style_line: Optional highlight style override.
        primary_keyword:     Keyword to highlight in yellow.
        subtitle_position:   "lower_third" | "center" | "top".
        subtitle_mode:       "dynamic" (fluid 2-3 words active highlight) | "phrase" | "word".
    """
    margin_v = _SUBTITLE_MARGIN_V.get(subtitle_position, _SUBTITLE_MARGIN_V["lower_third"])

    effective_style = _build_style_line(
        style_line if style_line is not None else ASS_STYLE_LINE,
        margin_v,
    )
    effective_highlight = _build_style_line(
        highlight_style_line if highlight_style_line is not None else ASS_HIGHLIGHT_STYLE_LINE,
        margin_v,
    )

    clean_keyword = ""
    if primary_keyword:
        clean_keyword = primary_keyword.translate(str.maketrans('', '', string.punctuation)).lower().strip()

    dialogue_lines: list[str] = []
    
    # Flatten, sort, and sanitize all words with strictly monotonic start times
    raw_words: list[tuple[float, float, str]] = []
    for seg in segments:
        if seg.words:
            raw_words.extend(seg.words)

    raw_words.sort(key=lambda w: (w[0], w[1]))
    all_words: list[tuple[float, float, str]] = []
    for w_s, w_e, w_t in raw_words:
        clean_t = w_t.strip()
        if not clean_t:
            continue
        w_start_val = max(0.0, float(w_s))
        w_end_val = max(w_start_val + 0.05, float(w_e))
        if all_words:
            prev_s = all_words[-1][0]
            if w_start_val <= prev_s:
                w_start_val = prev_s + 0.05
                w_end_val = max(w_end_val, w_start_val + 0.05)
        all_words.append((w_start_val, w_end_val, clean_t))

    min_word_duration = 0.22
    gap_bridge_threshold = 0.35

    # If no words available, fallback to segments
    if not all_words:
        sorted_segs = sorted(segments, key=lambda s: s.start)
        for k, seg in enumerate(sorted_segs):
            s_start = max(0.0, float(seg.start))
            s_end = max(s_start + 0.05, float(seg.end))
            if k + 1 < len(sorted_segs):
                next_start = max(0.0, float(sorted_segs[k + 1].start))
                if next_start > s_start:
                    s_end = min(s_end, next_start)
                else:
                    s_end = s_start + 0.05
            start = _seconds_to_ass_time(s_start)
            end = _seconds_to_ass_time(s_end)
            text_field = _escape_ass_text(seg.text)
            dialogue_lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text_field}")
    elif subtitle_mode == "dynamic":
        # Group words into 2-3 word natural phrases with real-time active-word karaoke highlighting
        idx = 0
        while idx < len(all_words):
            chunk = [all_words[idx]]
            idx += 1
            while idx < len(all_words) and len(chunk) < 3:
                curr_w = all_words[idx]
                prev_w = chunk[-1]
                # Break on long pauses between words
                if curr_w[0] - prev_w[1] > 0.40:
                    break
                combined_len = sum(len(w[2]) for w in chunk) + len(curr_w[2]) + len(chunk)
                if combined_len > 22:
                    break
                chunk.append(curr_w)
                idx += 1

            # Format each active word inside the chunk for seamless fluid highlighting
            for w_i, active_w in enumerate(chunk):
                w_start = active_w[0]
                if w_i + 1 < len(chunk):
                    w_end = chunk[w_i + 1][0]
                else:
                    w_end = max(active_w[1], w_start + min_word_duration)
                    if idx < len(all_words):
                        next_chunk_start = all_words[idx][0]
                        if next_chunk_start > w_start:
                            if w_end >= next_chunk_start or (next_chunk_start - w_end < gap_bridge_threshold):
                                w_end = next_chunk_start
                            else:
                                w_end = min(w_end + 0.15, next_chunk_start)
                        else:
                            w_end = w_start + 0.05
                    else:
                        w_end = w_end + 0.15

                if w_end <= w_start:
                    w_end = w_start + 0.05

                t_start = _seconds_to_ass_time(w_start)
                t_end = _seconds_to_ass_time(w_end)

                words_formatted: list[str] = []
                for j, w in enumerate(chunk):
                    w_safe = _escape_ass_text(w[2])
                    if j == w_i:
                        # Active spoken word in bright yellow with subtle dynamic scale pop
                        words_formatted.append(rf"{{\c&H00FFFF&\fscx106\fscy106}}{w_safe}{{\r}}")
                    else:
                        words_formatted.append(rf"{{\c&H00FFFFFF&}}{w_safe}{{\r}}")

                phrase_text = " ".join(words_formatted)
                dialogue_lines.append(f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{phrase_text}")
    elif subtitle_mode == "phrase":
        # Group words into 2-3 word natural phrases
        idx = 0
        while idx < len(all_words):
            chunk = [all_words[idx]]
            idx += 1
            while idx < len(all_words) and len(chunk) < 3:
                curr_w = all_words[idx]
                prev_w = chunk[-1]
                # Break on long pauses between words
                if curr_w[0] - prev_w[1] > 0.45:
                    break
                combined_len = sum(len(w[2]) for w in chunk) + len(curr_w[2]) + len(chunk)
                if combined_len > 24:
                    break
                chunk.append(curr_w)
                idx += 1
            
            p_start = chunk[0][0]
            p_end = max(chunk[-1][1], p_start + 0.40)
            if idx < len(all_words):
                next_start = all_words[idx][0]
                if next_start > p_start:
                    if p_end >= next_start:
                        p_end = next_start
                    else:
                        gap = next_start - p_end
                        if gap < gap_bridge_threshold:
                            p_end = next_start
                        else:
                            p_end = min(p_end + 0.15, next_start)
                else:
                    p_end = p_start + 0.05
            else:
                p_end = p_end + 0.20

            t_start = _seconds_to_ass_time(p_start)
            t_end = _seconds_to_ass_time(p_end)

            words_formatted: list[str] = []
            for w in chunk:
                w_safe = _escape_ass_text(w[2])
                clean_w = w[2].translate(str.maketrans('', '', string.punctuation)).lower().strip()
                if clean_keyword and clean_w and (clean_w == clean_keyword or clean_w in clean_keyword.split()):
                    words_formatted.append(rf"{{\c&H00FFFF&}}{w_safe}{{\c&H00FFFFFF&}}")
                else:
                    words_formatted.append(w_safe)

            pop = r"{\fscx85\fscy85\t(0,50,\fscx108\fscy108)\t(50,130,\fscx100\fscy100)}"
            phrase_text = " ".join(words_formatted)
            dialogue_lines.append(f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{pop}{phrase_text}")
    else:
        for i, (w_start, w_end, w_text) in enumerate(all_words):
            display_end = max(w_start + min_word_duration, w_end)
            
            # Bridge short gaps to next word without exceeding next word's start time
            if i + 1 < len(all_words):
                next_start = all_words[i + 1][0]
                if next_start > w_start:
                    if display_end >= next_start:
                        display_end = next_start
                    else:
                        gap = next_start - display_end
                        if gap < gap_bridge_threshold:
                            display_end = next_start
                        else:
                            display_end = min(display_end + 0.15, next_start)
                else:
                    display_end = w_start + 0.05
            else:
                display_end = display_end + 0.15

            t_start = _seconds_to_ass_time(w_start)
            t_end = _seconds_to_ass_time(display_end)
            
            safe = _escape_ass_text(w_text)
            
            # Animation: Pop in from 80% to 115%, then settle at 100%
            pop = r"{\fscx80\fscy80\t(0,40,\fscx115\fscy115)\t(40,120,\fscx100\fscy100)}"
            
            # Color: Highlight the primary keyword in Yellow, otherwise stay White
            color = ""
            if clean_keyword:
                clean_word = w_text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
                if clean_word and (clean_word == clean_keyword or clean_word in clean_keyword.split()):
                    color = r"{\c&H00FFFF&}"  # Yellow BGR
                    
            text_field = f"{pop}{color}{safe}"
            dialogue_lines.append(f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{text_field}")

    return ASS_HEADER_TEMPLATE.format(
        style_line=effective_style,
        highlight_style_line=effective_highlight,
        dialogue_lines="\n".join(dialogue_lines),
    )


def write_ass_file(
    segments: list[TranscriptionSegment],
    output_path: Path,
    style_line: str | None = None,
    highlight_style_line: str | None = None,
    primary_keyword: str | None = None,
    subtitle_position: str = "lower_third",
    subtitle_mode: str = "dynamic",
) -> Path:
    """
    Generate and write an ASS subtitle file for the given segments.

    Args:
        segments:             Ordered list of TranscriptionSegment objects.
        output_path:          Destination path for the .ass file.
        style_line:           Optional style override (see segments_to_ass).
        highlight_style_line: Optional highlight style override.
        primary_keyword:      Optional keyword to statically highlight.
        subtitle_position:    Vertical placement: "lower_third" | "center" | "top".
        subtitle_mode:        "dynamic" | "phrase" | "word".

    Returns:
        The resolved, written output_path.

    Raises:
        OSError: If the file cannot be written.
    """
    ass_content = segments_to_ass(
        segments,
        style_line=style_line,
        highlight_style_line=highlight_style_line,
        primary_keyword=primary_keyword,
        subtitle_position=subtitle_position,
        subtitle_mode=subtitle_mode,
    )

    # ASS files must be UTF-8 encoded to preserve Greek glyphs
    output_path.write_text(ass_content, encoding="utf-8")
    logger.info("ASS subtitle file written to '%s'.", output_path)
    return output_path
