"""
services/logic_guardrail.py — Logical coherence and editorial "make sense" guardrails.

Responsibilities:
  1. Programmatic text & boundary validation: Ensure clips begin with a clean,
     complete sentence and never cut off mid-word, mid-breath, or on dangling
     conjunctions (e.g. "και", "ότι", "αλλά", "να", "που").
  2. Gemini AI Logical Coherence Verification: Evaluate whether a Short's narrative
     (standalone speaker clip, full AI script, or hybrid Part 1 + Part 2 combo)
     makes 100% logical sense, has a clear premise, factual progression, and
     satisfying conclusion without confusing non-sequiturs or disjointed jumps.
  3. Script refinement: If a generated script has logical gaps, repair it to ensure
     airtight narrative flow before voiceover synthesis.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

try:
    from services.seo_generator import _call_gemini_with_fallback
    from services.transcriber import TranscriptionSegment
except ImportError:
    from shorts_engine.services.seo_generator import _call_gemini_with_fallback
    from shorts_engine.services.transcriber import TranscriptionSegment


def clean_json_markdown(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()

logger = logging.getLogger(__name__)

# Dangling Greek words that must NEVER appear at the end of a clip or sentence cut
DANGLING_END_WORDS: set[str] = {
    "και", "κι", "ότι", "πως", "αλλά", "όμως", "όταν", "γιατί", "επειδή",
    "να", "που", "σε", "με", "για", "ή", "είτε", "άρα", "λοιπόν", "καθώς",
    "αν", "εάν", "ώστε", "ενώ", "αφού", "πριν", "χωρίς", "προς", "από",
    "τη", "την", "το", "τον", "τους", "της", "των", "τα", "ο", "η", "οι",
}

# Weak, trailing, or stuttering starts that ruin a hook
DANGLING_START_WORDS: set[str] = {
    "και", "κι", "λοιπόν", "εεε", "ααα", "μμμ", "ναι", "όπως", "δηλαδή",
}


@dataclass
class CoherenceAssessment:
    makes_sense: bool
    score: int  # 1 to 10
    reason: str
    repaired_script: str | None = None


@dataclass
class RetentionAudit:
    """
    Inflowave-inspired pre-publish Short retention & virality audit.
    Evaluates speech cadence, intro dead air, and curiosity hooks.
    """
    score: int  # 0 to 100
    grade: str  # "S", "A", "B", "C", "D"
    hook_speed_score: int  # 0 to 100
    pacing_cadence_score: int  # 0 to 100
    curiosity_gap_score: int  # 0 to 100
    intro_silence_seconds: float
    words_per_second: float
    actionable_recommendations: list[str]


# High-converting Greek curiosity, investigation, and contrast trigger words
GREEK_POWER_KEYWORDS: list[str] = [
    "αποκάλυψη", "σκάνδαλο", "αλήθεια", "σοκάρει", "ανατροπή", "κίνδυνος",
    "μυστικό", "έρευνα", "στοιχεία", "αποδεικνύει", "ψέμα", "κρυφό", "απίστευτο",
    "γιατί", "πώς", "ποιος", "ποια", "πόσα", "πότε", "πραγματικότητα", "δόλος",
    "διαφθορά", "χρήματα", "δισεκατομμύρια", "εκατομμύρια", "φόροι", "ακρίβεια",
]


def audit_retention_signals(
    segments: list[TranscriptionSegment] | None,
    text: str,
    duration: float,
    hook_summary: str = "",
    title: str = "",
) -> RetentionAudit:
    """
    Audit retention and virality signals of a Short candidate based on Inflowave principles:
      1. Hook Speed: Dead air before the first spoken syllable (< 0.35s is optimal).
      2. Speech Cadence: Speech velocity in words/second (2.8–4.2 wps is optimal for retention).
      3. Curiosity Gap: Presence of investigative hooks, rhetorical questions, numbers, and contrast.
    """
    safe_dur = max(duration, 1.0)
    recommendations: list[str] = []

    # 1. Intro silence / hook speed
    intro_silence = 0.0
    if segments and len(segments) > 0:
        first_seg = segments[0]
        if first_seg.words and len(first_seg.words) > 0:
            intro_silence = max(0.0, float(first_seg.words[0][0]))
        else:
            intro_silence = max(0.0, float(first_seg.start))

    if intro_silence <= 0.35:
        hook_speed = 100
    elif intro_silence <= 0.70:
        hook_speed = 90
        recommendations.append(f"Ελαφρύ κενό έναρξης ({intro_silence:.2f}s). Μειώστε το σε <0.3s για ακαριαίο hook.")
    elif intro_silence <= 1.10:
        hook_speed = 75
        recommendations.append(f"Νεκρός χρόνος ({intro_silence:.2f}s) στην αρχή. Κόψτε το άνοιγμα απευθείας στην πρώτη λέξη.")
    elif intro_silence <= 1.60:
        hook_speed = 55
        recommendations.append(f"Σημαντική καθυστέρηση ομιλίας ({intro_silence:.2f}s). Κίνδυνος άμεσου scroll-away.")
    else:
        hook_speed = max(15, 100 - int(intro_silence * 45))
        recommendations.append(f"Μεγάλη αρχική σιωπή ({intro_silence:.2f}s). Απαιτείται άμεση περικοπή (trim).")

    # 2. Words per second (speech cadence)
    words = re.findall(r"\b[\wΆ-ώ]+\b", text or "", re.UNICODE)
    word_count = len(words)
    speech_active_time = max(1.0, safe_dur - intro_silence)
    wps = round(word_count / speech_active_time, 2)

    if 2.8 <= wps <= 4.2:
        cadence_score = 100
    elif 2.3 <= wps < 2.8:
        cadence_score = 85
        recommendations.append(f"Ρυθμός ομιλίας ελαφρώς χαλαρός ({wps:.1f} λ/δευτ.). Προτείνεται ελαφριά επιτάχυνση 1.08x.")
    elif 4.2 < wps <= 4.9:
        cadence_score = 88
    elif wps < 2.3:
        cadence_score = max(25, int(wps * 35))
        recommendations.append(f"Αργός ρυθμός ({wps:.1f} λ/δευτ.). Αφαιρέστε ενδιάμεσες παύσεις για διατήρηση προσοχής.")
    else:  # wps > 4.9
        cadence_score = max(40, 100 - int((wps - 4.9) * 25))
        recommendations.append(f"Πολύ γρήγορη εκφορά ({wps:.1f} λ/δευτ.). Εξασφαλίστε ευανάγνωστους υπότιτλους 1-2 λέξεων.")

    # 3. Curiosity Gap & Hook Potency
    curiosity_score = 55
    combined_eval_text = f"{title} {hook_summary} {text}".lower()

    # Questions trigger an open loop in viewer's mind
    if "?" in combined_eval_text or ";" in combined_eval_text:
        curiosity_score += 15

    # Concrete numbers or percentages anchor factual credibility
    if re.search(r"(\d+%|\d+\s*(?:ευρώ|€|δις|εκατ|χιλιάδες|%))", combined_eval_text):
        curiosity_score += 15

    # Power keywords
    power_matches = sum(1 for kw in GREEK_POWER_KEYWORDS if kw in combined_eval_text)
    curiosity_score += min(20, power_matches * 5)
    curiosity_score = min(100, max(20, curiosity_score))

    if curiosity_score < 70:
        recommendations.append("Προσθέστε συγκεκριμένα νούμερα ή ανοιχτό ερώτημα (π.χ. 'Γιατί...') στο άνοιγμα.")

    # Overall weighted composite score
    overall_score = round(hook_speed * 0.35 + cadence_score * 0.35 + curiosity_score * 0.30)
    overall_score = max(10, min(100, overall_score))

    if overall_score >= 90:
        grade = "S"
        recommendations.insert(0, "🔥 Κορυφαία δυναμική Hook & Retention (S-Tier).")
    elif overall_score >= 80:
        grade = "A"
        recommendations.insert(0, "✓ Υψηλή δυναμική διατήρησης προσοχής (A-Tier).")
    elif overall_score >= 70:
        grade = "B"
    elif overall_score >= 60:
        grade = "C"
    else:
        grade = "D"

    return RetentionAudit(
        score=overall_score,
        grade=grade,
        hook_speed_score=hook_speed,
        pacing_cadence_score=cadence_score,
        curiosity_gap_score=curiosity_score,
        intro_silence_seconds=round(intro_silence, 2),
        words_per_second=wps,
        actionable_recommendations=recommendations,
    )



def validate_text_boundaries(text: str) -> tuple[bool, str]:
    """
    Check if a transcript or script snippet has complete sentence boundaries.
    Rejects text that cuts off mid-thought or ends on dangling conjunctions.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return False, "Empty text"

    words = re.findall(r"\b[\wΆ-ώ]+\b", cleaned, re.UNICODE)
    if len(words) < 5:
        return False, f"Too short to convey a complete thought ({len(words)} words)"

    # Check last word
    last_word = words[-1].lower()
    if last_word in DANGLING_END_WORDS:
        return False, f"Ends abruptly on dangling word '{last_word}'"

    # Check start word
    first_word = words[0].lower()
    if first_word in DANGLING_START_WORDS and len(words) > 1:
        # Warning/minor penalty but allowable if followed by complete thought
        pass

    # Must end with valid punctuation (. ! ? ; …)
    if cleaned[-1] not in (".", "!", "?", ";", "…") and not re.search(r"[\.!\?;…]$", cleaned):
        return False, "Missing terminal sentence punctuation"

    return True, "Valid boundaries"


_COHERENCE_GUARDRAIL_PROMPT = """\
You are an expert editorial director and fact-checking logic evaluator for Greek news & investigative YouTube Shorts (@DianismaNews).

YOUR TASK:
Analyze the following Short content for LOGICAL COHERENCE and narrative continuity.
A Short "makes sense" if:
1. It is understandable to a general Greek audience who has NOT watched any prior footage.
2. The speaker's statement and/or AI narration flow logically from premise -> fact/context -> punchline/conclusion.
3. It does NOT cut off mid-thought, contradict itself, or contain nonsense, disjointed jumps, or gibberish.
4. Part 2 (if present) directly addresses and complements Part 1, providing closure.

CONTENT TO EVALUATE:
Topic / Source Title: {topic_title}
Speaker Clip (Part 1): "{part1_text}"
AI Breakdown / Conclusion (Part 2, if applicable): "{part2_text}"

EVALUATION CRITERIA:
- makes_sense: boolean (true if the Short tells a clear, self-contained, sensible story; false if broken, confusing, or illogical)
- score: integer from 1 to 10 (10 = perfect logical clarity, < 7 = confusing or incomplete)
- reason: brief 1-sentence explanation of your evaluation
- repaired_script: if score < 8, provide a perfectly coherent, natural Greek narration script (30-45 words) that makes 100% sense with Part 1 and provides a punchy conclusion. Otherwise, return null.

Return ONLY a valid JSON object matching this schema:
{{
  "makes_sense": true,
  "score": 9,
  "reason": "Clear thesis and logical conclusion with hard numbers.",
  "repaired_script": null
}}
"""


def evaluate_logical_coherence(
    topic_title: str,
    part1_text: str,
    part2_text: str = "",
    gemini_api_key: str = "",
) -> CoherenceAssessment:
    """
    Run an LLM-powered 'make sense' guardrail check on a candidate Short script.
    """
    # 1. Quick rule-based boundary checks
    full_text = f"{part1_text} {part2_text}".strip()
    valid_bounds, bound_msg = validate_text_boundaries(full_text)
    if not valid_bounds:
        logger.warning("[Logic Guardrail] Boundary pre-check warning: %s", bound_msg)

    if not gemini_api_key or not gemini_api_key.strip():
        # Fallback to rule-based evaluation if no API key
        score = 8 if valid_bounds else 5
        return CoherenceAssessment(
            makes_sense=valid_bounds,
            score=score,
            reason=bound_msg,
            repaired_script=None,
        )

    try:
        client = genai.Client(api_key=gemini_api_key)
        prompt = _COHERENCE_GUARDRAIL_PROMPT.format(
            topic_title=topic_title or "Ελληνική Επικαιρότητα",
            part1_text=part1_text.strip() or "(Κανένα κείμενο ομιλητή)",
            part2_text=part2_text.strip() or "(Κανένα κείμενο Part 2)",
        )

        config = genai_types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        )
        raw_text = _call_gemini_with_fallback(client=client, contents=prompt, config=config)

        raw_json = clean_json_markdown(raw_text or "{}")
        data = json.loads(raw_json)

        makes_sense = bool(data.get("makes_sense", True))
        score = int(data.get("score", 8))
        reason = str(data.get("reason", "Assessment completed."))
        repaired = data.get("repaired_script")
        if repaired and isinstance(repaired, str) and repaired.strip():
            repaired_script = repaired.strip()
        else:
            repaired_script = None

        logger.info(
            "[Logic Guardrail] Assessment result: makes_sense=%s, score=%d/10 ('%s')",
            makes_sense,
            score,
            reason,
        )

        return CoherenceAssessment(
            makes_sense=makes_sense,
            score=score,
            reason=reason,
            repaired_script=repaired_script,
        )

    except (genai_errors.APIError, RuntimeError, ValueError, KeyError, OSError, TypeError) as exc:
        logger.warning("[Logic Guardrail] LLM evaluation encountered error (%s) — using rule check.", exc)
        return CoherenceAssessment(
            makes_sense=valid_bounds,
            score=7 if valid_bounds else 5,
            reason=f"Rule fallback: {bound_msg}",
            repaired_script=None,
        )


def enforce_clip_coherence(
    segments: list[TranscriptionSegment],
    clip_start: float,
    clip_end: float,
) -> tuple[float, float, str]:
    """
    Adjust clip start and end boundaries to ensure the selected snippet
    starts and ends on complete words and valid sentences without dangling cuts.
    """
    if not segments:
        return clip_start, clip_end, ""

    # Filter segments within [clip_start, clip_end]
    in_range = [s for s in segments if s.end > clip_start and s.start < clip_end]
    if not in_range:
        return clip_start, clip_end, ""

    words: list[tuple[float, float, str]] = []
    for s in in_range:
        if s.words:
            words.extend(s.words)
        else:
            w_list = s.text.split()
            dur = max(s.end - s.start, 0.1)
            for i, w in enumerate(w_list):
                words.append((s.start + (i / max(1, len(w_list))) * dur, s.start + ((i + 1) / max(1, len(w_list))) * dur, w))

    words_in_cut = [w for w in words if w[0] >= clip_start - 0.2 and w[1] <= clip_end + 0.2]
    if not words_in_cut:
        return clip_start, clip_end, " ".join(s.text for s in in_range)

    # Check if last word is dangling
    last_w = words_in_cut[-1][2].lower().strip(".,!?;:…")
    adj_end = clip_end
    if last_w in DANGLING_END_WORDS and len(words_in_cut) > 3:
        # Snap back to previous word end
        adj_end = words_in_cut[-2][1]
        words_in_cut = words_in_cut[:-1]

    text = " ".join(w[2] for w in words_in_cut)
    return clip_start, adj_end, text
