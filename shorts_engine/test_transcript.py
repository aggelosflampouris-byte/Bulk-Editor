import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from config import Settings
from services.seo_generator import correct_transcript_greek
from services.transcriber import TranscriptionSegment

settings = Settings()
api_key = settings.gemini_api_key

segments = [
    TranscriptionSegment(start=0.0, end=2.0, text="Καλημέρες ημέρες σε όλους", words=[]),
    TranscriptionSegment(start=2.0, end=4.0, text="Σήμερα το πρωί, μαζί μες τελέχη", words=[]),
    TranscriptionSegment(start=4.0, end=6.0, text="και λέμε ότι αυτό δεν είναι σωστό", words=[])
]

print(f"Original:")
for s in segments: print(s.text)

corrected = correct_transcript_greek(segments, api_key)

print(f"\nCorrected:")
for s in corrected: print(s.text)
