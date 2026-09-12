import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from config import Settings
from services.seo_generator import generate_seo

settings = Settings()
api_key = settings.gemini_api_key

text = "Αυτή είναι μια δοκιμή. Το YouTube Shorts SEO τώρα έχει criticism loop."
seo = generate_seo(text, api_key, "Test Video", brand_voice="Aggressive marketer")

print(f"Title: {seo.title}")
print(f"Curiosity: {seo.curiosity_title}")
print(f"Authority: {seo.authority_title}")
print(f"Contrarian: {seo.contrarian_title}")
print(f"Comment: {seo.pinned_comment}")
