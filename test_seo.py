import os
from shorts_engine.services.seo_generator import generate_seo
from shorts_engine.config import Settings

# Clear cache to force generation
import shutil
if os.path.exists(".cache/seo_metadata"):
    shutil.rmtree(".cache/seo_metadata")

settings = Settings()
test_transcript = "Καλησπέρα! Σήμερα θα μιλήσουμε για την Ελληνική Λύση και τον Βελόπουλο. Ποιος τον τρέμει; Η κυβέρνηση Μητσοτάκη έχει μεγάλο πρόβλημα."

seo = generate_seo(
    transcript_text=test_transcript,
    api_key=settings.gemini_api_key,
    source_title="Test Video",
    brand_voice="Politics/Economy"
)
print("Title:", seo.title)
print("Curiosity:", repr(seo.curiosity_title))
print("Authority:", repr(seo.authority_title))
print("Contrarian:", repr(seo.contrarian_title))
