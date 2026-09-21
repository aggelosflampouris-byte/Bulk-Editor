# 🎬 Batch Greek Shorts Processing Engine

> Production-grade desktop web app for automated, high-retention 9:16 vertical video shorts generation tailored for Greek-language content.

---

## ⚡ Quickstart — 1-Click Launch

No complex setup needed. The engine includes zero-friction launchers that automatically verify Python, set up the virtual environment, install all dependencies, configure FFmpeg, and launch the application directly in your web browser.

### 🪟 Windows
Simply **double-click** `run.bat` in this folder.
*(Or open Command Prompt / PowerShell here and run `.\run.bat`)*

### 🐧 Linux & 🍎 macOS
Open Terminal in this directory and execute:
```bash
chmod +x run.sh
./run.sh
```

---

## 🔑 API Keys Configuration

On the first launch, a `shorts_engine/.env` file is automatically created from `shorts_engine/.env.example`. 

Open `shorts_engine/.env` in any text editor and fill in your keys (or enter them directly in the Streamlit web sidebar during your session):

```env
# Google Gemini API Key — free at https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Pexels API Key — free at https://www.pexels.com/api/
PEXELS_API_KEY=your_pexels_api_key_here

# (Optional) Primary YouTube Channel for Niche Explorer
CONNECTED_CHANNEL=https://www.youtube.com/@YourChannel
```

---

## 🌟 Core Features

- **9 Niche Presets:** Instant configuration tailored for Politics, Society, Science, Technology, Entertainment, Business, Education, Lifestyle, and Gaming (plus Custom mode). Each preset fine-tunes duration, brand voice, subtitle positioning, VFX color grading, and Whisper domain vocabularies.
- **Smart Pose & Face Centering:** Automatically detects and tracks speakers using pose keypoints (`yolov8n-pose.pt`) to keep faces centered in 9:16 vertical frames.
- **Greek-Specialized Transcription:** Whisper with forced Greek vocabulary prompting (`el`) and acronym post-processing (e.g. ΔΕΔΔΗΕ, ΕΣΥ, ΑΣΕΠ, ΑΑΔΕ, ΟΠΕΚΕΠΕ).
- **Gemini Vision OCR:** Reads on-screen banners, chyrons, and graphics from source video frames to provide rich semantic context for clip selection.
- **Dynamic Karaoke Subtitles:** Generates styled `.ass` subtitles with active word highlighting, safe for Greek typography.
- **B-Roll & Visual Enhancements:** Automated keyword search via Pexels API and cinematic VFX filters (contrast boost, warm cinematic, high impact).
- **Automated YouTube Integration:** Inspect competing channels, calculate view velocities, and schedule uploads directly with metadata and tags.

---

## 📂 Project Structure

```text
Bulk-Editor/
├── run.bat                    # 1-Click launcher for Windows
├── run.sh                     # 1-Click launcher for Linux/macOS
├── pytest.ini                 # Pytest test discovery configuration
├── README.md                  # Project documentation & quickstart
├── shorts_engine/
│   ├── app.py                 # Streamlit web application interface
│   ├── pipeline.py            # End-to-end batch processing pipeline
│   ├── config.py              # Configuration & settings management
│   ├── requirements.txt       # Python package dependencies
│   ├── .env.example           # Template for environment variables
│   ├── run.bat                # Direct engine launcher (Windows)
│   ├── run.sh                 # Direct engine launcher (Linux/macOS)
│   ├── services/              # Core business logic & AI models
│   │   ├── clip_selector.py   # AI hook & clip detection
│   │   ├── face_tracker.py    # Speaker tracking & centering
│   │   ├── niche_templates.py # Niche presets & brand profiles
│   │   ├── ocr_engine.py      # Gemini Vision frame text extraction
│   │   ├── seo_generator.py   # Greek title, tags & description generator
│   │   ├── transcriber.py     # Whisper transcription & subtitle generation
│   │   ├── video_engine.py    # FFmpeg cropping, audio mixing & burning
│   │   ├── vfx_engine.py      # Color grading & scene analysis
│   │   └── youtube_uploader.py# YouTube Data API upload handling
│   ├── ui/                    # Streamlit modular tabs and components
│   └── tests/                 # Comprehensive test suite (79 unit tests)
```

---

## 🧪 Running Tests

To run the full unit test suite:
```bash
cd shorts_engine
./.venv/bin/pytest
```
*(Or from the repository root: `pytest`)*
