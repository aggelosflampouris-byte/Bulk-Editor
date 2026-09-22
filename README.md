# 🎬 Batch Greek Shorts Processing Engine

> Production-grade desktop web app for automated, high-retention 9:16 vertical video shorts generation tailored for Greek-language content.

[![GitHub Release](https://img.shields.io/github/v/release/aggelosflampouris-byte/Bulk-Editor?color=blue&label=Latest%20Release)](https://github.com/aggelosflampouris-byte/Bulk-Editor/releases/latest)
[![Download Zip](https://img.shields.io/badge/Download-Bulk--Editor--Release.zip-brightgreen?logo=github)](https://github.com/aggelosflampouris-byte/Bulk-Editor/releases/latest/download/Bulk-Editor-Release.zip)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)

---

## 📦 Direct Download (Latest Release)

The fastest and easiest way to use the program without cloning git:

👉 **[Download Latest Release (Bulk-Editor-Release.zip)](https://github.com/aggelosflampouris-byte/Bulk-Editor/releases/latest/download/Bulk-Editor-Release.zip)** *(~4.4 MB, self-contained)*

> **Alternative Download Locations:**
> - [Browse GitHub Releases Page](https://github.com/aggelosflampouris-byte/Bulk-Editor/releases)
> - Or click the green **Code → Download ZIP** button at the top of this repository.

---

## ⚡ Quickstart — 1-Click Launch

1. **Download and Extract:** Unzip `Bulk-Editor-Release.zip` to any folder on your computer.
2. **Run the launcher:**
   - 🪟 **Windows:** Simply double-click `run.bat` in the extracted folder.
   - 🐧 **Linux & 🍎 macOS:** Open terminal in the folder and run:
     ```bash
     chmod +x run.sh && ./run.sh
     ```
3. **That's it!** The launcher automatically verifies Python 3.10+, installs dependencies, downloads FFmpeg into `./bin/` if missing, initializes your `.env` template, and opens the app in your browser.

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
- **Hardware Acceleration & ThinkPad Efficiency:** Automatic GPU acceleration via Intel Quick Sync (`h264_qsv` for Intel/ThinkPad), AMD AMF (`h264_amf` for Radeon GPUs), and Linux VA-API (`h264_vaapi`). Built-in CPU thread capping prevents thermal throttling and fan noise on laptops.
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
│   │   ├── hw_encoder.py      # Hardware encoder detection (QSV/AMF/VAAPI)
│   │   ├── niche_templates.py # Niche presets & brand profiles
│   │   ├── ocr_engine.py      # Gemini Vision frame text extraction
│   │   ├── seo_generator.py   # Greek title, tags & description generator
│   │   ├── transcriber.py     # Whisper transcription & subtitle generation
│   │   ├── video_engine.py    # FFmpeg cropping, audio mixing & burning
│   │   ├── vfx_engine.py      # Color grading & scene analysis
│   │   └── youtube_uploader.py# YouTube Data API upload handling
│   ├── ui/                    # Streamlit modular tabs and components
│   └── tests/                 # Comprehensive test suite (44 unit tests)
```

---

## 🧪 Running Tests

To run the full unit test suite:
```bash
cd shorts_engine
./.venv/bin/pytest
```
*(Or from the repository root: `pytest`)*
