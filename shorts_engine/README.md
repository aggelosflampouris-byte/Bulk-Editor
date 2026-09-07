# 🎬 Batch Greek Shorts Processing Engine

> Zero-config, production-grade desktop web app for bulk-converting raw Greek-speech video clips into standardised 9:16 YouTube Shorts.

---

## AI Model Stack & Roles

| Model | Role | Exact Task |
|---|---|---|
| **faster-whisper-large-v3** | Audio transcription & word-level timestamping | Converts Greek speech into precise text with millisecond timestamps to pinpoint cut locations and power animated karaoke subtitles. |
| **Qwen 2.5-32B-Instruct** (or **72B**) | Content analysis, highlight scoring & Greek SEO | Reads the Greek timestamped transcript, identifies high-retention/viral 30–60s hooks, and generates Greek search titles, descriptions, and hashtags. |
| **YOLOv8n-face** (or **YOLOv11n-face**) | Active speaker tracking & dynamic 9:16 auto-framing | Detects face coordinates frame-by-frame to keep the speaker centered when cropping 16:9 widescreen video down to vertical YouTube Shorts. |

---

## Features

| Feature | Detail |
|---|---|
| **Greek Transcription** | `faster-whisper` (`large-v3` / `small`) — word-level karaoke timestamps with forced `el` language |
| **Active Framing** | Dynamic center/speaker tracking for 9:16 conversion |
| **B-Roll Cutaways** | Pexels API — topic-matched HD B-roll filling the 9:16 frame |
| **Subtitle Burn-in** | ASS format, Greek-glyph-safe Arial with active-word highlight |
| **Outro Concatenation** | Optional branded bumper appended to every Short |
| **SEO & Viral Hooks** | LLM content analysis (Greek titles, descriptions, tags, and hook scoring) |
| **Batch Processing** | Bulk process multiple clips with single-click orchestration |
| **Zero Config** | Cross-platform launcher (`run.bat` / `run.sh`) with auto-dependency setup |

---

## 1-Click Start (Windows)

```
git clone https://github.com/YOUR_USERNAME/shorts-engine.git
cd shorts-engine/shorts_engine
run.bat
```

`run.bat` will:
1. Check for Python 3.10+ (exits with instructions if missing)
2. Download the FFmpeg static build into `./bin/` if `ffmpeg` is not already on PATH
3. Create a `.venv` virtual environment
4. Install all Python dependencies from `requirements.txt`
5. Launch the Streamlit app in your default browser

---

## Linux / macOS Start

```bash
# Prerequisites: Python 3.10+, ffmpeg (via apt / brew / etc.)
cd shorts_engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## API Keys Required

Set the following in the Streamlit sidebar (or create a `.env` file in `shorts_engine/`):

| Variable | Where to get it |
|---|---|
| `PEXELS_API_KEY` | https://www.pexels.com/api/ — free tier, 200 req/hour |
| `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey — free tier available |

> Keys entered in the sidebar are **never persisted to disk** — they live only in the Streamlit session.

---

## Architecture

```
app.py          ← Streamlit UI (zero business logic)
config.py       ← Settings, PATH injection, binary assertions
pipeline.py     ← Batch orchestrator, TempDirectory lifecycle
services/
  transcriber.py   ← faster-whisper → ASS subtitles
  broll_fetcher.py ← Pexels API search + download
  video_engine.py  ← FFmpeg crop / overlay / burn / concat
  seo_generator.py ← Gemini 2.5 Flash → SEO JSON
```

---

## Output

Each processed Short is written to `./output/<YYYYMMDD_HHMMSS>/<original_filename>_short.mp4`.
An `seo_<original_filename>.json` metadata file is placed alongside it.

---

## Notes

- **Whisper model size**: defaults to `base` for speed. Change `WHISPER_MODEL_SIZE` in `config.py` to `small`/`medium`/`large-v3` for higher accuracy.
- **No CUDA required**: the pipeline runs entirely on CPU.
- **Pexels B-roll**: if the API key is omitted or quota is exhausted, the pipeline skips the B-roll step gracefully.
