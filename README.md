# ChatterBox Turbo — Web UI

A local web interface for [ChatterBox Turbo](https://github.com/resemble-ai/chatterbox) text-to-speech with voice cloning, plus a Bengali TTS backend (Meta MMS-TTS bn_BD).

## Features

- Text-to-speech with paralinguistic/emotion tags (19 tags)
- Voice cloning from reference audio (single or multiple samples per voice)
- **Multi-sample voices**: add many reference clips to one voice; samples are merged automatically to improve quality and emotion
- Voice library with favorites, ratings, sample management (add/delete individual samples)
- Real-time generation progress (SSE)
- Output formats: WAV / MP3
- Max-duration fit (speeds up to fit a target length)
- Persistent history with metadata (date, duration, size, format) — deletable per item
- Bengali (বাংলা) TTS via fully-local Meta MMS-TTS bn_BD
- Emotion/pitch/tempo/intensity controls
- Local inference on GPU (CUDA)

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install chatterbox-tts flask soundfile numpy librosa transformers
   ```

2. Place the ChatterBox Turbo checkpoint in `models/chatterbox-turbo/`.

3. Run:

   ```bash
   venv\Scripts\python app.py
   ```

   Open http://localhost:7860

## Notes

- The app runs a threaded Flask dev server on port 7860.
- Voice samples are stored in `voices/` (one folder per voice).
- Generated audio is stored in `static/outputs/`.
- `favorites.json` and `history.json` are user data (git-ignored).

## License

ChatterBox is released under the MIT License by Resemble AI. This web UI wrapper is also MIT-licensed. See `LICENSE`.