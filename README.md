# TalkBridge Server

> Python backend for real-time speech recognition, translation, and text-to-speech synthesis.

---

## Overview

The TalkBridge server is a **FastAPI** application that receives audio from the Android client, transcribes it with **faster-whisper**, translates it with **argostranslate**, and synthesizes speech with **edge-tts**. It communicates with clients over WebSocket for real-time streaming and via HTTP for on-demand transcription and translation.

---

## Features

- **Real-time WebSocket pipeline** — receives PCM 16-bit audio chunks and returns translated text + TTS audio
- **Voice Activity Detection (VAD)** via `webrtcvad` — segments audio at natural speech boundaries
- **Speech-to-Text** via `faster-whisper` (`large-v3-turbo` on CUDA, `medium` on CPU)
- **Machine translation** via `OPUS-MT` and `argostranslate` — offline, no API costs
- **Text-to-Speech** via `piper` - offline, available for 30 languages
- **SSE streaming** for the `/transcript` endpoint — sends estimated time before the result
- **GPU acceleration** via CUDA (NVIDIA) — sub-0.25s transcription on RTX 3070
- **CPU fallback** with `int8` quantization when no GPU is available

---

## Tech Stack

| Component | Technology |
|---|---|
| Web framework | FastAPI + uvicorn |
| Speech-to-Text | faster-whisper |
| Translation | OPUS-MT & argostranslate |
| Text-to-Speech | piper |
| VAD | webrtcvad |
| Audio processing | NumPy |
| GPU acceleration | CUDA via PyTorch |

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (strongly recommended for real-time performance)
- CUDA toolkit installed and accessible

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Install the CUDA-enabled build of PyTorch. See [pytorch.org](https://pytorch.org/get-started/locally/) for the correct install command for your system.

>  Installing via the requirements.txt can have problems - pytoml coming soon

### Install argostranslate language packages

Before running the server, install the required translation and tts packages (see /helper folder for OPUS-MT installation script - piper and argostranslate download scripts might be added in the future for more convenience) 

---

## Running the Server

```bash
python server.py
```

The server starts on `0.0.0.0:80` by default.

---

## API Reference

### `WS /ws/translate`

Main real-time translation endpoint.

**Init message (client → server):**
```json
{
  "source_lang": "de",
  "target_lang": "en",
  "use_better_translation": true, //true => opus, false => argos
  "voice_gender": "male" // not unfinished
}
```

**Server responses:**

| Type | Format | Description |
|---|---|---|
| `connected` | JSON | Connection acknowledged |
| `ready` | JSON | Session ready, model warmed up |
| `final` (translated) | JSON | Translated text |
| audio bytes | binary | TTS audio of the translated text |

**Audio format expected from client:**
- PCM signed 16-bit little-endian
- Sample rate: 16,000 Hz
- Chunk size: ~3,200 bytes (≈ 100ms)

---

### `POST /transcript`

Transcribes a WAV audio file. Returns an SSE stream.

**Request:** `multipart/form-data`
- `lang` — BCP-47 language code (e.g. `de`, `en`). Use `auto` for automatic detection.
- `audio` — WAV file (PCM 16-bit, 16kHz)

**Response:** `text/event-stream`
```
data: {"type": "estimated_time", "seconds": 2}

data: {"type": "transcript", "text": "Hallo, wie geht es dir?"}
```

---

### `POST /translate`

Translates a text string synchronously.

**Request:** `multipart/form-data`
- `text` — source text
- `source_lang` — e.g. `de`
- `target_lang` — e.g. `en`

**Response:**
```json
{ "translated": "Hello, how are you?" }
```

---

### `GET /`

Health check.

```json
{ "message": "TalkBridge WebSocket Server", "status": "running" }
```

---

## Notes

- **uvicorn streaming:** `StreamingResponse` chunks require `await asyncio.sleep(0)` after each `yield` to flush the event loop buffer. Blocking calls (e.g. `model.transcribe`) must be wrapped in `loop.run_in_executor(None, ...)`.
- **Model warmup:** The server runs a warmup translation on session init to avoid cold-start latency on the first real request.
- **Argostranslate model path:** Set via `ARGOS_PACKAGES_DIR` environment variable, resolved relative to `server.py` at startup.

---

## License

This project was developed as part of a **Jugend Forscht** youth science competition entry.




