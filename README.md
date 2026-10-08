# ChatServer

`ChatServer` is the optional local backend for [OllamaChat](https://github.com/vaccarov/OllamaChat). FastAPI + Uvicorn, designed to keep heavy local models out of the browser.

## Features

- **Image generation & editing**: text-to-image and image-to-image with SDXL, LCM and an optional SDXL refiner, streamed as Server-Sent Events.
- **Audio transcription**: WEBM/WAV/M4A to text with OpenAI Whisper.
- **Text-to-speech**: VoxCPM synthesis, returned as a WAV stream.
- **RAG over PDFs**: text extraction with PyMuPDF, OCR fallback with Tesseract, chunking, embeddings and ChromaDB vector search.
- **Provider-agnostic embeddings**: works with Ollama's `/api/embed` *and* with any OpenAI-compatible `/v1/embeddings` server (LM Studio, llama.cpp, vLLM…).
- **Async by design**: model inference runs on worker threads so the event loop stays responsive.
- **Lazy model loading**: nothing is downloaded or loaded until the matching endpoint is used.

## Prerequisites

- **Python 3.14+** and [**uv**](https://docs.astral.sh/uv/)
- **ffmpeg** — `brew install ffmpeg` (`apt install ffmpeg`)
- **tesseract** — `brew install tesseract` (`apt install tesseract-ocr`), only needed for scanned PDFs
- A running LLM server for embeddings: Ollama (default, `http://localhost:11434`) or an OpenAI-compatible one

Image generation models are downloaded on first use into `~/.cache/huggingface/hub` (login with the [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) first). The Whisper model is cached in `~/.cache/whisper` — pick a smaller one by editing `MODEL_NAME` in [`app/services/audio/core.py`](app/services/audio/core.py).

| Whisper model | Params | VRAM (approx.) | Relative speed |
| --- | --- | --- | --- |
| `tiny` | 39 M | ~1 GB | ~32x |
| `base` | 74 M | ~1 GB | ~16x |
| `small` | 244 M | ~2 GB | ~6x |
| `medium` | 769 M | ~5 GB | ~2x |
| `large` | 1550 M | ~10 GB | 1x |

VoxCPM is downloaded on the first `POST /tts` call, together with its ZipEnhancer denoiser (from ModelScope). That first call also runs a `torch.compile` warm-up, so expect it to take a while. The model id is in [`app/api/tts.py`](app/api/tts.py) — it defaults to `openbmb/VoxCPM2` (30 languages); `openbmb/VoxCPM-0.5B` is smaller but speaks English and Chinese only.

Disk usage: Whisper `large-v3-turbo` ≈ 1.6 GB, SDXL base ≈ 7.1 GB, SDXL refiner ≈ 4.7 GB, LCM ≈ 5.1 GB, VoxCPM2 ≈ 5 GB.

## Installation and startup

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --reload
```

The server listens on `http://127.0.0.1:8000`.

## API

### `GET /`

Health check: `{"success": true}`.

### `POST /audio/decode`

Transcribes a recording.

| Field | Type | Description |
| --- | --- | --- |
| `file` | file | The audio file |
| `language` | form string | Language of the audio, e.g. `fr` |

```bash
curl -X POST -F "file=@audio.webm" -F "language=fr" http://127.0.0.1:8000/audio/decode
```

### `POST /image/generate`

Generates or modifies an image. Returns a `text/event-stream` of progress events:
`loading_model`, `generating`, `progress`, `starting_image`, `success`, `error`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt` | form string | — | Main prompt |
| `model_name` | form string | `sdxl` | `sdxl` or `lcm` |
| `steps` | form int | `25` | Diffusion steps |
| `num_images_per_prompt` | form int | `1` | Images to generate (1 for image-to-image) |
| `negative_prompt` | form string | – | Terms to exclude |
| `strength` | form float | – | Image-to-image influence (0–1) |
| `guidance_scale` | form float | – | Prompt adherence |
| `denoising` | form float | – | Refiner denoising strength (0–1) |
| `use_refiner` | form bool | – | Use the SDXL refiner (not with `lcm`) |
| `image` | file | – | Input image for image-to-image |

```bash
curl -X POST -F "prompt=A futuristic cityscape at sunset" http://127.0.0.1:8000/image/generate
```

### `GET /image/models`

Lists the image models present in the local HuggingFace cache and whether they are loaded.

### `POST /documents/upload`

Uploads PDFs into a ChromaDB collection named after the embedding model. PDFs without a text layer fall back to OCR.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `files` | files | — | One or more PDFs |
| `embedding_model` | form string | — | Embedding model id, also used as the collection name |
| `chat_id` | form string | – | Scope the documents to one chat |
| `embedding_provider` | form string | `ollama` | `ollama` or anything else for OpenAI-compatible |
| `embedding_base_url` | form string | `http://localhost:11434` | Base URL of the embedding server |

### `GET /documents/list`

Lists the unique documents in a collection. Optional `embedding_model` and `chat_id` query parameters.

### `POST /documents/rag_chat`

Embeds the query, retrieves the 5 closest chunks and returns the augmented prompt.

```json
{
  "query": "What is the mitochondria?",
  "embedding_model": "nomic-embed-text",
  "chat_id": "optional-session-id",
  "embedding_provider": "ollama",
  "embedding_base_url": "http://localhost:11434"
}
```

### `POST /tts`

Synthesises speech with VoxCPM and returns `audio/wav`.

```json
{ "text": "Hello there" }
```

## Docker

```bash
docker build -t chatserver .
docker run -d -p 8000:8000 --name chatserver-container chatserver
```

The image is large: it includes Python, ffmpeg, tesseract, PyTorch and the diffusion stack. Model weights are still downloaded at runtime, so mount a volume for `~/.cache` if you want them to persist.
