# ChatServer

`ChatServer` is a lightweight FastAPI backend designed for audio-to-text transcription, utilizing the OpenAI Whisper model. It is optimized to handle transcription requests asynchronously.

## Features

*   **Audio Transcription:** Converts audio files (WEBM format) into text.
*   **Multilingual Support:** Allows specifying the audio language for accurate transcription.
*   **Asynchronous Processing:** Uses `fastapi.concurrency.run_in_threadpool` to offload resource-intensive transcription operations to a separate thread pool.
*   **Temporary File Management:** Automatically cleans up temporary audio files (`.webm` and `.wav`) after transcription.

## Technologies Used

*   **Python**
*   **FastAPI:** Web framework for building the API.
*   **Uvicorn:** ASGI server to run the FastAPI application.
*   **OpenAI Whisper:** Speech recognition model for transcription.
*   **FFmpeg:** Command-line tool for converting audio formats.

## Prerequisites

Before starting the server, ensure you have the following installed:

*   **Python 3.9+**
*   **FFmpeg:** `brew install ffmpeg`

## Installation and Startup

Follow these steps to set up and launch the server:

1.  **Navigate to the project directory:**
    ```bash
    cd ./ChatServer
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: On the first run, the Whisper model (`large-v3-turbo`) will be automatically downloaded and cached in `~/.cache/whisper/`.)*

4.  **Start the Uvicorn server:**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --reload
    ```
    The server will be accessible at `http://127.0.0.1:8000` (or another port if specified by Uvicorn).

## API Usage

The server exposes a single endpoint for transcription:

### `POST /transcribe`

*   **Description:** Transcribes an audio file into text.
*   **Method:** `POST`
*   **URL:** `http://127.0.0.1:8000/transcribe`
*   **Form parameters (`multipart/form-data`):
    *   `file` (type `File`): The audio file to transcribe (e.g., `.webm`, `.m4a`, `.wav`).
    *   `language` (type `Form`, `string`): The language of the audio (e.g., `"french"`, `"english"`, `"fr"`, `"en"`).

#### Example with `curl`:

```bash
curl -X POST \
  -F "file=@/path/to/your/audio.webm" \
  -F "language=english" \
  http://127.0.0.1:8000/transcribe
```

#### Expected response (JSON):

```json
{
  "transcript": "This is an example of transcribed text."
}
```

## Whisper Model Configuration

The Whisper model used is `large-v3`. You can change it in `app/whisper_utils.py` if you want to use a smaller model (e.g., `"medium"`, `"base"`) for faster performance, or if you encounter memory issues.

```python
# app/whisper_utils.py
MODEL = whisper.load_model("large-v3") # Change "large-v3" to "medium" or "base" if needed
```

## Docker Container Information

The size of the Docker image depends on the Whisper model selected in `app/whisper_utils.py`:

*   If `MODEL_NAME = "large-v3-turbo"` is selected, the image size is approximately **4.92 GB**.
*   If `MODEL_NAME = "tiny"` is selected, the image size is approximately **1.97 GB**.

### Building the Docker Image

To build & run the Docker image for the ChatServer, navigate to the `ChatServer` directory and run the following command:

```bash
docker build -t chatserver .
docker run -d -p 8000:8000 --name chatserver-container chatserver
```