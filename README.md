# ChatServer

`ChatServer` est un backend FastAPI léger conçu pour la transcription audio en texte, utilisant le modèle OpenAI Whisper. Il est optimisé pour gérer les requêtes de transcription de manière asynchrone.

## Fonctionnalités

*   **Transcription Audio :** Convertit les fichiers audio (format WEBM) en texte.
*   **Support Multilingue :** Permet de spécifier la langue de l'audio pour une transcription précise.
*   **Traitement Asynchrone :** Utilise `fastapi.concurrency.run_in_threadpool` pour décharger les opérations de transcription gourmandes en ressources vers un pool de threads séparé.
*   **Gestion des Fichiers Temporaires :** Nettoyage automatique des fichiers audio temporaires (`.webm` et `.wav`) après transcription.

## Technologies Utilisées

*   **Python**
*   **FastAPI :** Framework web pour construire l'API.
*   **Uvicorn :** Serveur ASGI pour exécuter l'application FastAPI.
*   **OpenAI Whisper :** Modèle de reconnaissance vocale pour la transcription.
*   **FFmpeg :** Outil de ligne de commande pour la conversion de formats audio.

## Prérequis

Avant de démarrer le serveur, assurez-vous d'avoir les éléments suivants installés :

*   **Python 3.9+**
*   **FFmpeg :** `brew install ffmpeg`

## Installation et Démarrage

Suivez ces étapes pour configurer et lancer le serveur :

1.  **Naviguez vers le répertoire du projet :**
    ```bash
    cd ./ChatServer
    ```

2.  **Créez et activez un environnement virtuel :**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Installez les dépendances Python :**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note : Lors de la première exécution, le modèle Whisper (`large-v3-turbo`) sera automatiquement téléchargé et mis en cache dans `~/.cache/whisper/`.)*

4.  **Démarrez le serveur Uvicorn :**
    ```bash
    uvicorn app.main:app --reload
    ```
    Le serveur sera accessible à `http://127.0.0.1:8000` (ou un autre port si spécifié par Uvicorn).

## Utilisation de l'API

Le serveur expose un endpoint unique pour la transcription :

### `POST /transcribe`

*   **Description :** Transcrit un fichier audio en texte.
*   **Méthode :** `POST`
*   **URL :** `http://127.0.0.1:8000/transcribe`
*   **Paramètres de formulaire (`multipart/form-data`) :**
    *   `file` (type `File`) : Le fichier audio à transcrire (ex: `.webm`, `.m4a`, `.wav`).
    *   `language` (type `Form`, `string`) : La langue de l'audio (ex: `"french"`, `"english"`, `"fr"`, `"en"`).

#### Exemple avec `curl` :

```bash
curl -X POST \
  -F "file=@/path/to/your/audio.webm" \
  -F "language=french" \
  http://127.0.0.1:8000/transcribe
```

#### Réponse attendue (JSON) :

```json
{
  "transcript": "Ceci est un exemple de texte transcrit."
}
```

## Configuration du Modèle Whisper

Le modèle Whisper utilisé est `large-v3`. Vous pouvez le modifier dans `app/whisper_utils.py` si vous souhaitez utiliser un modèle plus petit (ex: `"medium"`, `"base"`) pour des performances plus rapides, ou si vous rencontrez des problèmes de mémoire.

```python
# app/whisper_utils.py
MODEL = whisper.load_model("large-v3") # Change "large-v3" to "medium" or "base" if needed
```