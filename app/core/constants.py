"""
This file contains constants used throughout the application.
"""

# Device specific constants
MODELS_PATH = '~/.cache/huggingface/hub'

# Statuses
STATUS_LOADING_MODEL = 'loading_model'
STATUS_GENERATING = 'generating'
STATUS_REFINING = 'refining'
STATUS_PROGRESS = 'progress'
STATUS_SUCCESS = 'success'
STATUS_ERROR = 'error'
STATUS_STARTING_IMAGE = 'starting_image'

# Model Paths
SDXL_BASE_MODEL = 'stabilityai/stable-diffusion-xl-base-1.0'
SDXL_REFINER_MODEL = 'stabilityai/stable-diffusion-xl-refiner-1.0'
LCM_SDXL_MODEL = 'latent-consistency/lcm-sdxl'

# Model Names
MODEL_SDXL = 'sdxl'
MODEL_LCM = 'lcm'

# ChromaDB Constants
CHROMA_PATH = 'db'

# LLM server used for embeddings when the client does not send one (Ollama by default)
DEFAULT_LLM_BASE_URL = 'http://localhost:11434'
