"""
This file contains constants used throughout the application.
"""

# Device specific constants
MODELS_PATH = "~/.cache/huggingface/hub"
DEVICE = "mps" # Macbook MX processors
VARIANT = "fp16"

# Statuses
STATUS_LOADING_MODEL = "loading_model"
STATUS_GENERATING = "generating"
STATUS_REFINING = "refining"
STATUS_PROGRESS = "progress"
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_STARTING_IMAGE = "starting_image"

# Model Paths
SDXL_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
LCM_SDXL_MODEL = "latent-consistency/lcm-sdxl"

# Model Names
MODEL_SDXL = "sdxl"
MODEL_LCM = "lcm"
