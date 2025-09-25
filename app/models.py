"""
This file contains the Pydantic models used for data validation and serialization.
"""
from pydantic import BaseModel
from app.constants import SDXL_BASE_MODEL
from typing import Optional
from PIL import Image

class ImageGenerationRequest(BaseModel):
    """Defines the structure for an image generation request."""
    prompt: str  # The main text prompt that describes the desired image.
    model_name: str = SDXL_BASE_MODEL  # The generation model to use, ex: 'sdxl' or 'lcm'.
    steps: int = 25  # The number of diffusion steps to run.
    batch_size: int = 1  # The number of images to generate.
    negative_prompt: str | None  # A comma-separated list of terms to exclude from the image.
    strength: float | None  # The influence of the input image in image-to-image generation (0.0 to 1.0).
    guidance_scale: float | None  # The guidance scale for the diffusion model.
    denoising: float | None  # The denoising strength for the refiner model.
    use_refiner: bool | None  # Whether to use the SDXL refiner model to improve image details.
    input_image_pil: Optional[Image.Image] = None

    class Config:
        arbitrary_types_allowed: bool = True
