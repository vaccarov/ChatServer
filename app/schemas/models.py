"""
This file contains the Pydantic models used for data validation and serialization.
"""

from pydantic import BaseModel, Field, model_validator
from app.core.constants import SDXL_BASE_MODEL, MODEL_LCM
from typing import Optional
from PIL import Image


class ImageGenerationRequest(BaseModel):
	"""Defines the structure for an image generation request."""

	prompt: str  # The main text prompt that describes the desired image.
	model_name: str = SDXL_BASE_MODEL  # The generation model to use, ex: 'sdxl' or 'lcm'.
	steps: int = 25  # The number of diffusion steps to run.
	num_images_per_prompt: int = 1  # The number of images to generate.
	negative_prompt: Optional[str] = None  # A comma-separated list of terms to exclude from the image.
	strength: Optional[float] = Field(
		default=None, ge=0.0, le=1.0
	)  # The influence of the input image in image-to-image generation (0.0 to 1.0).
	guidance_scale: Optional[float] = None  # The guidance scale for the diffusion model.
	denoising: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # The denoising strength for the refiner model.
	use_refiner: Optional[bool] = None  # Whether to use the SDXL refiner model to improve image details.
	input_image_pil: Optional[Image.Image] = None  # Input Image to modify

	class Config:
		arbitrary_types_allowed: bool = True

	@model_validator(mode='after')
	def _validate_fields_combination(self) -> 'ImageGenerationRequest':
		if self.model_name == MODEL_LCM and self.use_refiner:
			raise ValueError('The refiner cannot be used with the LCM model.')
		if self.input_image_pil is not None and self.num_images_per_prompt > 1:
			raise ValueError('Batch generation (num_images_per_prompt > 1) is not supported for image-to-image.')
		if self.input_image_pil is None and self.strength is not None:
			raise ValueError("The 'strength' parameter is only applicable for image-to-image generation.")
		return self
