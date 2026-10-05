from PIL import Image
from pydantic import BaseModel, Field, model_validator

from app.core.constants import MODEL_LCM, MODEL_SDXL


class ImageGenerationRequest(BaseModel):
	"""Defines the structure for an image generation request."""

	model_config = {'arbitrary_types_allowed': True}

	prompt: str  # The main text prompt that describes the desired image.
	model_name: str = MODEL_SDXL  # The generation model to use, ex: 'sdxl' or 'lcm'.
	steps: int = 25  # The number of diffusion steps to run.
	num_images_per_prompt: int = 1  # The number of images to generate.
	negative_prompt: str | None = None  # A comma-separated list of terms to exclude from the image.
	strength: float | None = Field(
		default=None, ge=0.0, le=1.0
	)  # The influence of the input image in image-to-image generation (0.0 to 1.0).
	guidance_scale: float | None = None  # The guidance scale for the diffusion model.
	denoising: float | None = Field(default=None, ge=0.0, le=1.0)  # The denoising strength for the refiner model.
	use_refiner: bool | None = None  # Whether to use the SDXL refiner model to improve image details.
	input_image_pil: Image.Image | None = None  # Input Image to modify

	@model_validator(mode='after')
	def _validate_fields_combination(self) -> 'ImageGenerationRequest':
		if self.model_name == MODEL_LCM and self.use_refiner:
			raise ValueError('The refiner cannot be used with the LCM model.')
		if self.input_image_pil is not None and self.num_images_per_prompt > 1:
			raise ValueError('Batch generation (num_images_per_prompt > 1) is not supported for image-to-image.')
		if self.input_image_pil is None and self.strength is not None:
			raise ValueError("The 'strength' parameter is only applicable for image-to-image generation.")
		return self
