from fastapi import Form
from app.schemas.models import ImageGenerationRequest


class ImageGenerationForm:
	def __init__(
		self,
		prompt: str = Form(...),
		model_name: str = Form(ImageGenerationRequest.model_fields['model_name'].default),
		num_images_per_prompt: int = Form(ImageGenerationRequest.model_fields['num_images_per_prompt'].default),
		steps: int = Form(ImageGenerationRequest.model_fields['steps'].default),
		strength: float | None = Form(None),
		negative_prompt: str | None = Form(None),
		guidance_scale: float | None = Form(None),
		denoising: float | None = Form(None),
		use_refiner: bool | None = Form(None),
	):
		self.prompt = prompt
		self.model_name = model_name
		self.num_images_per_prompt = num_images_per_prompt
		self.steps = steps
		self.strength = strength
		self.negative_prompt = negative_prompt
		self.guidance_scale = guidance_scale
		self.denoising = denoising
		self.use_refiner = use_refiner
