from fastapi import Form
from app.models import ImageGenerationRequest
from PIL import Image

class ImageGenerationForm:
    def __init__(
        self,
        prompt: str = Form(...),
        model_name: str = Form(ImageGenerationRequest.model_fields['model_name'].default),
        batch_size: int = Form(ImageGenerationRequest.model_fields['batch_size'].default),
        steps: int = Form(ImageGenerationRequest.model_fields['steps'].default),
        strength: float | None = Form(None),
        negative_prompt: str | None = Form(None),
        guidance_scale: float | None = Form(None),
        denoising: float | None = Form(None),
        use_refiner: bool | None = Form(None),
    ):
        self.prompt = prompt
        self.model_name = model_name
        self.batch_size = batch_size
        self.steps = steps
        self.strength = strength
        self.negative_prompt = negative_prompt
        self.guidance_scale = guidance_scale
        self.denoising = denoising
        self.use_refiner = use_refiner