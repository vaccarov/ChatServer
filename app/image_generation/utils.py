import base64
import io
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    LCMScheduler,
    UNet2DConditionModel,
    DiffusionPipeline,
)
from PIL import Image
from app.constants import (
    DEVICE,
    VARIANT,
    LCM_SDXL_MODEL,
    MODEL_LCM,
    SDXL_BASE_MODEL,
    SDXL_REFINER_MODEL,
)
from app.models import ImageGenerationRequest

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True)
PIPELINE_CACHE: Dict[str, Tuple[DiffusionPipeline, Optional[DiffusionPipeline]]] = {}

class PipelineArguments:
    def __init__(self, req: ImageGenerationRequest):
        self.req = req
    def get_pipe_args(self, callback: Callable[[Any, int, Any, Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        pipe_args: Dict[str, Any] = {
            "prompt": self.req.prompt,
            "num_inference_steps": self.req.steps,
            "callback_on_step_end": callback,
            **({"guidance_scale": self.req.guidance_scale} if self.req.guidance_scale is not None else {}),
            **({"negative_prompt": self.req.negative_prompt} if self.req.negative_prompt is not None else {}),
        }

        if self.req.use_refiner:
            pipe_args["output_type"] = "latent"
            if self.req.denoising is not None:
                pipe_args["denoising_end"] = self.req.denoising

        if self.req.input_image_pil is not None:
            pipe_args["image"] = self.req.input_image_pil
            if self.req.strength is not None:
                pipe_args["strength"] = self.req.strength

        return pipe_args

    def get_refiner_args(self, image: Image.Image, callback: Callable[[Any, int, Any, Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        refiner_args: Dict[str, Any] = {
            "prompt": self.req.prompt,
            "num_inference_steps": self.req.steps,
            "image": image,
            "callback_on_step_end": callback,
            **({"denoising_start": self.req.denoising} if self.req.denoising is not None else {}),
        }

        return refiner_args

def _load_pipeline(model_name: str, use_refiner: bool | None, is_img2img: bool) -> Tuple[DiffusionPipeline, Optional[DiffusionPipeline]]:
    """Loads the appropriate diffusion pipeline based on the model name and task, using a cache."""
    cache_key = f"{model_name}_{use_refiner}_{is_img2img}"
    if cache_key in PIPELINE_CACHE:
        return PIPELINE_CACHE[cache_key]
    pipe: Optional[DiffusionPipeline] = None
    refiner: Optional[DiffusionPipeline] = None
    torch_dtype = torch.float16

    pipeline_class = AutoPipelineForImage2Image if is_img2img else AutoPipelineForText2Image
    pretrained_args: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "use_safetensors": True,
        "variant": VARIANT,
    }
    model_specific_args: Dict[str, Any] = {}
    base_model_id: str = SDXL_BASE_MODEL
    if model_name == MODEL_LCM:
        unet = UNet2DConditionModel.from_pretrained(
            LCM_SDXL_MODEL, torch_dtype=torch_dtype, variant=VARIANT
        )
        model_specific_args["unet"] = unet
    pipe = pipeline_class.from_pretrained(
        base_model_id, **pretrained_args, **model_specific_args
    ).to(DEVICE)
    if model_name == MODEL_LCM:
        scheduler_config = dict(pipe.scheduler.config)
        scheduler_config.pop("skip_prk_steps", None)
        pipe.scheduler = LCMScheduler.from_config(scheduler_config)

    if use_refiner:
        refiner = AutoPipelineForImage2Image.from_pretrained(
            SDXL_REFINER_MODEL,
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            **pretrained_args,
        ).to(DEVICE)
    PIPELINE_CACHE[cache_key] = (pipe, refiner)
    return pipe, refiner

def _image_to_base64(image: Image.Image) -> str:
    """Converts a PIL image to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
