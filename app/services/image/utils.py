import base64
import io
import torch
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_lcm import LCMScheduler
from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
)
from app.core.constants import (
    DEVICE,
    VARIANT,
    LCM_SDXL_MODEL,
    MODEL_LCM,
    SDXL_BASE_MODEL,
    SDXL_REFINER_MODEL,
)
from app.schemas.models import ImageGenerationRequest

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True)
PIPELINE_CACHE: Dict[str, DiffusionPipeline] = {}

class PipelineArguments:
    def __init__(self, req: ImageGenerationRequest):
        self.req = req

    def get_pipe_args(self, callback: Callable[[Any, int, Any, Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        pipe_args: Dict[str, Any] = {
            "prompt": self.req.prompt,
            "num_inference_steps": self.req.steps,
            "callback_on_step_end": callback,
            "num_images_per_prompt": self.req.num_images_per_prompt,
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

    def get_refiner_args(self, images, callback: Callable[[Any, int, Any, Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        refiner_args: Dict[str, Any] = {
            "prompt": self.req.prompt,
            "num_inference_steps": self.req.steps,
            "image": images,
            "callback_on_step_end": callback,
            "num_images_per_prompt": self.req.num_images_per_prompt,
            **({"denoising_start": self.req.denoising} if self.req.denoising is not None else {}),
        }

        return refiner_args

def _get_common_pipeline_args() -> Dict[str, Any]:
    """Returns a dictionary of common arguments for pipeline loading."""
    return {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
        "variant": VARIANT,
    }

def _load_sdxl_pipeline(pipeline_class):
    return pipeline_class.from_pretrained(
        SDXL_BASE_MODEL,
        **_get_common_pipeline_args(),
    ).to(DEVICE)

def _load_lcm_pipeline(pipeline_class):
    unet = UNet2DConditionModel.from_pretrained(
        LCM_SDXL_MODEL,
        **_get_common_pipeline_args(),
    )
    pipe = pipeline_class.from_pretrained(
        SDXL_BASE_MODEL,
        unet=unet,
        **_get_common_pipeline_args(),
    ).to(DEVICE)
    scheduler_config = dict(pipe.scheduler.config)
    scheduler_config.pop("skip_prk_steps", None)
    pipe.scheduler = LCMScheduler.from_config(scheduler_config)
    return pipe

def _load_pipeline(model_name: str, is_img2img: bool) -> DiffusionPipeline:
    """Loads the appropriate diffusion pipeline based on the model name and task, using a cache."""
    cache_key = f"{model_name}_{is_img2img}"
    if cache_key in PIPELINE_CACHE:
        return PIPELINE_CACHE[cache_key]
    pipeline_class = AutoPipelineForImage2Image if is_img2img else AutoPipelineForText2Image
    if model_name == MODEL_LCM:
        pipe = _load_lcm_pipeline(pipeline_class)
    else:
        pipe = _load_sdxl_pipeline(pipeline_class)
    PIPELINE_CACHE[cache_key] = pipe
    return pipe

def _load_refiner(pipe: DiffusionPipeline) -> Optional[DiffusionPipeline]:
    """Loads the refiner pipeline, reusing components from the base pipeline."""
    cache_key = "refiner"
    if cache_key in PIPELINE_CACHE:
        return PIPELINE_CACHE[cache_key]
    refiner = AutoPipelineForImage2Image.from_pretrained(
        SDXL_REFINER_MODEL,
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        **_get_common_pipeline_args(),
    ).to(DEVICE)
    PIPELINE_CACHE[cache_key] = refiner
    return refiner

def _image_to_base64(image: Image.Image) -> str:
    """Converts a PIL image to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
