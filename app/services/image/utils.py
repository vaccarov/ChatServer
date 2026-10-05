import base64
import io
from collections.abc import Callable
from typing import Any

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_lcm import LCMScheduler
from PIL import Image

from app.core.constants import LCM_SDXL_MODEL, MODEL_LCM, SDXL_BASE_MODEL, SDXL_REFINER_MODEL
from app.schemas.models import ImageGenerationRequest

PIPELINE_CACHE: dict[str, DiffusionPipeline] = {}

StepCallback = Callable[[Any, int, Any, dict[str, Any]], dict[str, Any]]


def get_pipe_args(req: ImageGenerationRequest, callback: StepCallback) -> dict[str, Any]:
	pipe_args: dict[str, Any] = {
		'prompt': req.prompt,
		'num_inference_steps': req.steps,
		'callback_on_step_end': callback,
		'num_images_per_prompt': req.num_images_per_prompt,
		**({'guidance_scale': req.guidance_scale} if req.guidance_scale is not None else {}),
		**({'negative_prompt': req.negative_prompt} if req.negative_prompt is not None else {}),
	}

	if req.use_refiner:
		pipe_args['output_type'] = 'latent'
		if req.denoising is not None:
			pipe_args['denoising_end'] = req.denoising

	if req.input_image_pil is not None:
		pipe_args['image'] = req.input_image_pil
		if req.strength is not None:
			pipe_args['strength'] = req.strength

	return pipe_args


def get_refiner_args(req: ImageGenerationRequest, images: Any, callback: StepCallback) -> dict[str, Any]:
	return {
		'prompt': req.prompt,
		'num_inference_steps': req.steps,
		'image': images,
		'callback_on_step_end': callback,
		'num_images_per_prompt': req.num_images_per_prompt,
		**({'denoising_start': req.denoising} if req.denoising is not None else {}),
	}


DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
# ponytail: half precision off CPU only; add a dtype knob if a device ever disagrees.
_COMMON_PIPELINE_ARGS: dict[str, Any] = {
	'torch_dtype': torch.float32 if DEVICE == 'cpu' else torch.float16,
	'use_safetensors': True,
	'variant': None if DEVICE == 'cpu' else 'fp16',
}


def load_pipeline(model_name: str, is_img2img: bool) -> DiffusionPipeline:
	"""Loads the appropriate diffusion pipeline based on the model name and task, using a cache."""
	cache_key = f'{model_name}_{is_img2img}'
	if cache_key in PIPELINE_CACHE:
		return PIPELINE_CACHE[cache_key]

	pipeline_class = AutoPipelineForImage2Image if is_img2img else AutoPipelineForText2Image
	if model_name == MODEL_LCM:
		unet = UNet2DConditionModel.from_pretrained(LCM_SDXL_MODEL, **_COMMON_PIPELINE_ARGS)
		pipe = pipeline_class.from_pretrained(SDXL_BASE_MODEL, unet=unet, **_COMMON_PIPELINE_ARGS).to(DEVICE)
		scheduler_config = dict(pipe.scheduler.config)
		scheduler_config.pop('skip_prk_steps', None)
		pipe.scheduler = LCMScheduler.from_config(scheduler_config)
	else:
		pipe = pipeline_class.from_pretrained(SDXL_BASE_MODEL, **_COMMON_PIPELINE_ARGS).to(DEVICE)

	PIPELINE_CACHE[cache_key] = pipe
	return pipe


def load_refiner(pipe: DiffusionPipeline) -> DiffusionPipeline:
	"""Loads the refiner pipeline, reusing components from the base pipeline."""
	if 'refiner' in PIPELINE_CACHE:
		return PIPELINE_CACHE['refiner']
	refiner = AutoPipelineForImage2Image.from_pretrained(
		SDXL_REFINER_MODEL, text_encoder_2=pipe.text_encoder_2, vae=pipe.vae, **_COMMON_PIPELINE_ARGS
	).to(DEVICE)
	PIPELINE_CACHE['refiner'] = refiner
	return refiner


def image_to_base64(image: Image.Image) -> str:
	"""Converts a PIL image to a base64 encoded string."""
	buffered = io.BytesIO()
	image.save(buffered, format='PNG')
	return base64.b64encode(buffered.getvalue()).decode('utf-8')
