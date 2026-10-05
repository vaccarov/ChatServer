import asyncio
import threading
from typing import Any

from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from app.core.constants import (
	STATUS_ERROR,
	STATUS_GENERATING,
	STATUS_LOADING_MODEL,
	STATUS_PROGRESS,
	STATUS_REFINING,
	STATUS_STARTING_IMAGE,
	STATUS_SUCCESS,
)
from app.schemas.models import ImageGenerationRequest
from app.services.image.utils import get_pipe_args, get_refiner_args, image_to_base64, load_pipeline, load_refiner


def _run_generation(req: ImageGenerationRequest, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
	"""Runs the whole diffusion pipeline on a worker thread, pushing progress onto `queue`."""

	def push(payload: dict | None) -> None:
		loop.call_soon_threadsafe(queue.put_nowait, payload)

	def on_step(pipe: DiffusionPipeline, step: int, timestep: Any, callback_kwargs: dict) -> dict:
		push({'status': STATUS_PROGRESS, 'step': step + 1, 'total_steps': req.steps})
		return callback_kwargs

	try:
		push({'status': STATUS_LOADING_MODEL, 'model': req.model_name})
		pipe = load_pipeline(req.model_name, is_img2img=req.input_image_pil is not None)

		push({'status': STATUS_GENERATING})
		images = pipe(**get_pipe_args(req, on_step)).images

		if req.use_refiner:
			push({'status': STATUS_LOADING_MODEL, 'model': STATUS_REFINING})
			refiner = load_refiner(pipe)
			images = refiner(**get_refiner_args(req, images, on_step)).images

		for index, image in enumerate(images):
			push(
				{'status': STATUS_STARTING_IMAGE, 'image_number': index + 1, 'total_images': req.num_images_per_prompt}
			)
			push({'status': STATUS_SUCCESS, 'image_data': image_to_base64(image)})
	except Exception as e:
		push({'status': STATUS_ERROR, 'message': str(e)})
	finally:
		push(None)


async def generate_image(req: ImageGenerationRequest):
	"""Generates or modifies an image based on a prompt using a non-blocking approach."""
	loop = asyncio.get_running_loop()
	queue: asyncio.Queue = asyncio.Queue()
	threading.Thread(target=_run_generation, args=(req, loop, queue)).start()

	while (progress := await queue.get()) is not None:
		yield progress
