import asyncio
import threading
from typing import Dict, Any
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from app.core.constants import (
	STATUS_ERROR,
	STATUS_GENERATING,
	STATUS_LOADING_MODEL,
	STATUS_PROGRESS,
	STATUS_REFINING,
	STATUS_SUCCESS,
	STATUS_STARTING_IMAGE,
)
from app.schemas.models import ImageGenerationRequest
from app.services.image.utils import _load_pipeline, _image_to_base64, _load_refiner, PipelineArguments


class ProgressCallback:
	def __init__(self, req: ImageGenerationRequest, progress_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
		self.req = req
		self.progress_queue = progress_queue
		self.loop = loop

	def __call__(
		self, pipe: DiffusionPipeline, step: int, timestep: Any, callback_kwargs: Dict[str, Any]
	) -> Dict[str, Any]:
		self.loop.call_soon_threadsafe(
			self.progress_queue.put_nowait, {'status': STATUS_PROGRESS, 'step': step + 1, 'total_steps': self.req.steps}
		)
		return callback_kwargs


class ImageGenerationProcess:
	def __init__(self, req: ImageGenerationRequest, loop: asyncio.AbstractEventLoop, progress_queue: asyncio.Queue):
		self.req = req
		self.loop = loop
		self.progress_queue = progress_queue
		self.callback_handler = ProgressCallback(req, progress_queue, loop)

	def _run_generation_thread(self):
		try:
			# SEND LOADING MESSAGE
			self.loop.call_soon_threadsafe(
				self.progress_queue.put_nowait, {'status': STATUS_LOADING_MODEL, 'model': self.req.model_name}
			)
			# SETUP BASE PIPELINE
			pipe = _load_pipeline(self.req.model_name, is_img2img=self.req.input_image_pil is not None)
			# SETUP PIPELINE ARGUMENTS
			pipeline_args_builder = PipelineArguments(self.req)
			pipe_args: Dict[str, Any] = pipeline_args_builder.get_pipe_args(self.callback_handler)
			# RUN BASE PIPELINE
			self.loop.call_soon_threadsafe(self.progress_queue.put_nowait, {'status': STATUS_GENERATING})
			images_or_latents = pipe(**pipe_args).images
			if self.req.use_refiner:
				# SETUP REFINER PIPELINE
				self.loop.call_soon_threadsafe(
					self.progress_queue.put_nowait, {'status': STATUS_LOADING_MODEL, 'model': STATUS_REFINING}
				)
				refiner = _load_refiner(pipe)
				refiner_args = pipeline_args_builder.get_refiner_args(images_or_latents, self.callback_handler)
				# RUN REFINER PIPELINE
				images_or_latents = refiner(**refiner_args).images
			# Process and send final images
			for i, image in enumerate(images_or_latents):
				self.loop.call_soon_threadsafe(
					self.progress_queue.put_nowait,
					{
						'status': STATUS_STARTING_IMAGE,
						'image_number': i + 1,
						'total_images': self.req.num_images_per_prompt,
					},
				)
				base64_image = _image_to_base64(image)
				self.loop.call_soon_threadsafe(
					self.progress_queue.put_nowait, {'status': STATUS_SUCCESS, 'image_data': base64_image}
				)
		except Exception as e:
			self.loop.call_soon_threadsafe(self.progress_queue.put_nowait, {'status': STATUS_ERROR, 'message': str(e)})
		finally:
			self.loop.call_soon_threadsafe(self.progress_queue.put_nowait, None)


async def generate_image(req: ImageGenerationRequest):
	"""Generates or modifies an image based on a prompt using a non-blocking approach."""
	loop = asyncio.get_running_loop()
	progress_queue = asyncio.Queue()

	process = ImageGenerationProcess(req, loop, progress_queue)
	thread = threading.Thread(target=process._run_generation_thread)
	thread.start()

	while True:
		progress = await progress_queue.get()
		if progress is None:
			break
		yield progress
