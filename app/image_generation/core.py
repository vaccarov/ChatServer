import asyncio
import threading
from typing import Optional, Dict, Any
from diffusers import DiffusionPipeline
from PIL import Image
from app.constants import (
    STATUS_ERROR,
    STATUS_GENERATING,
    STATUS_LOADING_MODEL,
    STATUS_PROGRESS,
    STATUS_REFINING,
    STATUS_SUCCESS,
    STATUS_STARTING_IMAGE,
)
from app.models import ImageGenerationRequest
from app.image_generation.utils import _load_pipeline, _image_to_base64, PipelineArguments, IMAGE_DIR

class ProgressCallback:
    def __init__(self, req: ImageGenerationRequest, progress_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.req = req
        self.progress_queue = progress_queue
        self.loop = loop

    def __call__(self, pipe: DiffusionPipeline, step: int, timestep: Any, callback_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        self.loop.call_soon_threadsafe(self.progress_queue.put_nowait, {"status": STATUS_PROGRESS, "step": step + 1, "total_steps": self.req.steps})
        return callback_kwargs

class ImageGenerationProcess:
    def __init__(self, req: ImageGenerationRequest, loop: asyncio.AbstractEventLoop, progress_queue: asyncio.Queue):
        self.req = req
        self.loop = loop
        self.progress_queue = progress_queue
        self.callback_handler = ProgressCallback(req, progress_queue, loop)

    def _run_generation_thread(self):
        try:
            self.loop.call_soon_threadsafe(
                self.progress_queue.put_nowait, {"status": STATUS_LOADING_MODEL, "model": self.req.model_name}
            )
            is_img2img: bool = self.req.input_image_pil is not None
            pipe, refiner = _load_pipeline(self.req.model_name, self.req.use_refiner, is_img2img=is_img2img)
            pipeline_args_builder = PipelineArguments(self.req)
            for i in range(self.req.batch_size):
                self.loop.call_soon_threadsafe(
                    self.progress_queue.put_nowait, {"status": STATUS_STARTING_IMAGE, "image_number": i + 1, "total_images": self.req.batch_size}
                )
                pipe_args: Dict[str, Any] = pipeline_args_builder.get_pipe_args(self.callback_handler)
                self.loop.call_soon_threadsafe(self.progress_queue.put_nowait, {"status": STATUS_GENERATING})
                generated_image: Image.Image = pipe(**pipe_args).images[0]
                if self.req.use_refiner and refiner:
                    refiner_args: Dict[str, Any] = pipeline_args_builder.get_refiner_args(generated_image, self.callback_handler)
                    self.loop.call_soon_threadsafe(self.progress_queue.put_nowait, {"status": STATUS_REFINING})
                    generated_image = refiner(**refiner_args).images[0]
                base64_image: str = _image_to_base64(generated_image)
                self.loop.call_soon_threadsafe(
                    self.progress_queue.put_nowait, {"status": STATUS_SUCCESS, "image_data": base64_image}
                )
        except Exception as e:
            self.loop.call_soon_threadsafe(
                self.progress_queue.put_nowait, {"status": STATUS_ERROR, "message": str(e)}
            )
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