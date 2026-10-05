import io
import json
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import ValidationError

from app.core.constants import LCM_SDXL_MODEL, MODEL_LCM, MODEL_SDXL, MODELS_PATH, SDXL_BASE_MODEL
from app.schemas.forms import ImageGenerationForm
from app.schemas.models import ImageGenerationRequest
from app.services.image.core import generate_image
from app.services.image.utils import PIPELINE_CACHE

router = APIRouter()


@router.post('/generate')
async def generate_image_endpoint(
	form: Annotated[ImageGenerationForm, Depends(ImageGenerationForm)],
	image: Annotated[UploadFile | None, File()] = None,
):
	payload: dict[str, Any] = vars(form)
	if image:
		payload['input_image_pil'] = Image.open(io.BytesIO(await image.read())).convert('RGB')
	try:
		req = ImageGenerationRequest.model_validate(payload)
	except ValidationError as e:
		# `include_input=False`: the payload holds a PIL image, which is not JSON serialisable.
		raise HTTPException(
			status_code=422, detail=e.errors(include_input=False, include_url=False, include_context=False)
		)

	async def event_stream() -> AsyncIterator[str]:
		try:
			async for progress in generate_image(req):
				yield f'data: {json.dumps(progress)}\n\n'
		except Exception as e:
			yield f'data: {json.dumps({"status": "error", "message": str(e)})}\n\n'

	return StreamingResponse(event_stream(), media_type='text/event-stream')


@router.get('/models')
async def get_models():
	"""Lists available text/image-to-image models and their loaded status."""
	try:
		cache_path = Path(MODELS_PATH).expanduser()
		model_info = {MODEL_SDXL: SDXL_BASE_MODEL, MODEL_LCM: LCM_SDXL_MODEL}
		loaded_model_names = {key.split('_')[0] for key in PIPELINE_CACHE}
		return [
			{'fullname': fullname, 'name': name, 'loaded': name in loaded_model_names}
			for name, fullname in model_info.items()
			if (cache_path / f'models--{fullname.replace("/", "--")}').exists()
		]
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
