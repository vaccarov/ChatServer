import json
import os
from pathlib import Path
from typing import Optional
from app.constants import LCM_SDXL_MODEL, MODEL_LCM, MODEL_SDXL, MODELS_PATH, SDXL_BASE_MODEL, SDXL_REFINER_MODEL
from app.forms import ImageGenerationForm
from app.image_generation.core import generate_image
from app.image_generation.utils import PIPELINE_CACHE
from app.models import ImageGenerationRequest
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import io

router = APIRouter()

@router.post("/generate")
async def generate_image_endpoint(
    form: ImageGenerationForm = Depends(ImageGenerationForm),
    image: Optional[UploadFile] = File(None),
):
    req = ImageGenerationRequest(**vars(form))

    if image:
        image_bytes = await image.read()
        req.input_image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Fixes wrong parameters
    if req.model_name == MODEL_LCM:
        req.use_refiner = False
    if req.input_image_pil is not None:
        req.batch_size = 1
    else:
        req.strength = None

    async def event_stream():
        try:
            async for progress in generate_image(req):
                yield f"data: {json.dumps(progress)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.get("/models")
async def get_models():
    """Lists available text/image-to-image models and their loaded status."""
    try:
        cache_path = Path(MODELS_PATH).expanduser()
        model_info = {
            MODEL_SDXL: SDXL_BASE_MODEL,
            MODEL_LCM: LCM_SDXL_MODEL,
        }
        model_patterns = {
            name: str(cache_path / f"models--{model_id.replace('/', '--')}")
            for name, model_id in model_info.items()
        }
        available_models = [
            name for name, pattern in model_patterns.items() if os.path.exists(pattern)
        ]
        loaded_model_names = {key.split('_')[0] for key in PIPELINE_CACHE.keys()}
        return [
            {
                "model_id": model_info.get(model_name),
                "name": model_name,
                "loaded": model_name in loaded_model_names,
            }
            for model_name in available_models
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
