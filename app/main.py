from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import images, audio, documents, tts

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
app.include_router(images.router, prefix='/image')
app.include_router(audio.router, prefix='/audio')
app.include_router(documents.router, prefix='/documents')
app.include_router(tts.router, prefix='/tts')


@app.get('/')
async def health_check():
	return {'success': True}
