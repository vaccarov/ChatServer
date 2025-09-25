from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import image, transcript

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
app.include_router(image.router, prefix="/image")
app.include_router(transcript.router, prefix="/transcript")

@app.get("/")
async def health_check():
    return {"success": True}