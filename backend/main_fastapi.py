import os
import tempfile
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import requests
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import asyncio
from huggingface_hub import login

login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

from src.agent import run_agent
from src.segmentor import SamSegmentor
from src.logger import get_logger
from src.image_utils import get_image_palette_with_mask
from src.utils import get_object_name

logger = get_logger(__name__)


app = FastAPI()


# Instantiation of the singleton
segmentor = SamSegmentor()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class ChatRequest(BaseModel):
    query: str
    chat_id: str

@app.post("/ask")
async def ask_agent(request: ChatRequest):
    logger.info(f"Received query: {request.query} for chat_id: {request.chat_id}")
    response = run_agent(
        message=request.query,
        chat_id=request.chat_id
    )
    logger.info(f"Agent response: {response.get('content', None)}")
    return {"response": response}



@app.post("/segment")
async def segment_images(
    prompts: List[str],
    image_urls: List[str] = None,
    images: List[UploadFile] = File(None),
):
    """Endpoint to segment images based on prompts."""
    logger.info(f"Received segmentation request with prompts: {prompts}")

    if not images and not image_urls:
        return {"error": "Provide either image URLs or uploaded images."}
    
    if not images:
        images = []
    
    if len(prompts) != len(images) and len(prompts) != len(image_urls or []):
        return {"error": "Number of prompts must match number of images."}
    if image_urls and images:
        return {"error": "Provide either image URLs or uploaded images, not both."}
    if image_urls:
        async def download_image(url):
            loop = asyncio.get_event_loop()
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Referer": url,
            }
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(url, headers=headers)
            )
            if response.status_code == 200:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp_file.write(response.content)
                temp_file.close()
                return UploadFile(
                    filename=os.path.basename(temp_file.name),
                    file=open(temp_file.name, "rb")
                )
            else:
                raise Exception(f"Failed to download image from {url}: {response}")
        
        downloaded = await asyncio.gather(*(download_image(url) for url in image_urls))
        images.extend([img for img in downloaded if img is not None])
        
    # Get object name based on descriptions
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(
            asyncio.gather(*(get_object_name(d) for d in prompts))
        )
        object_names = [name.content.strip() for name in results]
    finally:
        loop.close()


    results = {}
    id = 0
    for image, object_name in zip(images, object_names):
        logger.info(f"Image filename: {image.filename}, object_name: {object_name}")
        
        masks, bounds, scores = segmentor.segment_image(
            image=Image.open(image.file),
            prompt=object_name
        )
        logger.info(f"Obtained {len(masks)} masks with scores : {scores}.")
        
        keyname = f"{image.filename}"
        results[keyname] = []
        for mask, bound, score in zip(masks, bounds, scores):
            id += 1
            color_palette = get_image_palette_with_mask(
                img=Image.open(image.file), 
                binary_mask=mask[0].cpu().numpy()
            )
            results[keyname].append(
                {
                    "id": str(id),
                    "name": object_name,
                    "score": score,
                    "bounds": bound,
                    #"mask": mask[0],
                    "colors": color_palette,
                }
            )
    
    return {"response": results}