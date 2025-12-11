import os
from typing import Annotated
import asyncio
import requests
import tempfile
import json

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage

from src.bria import generate_image_with_fibo
from src.logger import get_logger

logger = get_logger(__name__)

image_studio_key = os.getenv("IMAGE_STUDIO_KEY")
image_studio_endpoint = os.getenv("IMAGE_STUDIO_ENDPOINT")




@tool()
def generate_image(
    prompt: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """
    Generate an image based on a description.
    Segment the image based on the objects described in the structured prompt.
    Detect the colors
    
    Args:
        prompt (str): The text prompt to generate the image.
    Returns:
        str: Message indicating the image generation status.
    """
    results = generate_image_with_fibo(prompt)
    
    if len(results.get("result_urls", [])) == 0:
        image_url = None
        structured_prompt = None
        message = "No image URL returned from BRIA API."
    else:
        if len(results.get("result_urls"))>1:
            logger.warning("More than one image generated, only the first one will be used.")
        
        image_url = results.get("result_urls")[0]
        structured_prompt = results["structured_prompt"]
        message = "Image generated successfully."
    
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=message,
                    tool_call_id=tool_call_id,
                    name="generate_image"
                )
            ],
            "image_url": image_url,
            "structured_prompt": structured_prompt,
        }
    )