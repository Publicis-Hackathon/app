from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from PIL import Image
from typing import List, Tuple
import torch

from src.logger import get_logger
logger = get_logger(__name__)



class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
    

class SamSegmentor(metaclass=Singleton):
    def __init__(
        self,
        device: torch.device | None = None
    ):
        logger.info("Initializing SamSegmentor")
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def segment_image(self, image: Image.Image, prompt: str):
        self.processor.reset_all_prompts({})
        inference_state = self.processor.set_image(image)
        # Prompt the model with text
        self.processor.reset_all_prompts(inference_state)
        output = self.processor.set_text_prompt(
            state=inference_state, 
            prompt=prompt
        )

        # Get the masks, bounding boxes, and scores
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        return masks, SamSegmentor.boxes_to_bounds(boxes), scores

    
    def segment_batch(
        self,
        images: List[Image.Image],
        prompts: List[str],
    ) -> List[Tuple]:
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length")
        
        inputs = self.processor(
            images=[image.convert("RGB") for image in images], 
            text=prompts, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results for both images
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )
        
        return results


    
    @staticmethod
    def boxes_to_bounds(boxes):
        """
        Convert bounding boxes from [x1, y1, x2, y2] format to bounds format.
        
        Args:
            boxes: torch.Tensor of shape (N, 4) with [x1, y1, x2, y2] coordinates
            
        Returns:
            list of dicts with format {'x': x, 'y': y, 'width': width, 'height': height}
        """
        bounds_list = []
        for box in boxes:
            x1, y1, x2, y2 = box.cpu().tolist()
            bounds = {
                'x': int(x1),
                'y': int(y1),
                'width': int(x2 - x1),
                'height': int(y2 - y1)
            }
            bounds_list.append(bounds)
        return bounds_list