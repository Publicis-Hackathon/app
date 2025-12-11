import requests
import os

from src.logger import get_logger
logger = get_logger(__name__)

image_studio_key = os.getenv("IMAGE_STUDIO_KEY")
image_studio_endpoint = os.getenv("IMAGE_STUDIO_ENDPOINT")



def generate_image_with_fibo(prompt: str) -> str:
    """
    Generate an image using the BRIA API with a Fibonacci-themed prompt.
    
    Args:
        prompt (str): The text prompt to generate the image.
    Returns:
        str: Message indicating the image generation status.
    """
    payload = {
        "prompt": prompt,
        "sync": True,
    }

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": image_studio_key
    }

    if os.getenv("TESTING_ENVIRONMENT").lower() in ("1", "true"):
            logger.info("Using mock response for BRIA API in testing environment.")
            import ast
            image_url = "https://www.thesprucepets.com/thmb/rW016PPM5VrD2cAJnMId-W1mLmM=/3865x0/filters:no_upscale():strip_icc()/close-up-of-gold-and-blue-macaw-perching-on-tree-962288862-5b50073e46e0fb0037c23c23.jpg"
            prompt = "blue parrot on a tree branch"
            results = ast.literal_eval("""{\'request_id\': \'1146a76e654544aca44942f358481e38\', \'task_id\': \'1146a76e654544aca44942f358481e38\', \'structured_prompt\': \'{"short_description": "A vibrant blue parrot with striking yellow eyes perches gracefully on a moss-covered branch in the lush, sun-dappled canopy of a tropical rainforest. Its brilliant plumage stands out against the rich greens of the foliage, creating a captivating and exotic scene.", "objects": [{"description": "A vibrant blue parrot, likely a Hyacinth Macaw, with brilliant cobalt blue feathers, a strong curved black beak, and distinctive bright yellow rings around its eyes. Its feathers show a slight iridescence.", "location": "center-right", "relationship": "The parrot is the main subject, perched on a branch that extends from the left.", "relative_size": "large within frame", "shape_and_color": "Elongated, avian shape, primarily vibrant blue with yellow accents.", "texture": "Feathery, smooth plumage.", "appearance_details": "Long tail feathers trail downwards, and its claws firmly grip the branch.", "orientation": "facing slightly left, head turned towards the viewer"}, {"description": "A sturdy tree branch, thick and gnarled, covered in soft, verdant moss and small patches of lichen. It serves as a natural perch for the parrot.", "location": "bottom-center to mid-left", "relationship": "The branch supports the parrot and forms part of its immediate environment.", "relative_size": "medium", "shape_and_color": "Irregular, cylindrical shape, primarily dark brown with green moss.", "texture": "Rough bark, soft moss.", "appearance_details": "Some dew drops glisten on the moss.", "orientation": "horizontal, slightly angled upwards from left to right"}], "background_setting": "A dense, vibrant tropical rainforest canopy, with a soft blur of various shades of green foliage, dappled sunlight filtering through the leaves, and hints of distant, darker tree trunks. The air appears humid and alive.", "lighting": {"conditions": "Bright, dappled sunlight", "direction": "Top-down, filtering through canopy", "shadows": "Soft, diffused shadows cast by leaves and the parrot itself, creating depth."}, "aesthetics": {"composition": "Rule of thirds, with the parrot positioned at intersection points, portrait composition.", "color_scheme": "Complementary, primarily vibrant blues against rich greens, with warm yellow accents.", "mood_atmosphere": "Exotic, lively, serene, and natural.", "preference_score": "very high", "aesthetic_score": "very high"}, "photographic_characteristics": {"depth_of_field": "Shallow, with the background softly blurred to emphasize the parrot.", "focus": "Sharp focus on the parrot\\\'s eyes and head.", "camera_angle": "Eye-level with the parrot.", "lens_focal_length": "Portrait lens (e.g., 85mm)"}, "style_medium": "photograph", "context": "This is a concept for a high-quality wildlife photograph, suitable for a nature documentary, a magazine cover, or environmental educational material, showcasing the beauty of tropical birds.", "artistic_style": "realistic"}\', \'seed\': 338420435, \'result_urls\': [\'https://www.thesprucepets.com/thmb/rW016PPM5VrD2cAJnMId-W1mLmM=/3865x0/filters:no_upscale():strip_icc()/close-up-of-gold-and-blue-macaw-perching-on-tree-962288862-5b50073e46e0fb0037c23c23.jpg']}""")
    else:
        response = requests.post(
            image_studio_endpoint, 
            json=payload, 
            headers=headers
        )
        
        logger.info(f"BRIA API response: {response.json()}")
        results = response.json().get("result")
    
    return results