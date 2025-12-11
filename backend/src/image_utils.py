from collections import Counter
from typing import List, Tuple, Union
from numpy.typing import NDArray
import numpy as np
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
from urllib.parse import urlparse
from sklearn.cluster import KMeans

from src.logger import get_logger
logger = get_logger(__name__)

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert an RGB color tuple to a hexadecimal color code.

    Args:
        rgb (Tuple[int, int, int]): A tuple containing the red, green, and blue values of the color.

    Returns:
        str: The hexadecimal color code.
    """
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"




def quantize_image(
    image: Union[Image.Image, NDArray[np.uint8]], n_colors: int = 8
) -> Image.Image:
    """
    Quantize the colors in an image using KMeans clustering.

    Args:
        image (Union[Image.Image, np.ndarray]): Input image.
        n_colors (int): Number of colors to quantize to.

    Returns:
        Image.Image: Quantized image.
    """
    # Convert PIL Image to numpy array if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    h, w, c = image.shape
    flat_img = image.reshape(-1, c)

    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(flat_img)  # type: ignore[arg-type]
    quantized_flat = kmeans.cluster_centers_[labels].astype(np.uint8)  # type: ignore[arg-type]
    quantized_img = quantized_flat.reshape(h, w, c)
    return Image.fromarray(quantized_img)


def get_image_palette_with_mask(
    img: Image.Image, binary_mask: np.typing.NDArray[np.bool_], threshold: float = 0.05
) -> List[str]:
    """Returns the image palette as a list of hexcodes, excluding transparent pixels.

    Args:
        img: The image.
        binary_mask: The binary mask as a boolean numpy array. Only pixels where the array is True
            will be kept.
        threshold: The discard threshold. Colors that are accounting for less than this value
            multiplied by the number of pixels in the mask will be discarded.

    Returns:
        List[str]: The colors as hexcodes.
    """
    
    alpha_im = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
    
    
    
    img = img.copy()  # Copy to keep the original image unchanged
    img.putalpha(alpha_im)

    qt_img = np.array(quantize_image(img, 3))
    
    # Filter out transparent pixels
    non_transparent_pixels = qt_img[qt_img[..., 3] != 0][..., :3]

    # Get unique colors
    color_count = Counter(tuple(color) for color in non_transparent_pixels)

    discard_threshold = non_transparent_pixels.shape[0] * threshold

    # Convert to hex and sort by frequence (most frequent first)
    return [
        rgb_to_hex(rgb=color)
        for color, freq in sorted(color_count.items(), reverse=True, key=lambda x: x[1])
        if freq >= discard_threshold
    ]
    
    
def is_url(s: str) -> bool:
    parsed = urlparse(s)
    return parsed.scheme in ("http", "https") and parsed.netloc != ""


def load_image(source: str):
    logger.info(f"Loading image from source: {source}")
    if is_url(source):
        with urlopen(source) as response:
            data = response.read()
        return Image.open(BytesIO(data))
    else:
        return Image.open(source)