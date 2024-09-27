import os
from PIL import Image
import pillow_heif
import numpy as np
import cv2

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from a file path, handling various formats including HEIC.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image as a NumPy array.
    """
    if image_path.lower().endswith('.heic'):
        heif_file = pillow_heif.read_heif(image_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
        )
        image = np.array(image)
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at path: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def default_preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Default preprocessing for images (e.g., resizing, normalization).

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The preprocessed image.
    """
    # Default preprocessing steps
    # Convert to RGB if not already
    if len(image.shape) == 2 or image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Resize image if needed
    # image = cv2.resize(image, (desired_width, desired_height))
    return image

def default_preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Default preprocessing for video frames.

    Args:
        frame (np.ndarray): The video frame.

    Returns:
        np.ndarray: The preprocessed frame.
    """
    # Default preprocessing steps
    # Convert to RGB if needed
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame
