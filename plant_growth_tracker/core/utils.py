import cv2
import numpy as np
from PIL import Image
import pillow_heif
from skimage.filters import threshold_otsu

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
    Default preprocessing for images:
    - Convert to RGB if needed
    - Convert to grayscale
    - Perform Otsu's thresholding to create a binary mask
    - Remove background using the mask
    - Remove small objects/noise
    - Find bounding box of significant regions and crop the image
    - Resize the image to a standard size

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The preprocessed image.
    """
    # Step 1: Convert to RGB if needed
    if len(image.shape) == 2 or image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Step 3: Apply Otsu's thresholding
    thresh_value = threshold_otsu(gray)
    binary_mask = gray < thresh_value  # Invert if plant is darker than background
    
    # Step 4: Remove small objects/noise
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    min_area = 1000  # Minimum area threshold; adjust based on your images
    filtered_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):  # Skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_mask[labels == i] = True
    
    # Step 5: Apply the filtered mask to the image
    filtered_image = image.copy()
    filtered_image[~filtered_mask] = 0  # Set background pixels to zero
    
    # Step 6: Find bounding box of significant regions and crop the image
    coords = cv2.findNonZero(filtered_mask.astype(np.uint8))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped_image = filtered_image[y:y+h, x:x+w]
    else:
        # If no significant regions are found, return the original image
        cropped_image = filtered_image
    
    # Step 7: Resize the image to a standard size
    desired_size = (2000, 2000)
    resized_image = cv2.resize(cropped_image, desired_size, interpolation=cv2.INTER_LINEAR)
    
    return resized_image

def default_preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Default preprocessing for video frames.

    Args:
        frame (np.ndarray): The video frame.

    Returns:
        np.ndarray: The preprocessed frame.
    """
    # Use the same preprocessing as for images
    return default_preprocess_image(frame)
