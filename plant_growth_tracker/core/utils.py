import cv2
import numpy as np
from PIL import Image
import pillow_heif
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

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

def default_preprocess_image(image: np.ndarray, visualize: bool = False) -> np.ndarray:
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
        visualize (bool): If True, display intermediate steps for debugging.

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

    if visualize:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(binary_mask, cmap='gray')
        plt.title('Binary Mask')
        plt.axis('off')
        plt.show()

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

    if visualize:
        plt.figure(figsize=(6, 6))
        plt.imshow(filtered_mask, cmap='gray')
        plt.title('Filtered Mask')
        plt.axis('off')
        plt.show()

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

    if visualize:
        plt.figure(figsize=(6, 6))
        plt.imshow(cropped_image)
        plt.title('Cropped Image')
        plt.axis('off')
        plt.show()

    # Step 7: Resize the image to a standard size
    desired_size = (1000, 1000)  # Adjust as needed
    resized_image = cv2.resize(cropped_image, desired_size, interpolation=cv2.INTER_LINEAR)

    if visualize:
        plt.figure(figsize=(6, 6))
        plt.imshow(resized_image)
        plt.title('Resized Image')
        plt.axis('off')
        plt.show()

    # Log the preprocessing steps
    logging.info(f"Image resized to {desired_size} pixels.")

    return resized_image

def default_preprocess_frame(frame: np.ndarray, visualize: bool = False) -> np.ndarray:
    """
    Default preprocessing for video frames.

    Args:
        frame (np.ndarray): The video frame.
        visualize (bool): If True, display intermediate steps for debugging.

    Returns:
        np.ndarray: The preprocessed frame.
    """
    # Use the same preprocessing as for images
    return default_preprocess_image(frame, visualize=visualize)
