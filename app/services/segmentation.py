# app/services/segmentation.py

import cv2
import numpy as np
from typing import List
from app.models.schemas import Plant, Leaf

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess an image from a file path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Preprocessed image array.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    # You can add more preprocessing steps here if needed
    return image

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess a video frame.

    Args:
        frame (np.ndarray): The video frame.

    Returns:
        np.ndarray: Preprocessed frame.
    """
    # You can add more preprocessing steps here if needed
    return frame

def segment_total_plant_area(image: np.ndarray) -> List[Plant]:
    """
    Segment the image to find plants and calculate their total area.

    Args:
        image (np.ndarray): The preprocessed image.

    Returns:
        List[Plant]: List of Plant objects with calculated areas.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold to create binary image
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # Remove noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    plants = []
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 1000:  # Threshold to filter out small areas
            continue
        plant = Plant(plant_id=idx + 1, plant_area=area)
        plants.append(plant)
    return plants

def segment_individual_leaves(image: np.ndarray) -> List[Plant]:
    """
    Segment the image to find individual leaves and calculate their areas.

    Args:
        image (np.ndarray): The preprocessed image.

    Returns:
        List[Plant]: List of Plant objects with associated leaves.
    """
    # First, segment the plants
    plants = segment_total_plant_area(image)
    # Create a mask for the entire image
    for plant in plants:
        # Assuming we have the contours from total plant area segmentation
        # Create a mask for the current plant
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # For simplicity, we assume plant contours are available
        # Here, you should extract the contour corresponding to the current plant
        # For demonstration, we will assume one plant contour
        # TODO: Replace with actual plant contour extraction
        # For now, we'll use the entire image
        mask[...] = 255
        # Apply mask to the image
        plant_image = cv2.bitwise_and(image, image, mask=mask)
        # Convert to grayscale
        gray = cv2.cvtColor(plant_image, cv2.COLOR_BGR2GRAY)
        # Threshold to create binary image for leaves
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Find leaf contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        leaves = []
        for leaf_idx, contour in enumerate(contours):
            leaf_area = cv2.contourArea(contour)
            if leaf_area < 500:  # Threshold to filter out small areas
                continue
            leaf = Leaf(leaf_id=leaf_idx + 1, leaf_area=leaf_area)
            leaves.append(leaf)
        plant.leaves = leaves
    return plants
