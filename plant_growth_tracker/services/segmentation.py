import cv2
import numpy as np
from typing import List
from ..models.schemas import Plant, Leaf

def segment_total_plant_area(image: np.ndarray) -> List[Plant]:
    """
    Segment the image to find plants and calculate their total area.

    Args:
        image (np.ndarray): The preprocessed image.

    Returns:
        List[Plant]: List of Plant objects with calculated areas.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Threshold to create binary image
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # Remove noise and fill holes
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Find contours of plants
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    plants = []
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 1000:  # Adjust threshold as needed
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
    # Create a mask for each plant and segment leaves within it
    for plant in plants:
        # For demonstration purposes, we're using the entire image
        # In practice, extract the specific region corresponding to the plant
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # Apply mask (here, the mask is the entire image)
        mask[...] = 255
        plant_image = cv2.bitwise_and(image, image, mask=mask)
        # Convert to grayscale
        gray = cv2.cvtColor(plant_image, cv2.COLOR_RGB2GRAY)
        # Edge detection or adaptive thresholding for leaf segmentation
        edges = cv2.Canny(gray, 50, 150)
        # Dilate edges to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        # Find contours of leaves
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        leaves = []
        for leaf_idx, contour in enumerate(contours):
            leaf_area = cv2.contourArea(contour)
            if leaf_area < 500:  # Adjust threshold as needed
                continue
            leaf = Leaf(leaf_id=leaf_idx + 1, leaf_area=leaf_area)
            leaves.append(leaf)
        plant.leaves = leaves
    return plants
