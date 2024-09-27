import cv2
import numpy as np
from typing import List
from plant_growth_tracker.models.schemas import Plant, Leaf

def segment_total_plant_area(image: np.ndarray) -> List[Plant]:
    """
    Segment the image to find plants and calculate their total area.

    Args:
        image (np.ndarray): The preprocessed image.

    Returns:
        List[Plant]: List of Plant objects with calculated areas.
    """
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Threshold the image to create a binary mask
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
        # For simplicity, use the entire image as the mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # Create a binary mask where plant pixels are 255
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask = binary.copy()
        
        # Apply the mask to the image
        plant_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Convert to grayscale
        gray_plant = cv2.cvtColor(plant_image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection for leaf segmentation
        edges = cv2.Canny(gray_plant, 50, 150)
        
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
