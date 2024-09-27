import cv2
import numpy as np
from typing import List
from plant_growth_tracker.models.schemas import Plant, Leaf
from plant_growth_tracker.models.custom_model import CustomSAMModel

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

def segment_individual_leaves(image: np.ndarray, custom_sam_model: CustomSAMModel) -> List[Plant]:
    """
    Segment the image using the custom SAM model to find individual leaves.

    Args:
        image (np.ndarray): The preprocessed image.
        custom_sam_model (CustomSAMModel): An instance of the custom trained SAM model.

    Returns:
        List[Plant]: List of Plant objects with associated leaves.
    """
    # Predict masks using the custom model
    masks = custom_sam_model.predict(image)  # Shape: (batch_size, num_masks, H, W)

    # Assuming batch_size is 1
    masks = masks[0]  # Shape: (num_masks, H, W)

    leaves = []
    for idx, mask in enumerate(masks):
        # Convert mask to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            leaf_area = cv2.contourArea(contour)
            if leaf_area < 500:  # Adjust threshold as needed
                continue
            leaf = Leaf(leaf_id=len(leaves) + 1, leaf_area=leaf_area)
            leaves.append(leaf)

    # Create a single Plant object since we are focusing on leaves
    plant = Plant(plant_id=1, plant_area=sum([leaf.leaf_area for leaf in leaves]), leaves=leaves)
    return [plant]