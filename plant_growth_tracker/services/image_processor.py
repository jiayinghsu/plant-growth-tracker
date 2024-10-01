from typing import List, Dict, Any, Optional, Callable
import os
import torch
from plant_growth_tracker.services.segmentation import (
    segment_total_plant_area,
    segment_individual_leaves,
)
from plant_growth_tracker.core.utils import load_image, default_preprocess_image
from plant_growth_tracker.models.custom_model import CustomSAMModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def process_total_plant_area_images(
    image_paths: List[str],
    preprocessing_function: Optional[Callable[[Any], Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Processes images to calculate total plant area.

    Args:
        image_paths (List[str]): List of image file paths.
        preprocessing_function (Callable, optional): Custom preprocessing function.

    Returns:
        List[Dict[str, Any]]: List of results for each plant.
    """
    results = []
    for image_path in image_paths:
        image = load_image(image_path)
        if preprocessing_function:
            image = preprocessing_function(image)
            logging.debug(f"Applied custom preprocessing to {os.path.basename(image_path)}")
        else:
            image = default_preprocess_image(image)
            logging.debug(f"Applied default preprocessing to {os.path.basename(image_path)}")
        plants = segment_total_plant_area(image)
        logging.info(f"Image: {os.path.basename(image_path)}, Detected Plants: {len(plants)}")
        for plant in plants:
            result = {
                'image_name': os.path.basename(image_path),
                'plant_id': plant.plant_id,
                'plant_area': plant.plant_area,
            }
            results.append(result)
    return results

def process_individual_leaf_area_images(
    image_paths: List[str],
    preprocessing_function: Optional[Callable[[Any], Any]] = None,
    custom_model_paths: dict = None,
) -> List[Dict[str, Any]]:
    """
    Processes images to calculate individual leaf areas using the custom SAM model.

    Args:
        image_paths (List[str]): List of image file paths.
        preprocessing_function (Callable, optional): Custom preprocessing function.
        custom_model_paths (dict): Dictionary containing 'model_path' and 'processor_path'.

    Returns:
        List[Dict[str, Any]]: List of results for each leaf.
    """
    if custom_model_paths is None:
        raise ValueError("custom_model_paths must be provided for individual leaf area processing.")

    
    # Initialize the custom SAM model
    custom_sam_model = CustomSAMModel(
        model_path=custom_model_paths['model_path'],
        processor_path=custom_model_paths['processor_path'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    results = []
    for image_path in image_paths:
        image = load_image(image_path)
        if preprocessing_function:
            image = preprocessing_function(image)
            logging.debug(f"Applied custom preprocessing to {os.path.basename(image_path)}")
        else:
            image = default_preprocess_image(image)
            logging.debug(f"Applied default preprocessing to {os.path.basename(image_path)}")
        plants = segment_individual_leaves(image, custom_sam_model)
        logging.info(f"Image: {os.path.basename(image_path)}, Detected Plants: {len(plants)}")
        for plant in plants:
            for leaf in plant.leaves:
                result = {
                    'image_name': os.path.basename(image_path),
                    'plant_id': plant.plant_id,
                    'leaf_id': leaf.leaf_id,
                    'leaf_area': leaf.leaf_area,
                }
                results.append(result)
    return results