from typing import List, Dict, Any, Optional, Callable
import os
from plant_growth_tracker.services.segmentation import (
    segment_total_plant_area,
    segment_individual_leaves,
)
from plant_growth_tracker.core.utils import load_image, default_preprocess_image

def process_total_plant_area_images(
    image_paths: List[str],
    preprocessing_function: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """
    Processes images to calculate total plant area.

    Args:
        image_paths (List[str]): List of image file paths.
        preprocessing_function (Callable): Optional preprocessing function.

    Returns:
        List[Dict[str, Any]]: List of results for each plant.
    """
    results = []
    for image_path in image_paths:
        image = load_image(image_path)
        if preprocessing_function is not None:
            image = preprocessing_function(image)
        else:
            image = default_preprocess_image(image)
        plants = segment_total_plant_area(image)
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
    preprocessing_function: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """
    Processes images to calculate individual leaf areas.

    Args:
        image_paths (List[str]): List of image file paths.
        preprocessing_function (Callable): Optional preprocessing function.

    Returns:
        List[Dict[str, Any]]: List of results for each leaf.
    """
    results = []
    for image_path in image_paths:
        image = load_image(image_path)
        if preprocessing_function is not None:
            image = preprocessing_function(image)
        else:
            image = default_preprocess_image(image)
        plants = segment_individual_leaves(image)
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
