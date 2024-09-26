# app/services/image_processor.py

from typing import List, Dict, Any
from app.services.segmentation import (
    preprocess_image,
    segment_total_plant_area,
    segment_individual_leaves,
)
from app.models.schemas import Plant, Leaf
import os

def process_total_plant_area_images(image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Process images to perform total plant area segmentation.

    Args:
        image_paths (List[str]): List of image file paths.

    Returns:
        List[Dict[str, Any]]: Segmentation results for each image.
    """
    results = []
    for image_path in image_paths:
        image = preprocess_image(image_path)
        plants = segment_total_plant_area(image)
        for plant in plants:
            result = {
                'image_name': os.path.basename(image_path),
                'plant_id': plant.plant_id,
                'plant_area': plant.plant_area,
            }
            results.append(result)
    return results

def process_individual_leaf_area_images(image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Process images to perform individual leaf area segmentation.

    Args:
        image_paths (List[str]): List of image file paths.

    Returns:
        List[Dict[str, Any]]: Segmentation results for each leaf in each image.
    """
    results = []
    for image_path in image_paths:
        image = preprocess_image(image_path)
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
