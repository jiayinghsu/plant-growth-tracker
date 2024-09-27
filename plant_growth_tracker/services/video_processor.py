import cv2
from typing import List, Dict, Any, Optional, Callable
from plant_growth_tracker.services.segmentation import (
    segment_total_plant_area,
    segment_individual_leaves,
)
from plant_growth_tracker.core.utils import default_preprocess_frame
import torch

def process_total_plant_area_video(
    video_path: str,
    preprocessing_function: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """
    Processes a video to calculate total plant area in each frame.

    Args:
        video_path (str): Path to the video file.
        preprocessing_function (Callable): Optional preprocessing function.

    Returns:
        List[Dict[str, Any]]: List of results for each frame and plant.
    """
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        if preprocessing_function is not None:
            preprocessed_frame = preprocessing_function(frame)
        else:
            preprocessed_frame = default_preprocess_frame(frame)
        plants = segment_total_plant_area(preprocessed_frame)
        for plant in plants:
            result = {
                'frame_number': frame_number,
                'plant_id': plant.plant_id,
                'plant_area': plant.plant_area,
            }
            results.append(result)
    cap.release()
    return results

def process_individual_leaf_area_video(
    video_path: str,
    preprocessing_function: Optional[Callable] = None,
    custom_model_paths: dict = None,
) -> List[Dict[str, Any]]:
    """
    Processes a video to calculate individual leaf areas in each frame.

    Args:
        video_path (str): Path to the video file.
        preprocessing_function (Callable): Optional preprocessing function.
        custom_model_paths (dict): Dictionary containing 'model_path' and 'processor_path'.

    Returns:
        List[Dict[str, Any]]: List of results for each frame, plant, and leaf.
    """
    if custom_model_paths is None:
        raise ValueError("custom_model_paths must be provided for individual leaf area processing.")

    from plant_growth_tracker.models.custom_model import CustomSAMModel
    # Initialize the custom SAM model
    custom_sam_model = CustomSAMModel(
        model_path=custom_model_paths['model_path'],
        processor_path=custom_model_paths['processor_path'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    cap = cv2.VideoCapture(video_path)
    results = []
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        if preprocessing_function is not None:
            preprocessed_frame = preprocessing_function(frame)
        else:
            preprocessed_frame = default_preprocess_frame(frame)
        plants = segment_individual_leaves(preprocessed_frame, custom_sam_model)
        for plant in plants:
            for leaf in plant.leaves:
                result = {
                    'frame_number': frame_number,
                    'plant_id': plant.plant_id,
                    'leaf_id': leaf.leaf_id,
                    'leaf_area': leaf.leaf_area,
                }
                results.append(result)
    cap.release()
    return results