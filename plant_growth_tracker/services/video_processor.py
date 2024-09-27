import cv2
from typing import List, Dict, Any, Optional, Callable
from plant_growth_tracker.services.segmentation import (
    segment_total_plant_area,
    segment_individual_leaves,
)
from plant_growth_tracker.core.utils import default_preprocess_frame

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
) -> List[Dict[str, Any]]:
    """
    Processes a video to calculate individual leaf areas in each frame.

    Args:
        video_path (str): Path to the video file.
        preprocessing_function (Callable): Optional preprocessing function.

    Returns:
        List[Dict[str, Any]]: List of results for each frame, plant, and leaf.
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
        plants = segment_individual_leaves(preprocessed_frame)
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
