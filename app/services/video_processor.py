# app/services/video_processor.py

import cv2
from typing import List, Dict, Any
from app.services.segmentation import (
    preprocess_frame,
    segment_total_plant_area,
    segment_individual_leaves,
)
from app.models.schemas import Plant, Leaf
import os

def process_total_plant_area_video(video_path: str) -> List[Dict[str, Any]]:
    """
    Process a video to perform total plant area segmentation on each frame.

    Args:
        video_path (str): Path to the video file.

    Returns:
        List[Dict[str, Any]]: Segmentation results for each frame.
    """
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        # You can adjust the frame skip rate here
        # if frame_number % N == 0:
        preprocessed_frame = preprocess_frame(frame)
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

def process_individual_leaf_area_video(video_path: str) -> List[Dict[str, Any]]:
    """
    Process a video to perform individual leaf area segmentation on each frame.

    Args:
        video_path (str): Path to the video file.

    Returns:
        List[Dict[str, Any]]: Segmentation results for each leaf in each frame.
    """
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        preprocessed_frame = preprocess_frame(frame)
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
