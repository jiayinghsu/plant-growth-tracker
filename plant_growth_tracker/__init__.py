# plant_growth_tracker/__init__.py

from .services.image_processor import (
    process_total_plant_area_images,
    process_individual_leaf_area_images,
)
from .services.video_processor import (
    process_total_plant_area_video,
    process_individual_leaf_area_video,
)
import pandas as pd
import os
from typing import Optional, Callable

def process_images(
    image_folder_path: str,
    output_csv: Optional[str] = None,
    segmentation_type: str = 'total_plant_area',
    preprocessing_function: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Process a folder of images for plant segmentation.

    Args:
        image_folder_path (str): Path to the folder containing images.
        output_csv (str, optional): Path to save the results as a CSV file.
        segmentation_type (str): 'total_plant_area' or 'individual_leaf_area'.
        preprocessing_function (Callable, optional): Custom preprocessing function.

    Returns:
        pd.DataFrame: DataFrame containing the segmentation results.
    """
    from .services.image_processor import (
        process_total_plant_area_images,
        process_individual_leaf_area_images,
    )

    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.heic')

    image_files = [
        os.path.join(image_folder_path, f)
        for f in os.listdir(image_folder_path)
        if f.lower().endswith(supported_formats)
    ]

    if not image_files:
        raise ValueError("No images found in the specified folder.")

    if segmentation_type == 'total_plant_area':
        results = process_total_plant_area_images(image_files, preprocessing_function)
    elif segmentation_type == 'individual_leaf_area':
        results = process_individual_leaf_area_images(image_files, preprocessing_function)
    else:
        raise ValueError("Invalid segmentation_type. Choose 'total_plant_area' or 'individual_leaf_area'.")

    df = pd.DataFrame(results)

    if output_csv:
        df.to_csv(output_csv, index=False)

    return df

def process_video(
    video_file_path: str,
    output_csv: Optional[str] = None,
    segmentation_type: str = 'total_plant_area',
    preprocessing_function: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Process a video file for plant segmentation.

    Args:
        video_file_path (str): Path to the video file.
        output_csv (str, optional): Path to save the results as a CSV file.
        segmentation_type (str): 'total_plant_area' or 'individual_leaf_area'.
        preprocessing_function (Callable, optional): Custom preprocessing function.

    Returns:
        pd.DataFrame: DataFrame containing the segmentation results.
    """
    from .services.video_processor import (
        process_total_plant_area_video,
        process_individual_leaf_area_video,
    )

    if not os.path.exists(video_file_path):
        raise ValueError("Video file does not exist.")

    if segmentation_type == 'total_plant_area':
        results = process_total_plant_area_video(video_file_path, preprocessing_function)
    elif segmentation_type == 'individual_leaf_area':
        results = process_individual_leaf_area_video(video_file_path, preprocessing_function)
    else:
        raise ValueError("Invalid segmentation_type. Choose 'total_plant_area' or 'individual_leaf_area'.")

    df = pd.DataFrame(results)

    if output_csv:
        df.to_csv(output_csv, index=False)

    return df
