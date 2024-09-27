from plant_growth_tracker.services.image_processor import (
    process_total_plant_area_images,
    process_individual_leaf_area_images,
)
from plant_growth_tracker.services.video_processor import (
    process_total_plant_area_video,
    process_individual_leaf_area_video,
)

def process_images(
    image_folder_path: str,
    segmentation_type: str = 'total_plant_area',
    preprocessing_function=None,
    custom_model_paths: dict = None
):
    """
    Processes images in a folder and returns the results as a DataFrame.

    Args:
        image_folder_path (str): Path to the folder containing images.
        segmentation_type (str): 'total_plant_area' or 'individual_leaf_area'.
        preprocessing_function (Callable): Optional preprocessing function.
        custom_model_paths (dict): Optional paths for custom model and processor. Required if segmentation_type is 'individual_leaf_area'.

    Returns:
        pandas.DataFrame: DataFrame containing the results.
    """
    import os
    from typing import List
    import pandas as pd

    # Get list of image paths
    image_paths = [
        os.path.join(image_folder_path, f)
        for f in os.listdir(image_folder_path)
        if os.path.isfile(os.path.join(image_folder_path, f))
    ]

    if segmentation_type == 'total_plant_area':
        results = process_total_plant_area_images(
            image_paths, preprocessing_function
        )
    elif segmentation_type == 'individual_leaf_area':
        if custom_model_paths is None:
            raise ValueError("custom_model_paths must be provided for individual_leaf_area segmentation.")
        results = process_individual_leaf_area_images(
            image_paths, preprocessing_function, custom_model_paths
        )
    else:
        raise ValueError(f"Invalid segmentation_type: {segmentation_type}")

    df = pd.DataFrame(results)
    return df

def process_video(
    video_path: str,
    segmentation_type: str = 'total_plant_area',
    preprocessing_function=None,
    custom_model_paths: dict = None
):
    """
    Processes a video and returns the results as a DataFrame.

    Args:
        video_path (str): Path to the video file.
        segmentation_type (str): 'total_plant_area' or 'individual_leaf_area'.
        preprocessing_function (Callable): Optional preprocessing function.
        custom_model_paths (dict): Optional paths for custom model and processor. Required if segmentation_type is 'individual_leaf_area'.

    Returns:
        pandas.DataFrame: DataFrame containing the results.
    """
    import pandas as pd

    if segmentation_type == 'total_plant_area':
        results = process_total_plant_area_video(
            video_path, preprocessing_function
        )
    elif segmentation_type == 'individual_leaf_area':
        if custom_model_paths is None:
            raise ValueError("custom_model_paths must be provided for individual_leaf_area segmentation.")
        results = process_individual_leaf_area_video(
            video_path, preprocessing_function, custom_model_paths
        )
    else:
        raise ValueError(f"Invalid segmentation_type: {segmentation_type}")

    df = pd.DataFrame(results)
    return df