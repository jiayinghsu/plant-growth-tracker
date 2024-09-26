# app/api/endpoints.py

from fastapi import APIRouter, UploadFile, File
from typing import List
from app.services.image_processor import (
    process_total_plant_area_images,
    process_individual_leaf_area_images,
)
from app.services.video_processor import (
    process_total_plant_area_video,
    process_individual_leaf_area_video,
)
from app.core.utils import save_upload_file
import os

router = APIRouter()

# Endpoint for total plant area segmentation on a single image
@router.post("/api/v1/segment/plant-area/image")
async def segment_plant_area_image(file: UploadFile = File(...)):
    image_path = await save_upload_file(file)
    results = process_total_plant_area_images([image_path])
    os.remove(image_path)  # Clean up the uploaded file
    return results

# Endpoint for individual leaf area segmentation on a single image
@router.post("/api/v1/segment/leaf-area/image")
async def segment_leaf_area_image(file: UploadFile = File(...)):
    image_path = await save_upload_file(file)
    results = process_individual_leaf_area_images([image_path])
    os.remove(image_path)
    return results

# Endpoint for total plant area segmentation on multiple images
@router.post("/api/v1/segment/plant-area/images")
async def segment_plant_area_images(files: List[UploadFile] = File(...)):
    image_paths = [await save_upload_file(file) for file in files]
    results = process_total_plant_area_images(image_paths)
    for path in image_paths:
        os.remove(path)
    return results

# Endpoint for individual leaf area segmentation on multiple images
@router.post("/api/v1/segment/leaf-area/images")
async def segment_leaf_area_images(files: List[UploadFile] = File(...)):
    image_paths = [await save_upload_file(file) for file in files]
    results = process_individual_leaf_area_images(image_paths)
    for path in image_paths:
        os.remove(path)
    return results

# Endpoint for total plant area segmentation on a video
@router.post("/api/v1/segment/plant-area/video")
async def segment_plant_area_video(file: UploadFile = File(...)):
    video_path = await save_upload_file(file)
    results = process_total_plant_area_video(video_path)
    os.remove(video_path)
    return results

# Endpoint for individual leaf area segmentation on a video
@router.post("/api/v1/segment/leaf-area/video")
async def segment_leaf_area_video(file: UploadFile = File(...)):
    video_path = await save_upload_file(file)
    results = process_individual_leaf_area_video(video_path)
    os.remove(video_path)
    return results
