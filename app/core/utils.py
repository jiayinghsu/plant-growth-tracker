# app/core/utils.py

import os
import shutil
from fastapi import UploadFile
import uuid

async def save_upload_file(upload_file: UploadFile, destination: str = "data/temp") -> str:
    """
    Save the uploaded file to the specified destination directory.

    Args:
        upload_file (UploadFile): The uploaded file from the client.
        destination (str): The directory where the file will be saved.

    Returns:
        str: The path to the saved file.
    """
    if not os.path.exists(destination):
        os.makedirs(destination)
    filename = f"{uuid.uuid4()}_{upload_file.filename}"
    file_path = os.path.join(destination, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path
