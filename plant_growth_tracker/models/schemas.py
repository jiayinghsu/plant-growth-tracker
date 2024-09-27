from pydantic import BaseModel
from typing import List, Optional

class Leaf(BaseModel):
    leaf_id: int
    leaf_area: float

class Plant(BaseModel):
    plant_id: int
    plant_area: float
    leaves: Optional[List[Leaf]] = []

class ImageResult(BaseModel):
    image_name: str
    plants: List[Plant]

class VideoFrameResult(BaseModel):
    frame_number: int
    plants: List[Plant]
