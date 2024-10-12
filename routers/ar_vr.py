from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.ar_vr_service import ARVRService, get_ar_vr_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class ImageData(BaseModel):
    image_data: str = Field(..., min_length=1)

class RoomData(BaseModel):
    room_id: str = Field(..., min_length=1)
    dimensions: Dict[str, float]

class GestureData(BaseModel):
    gesture_points: List[Dict[str, float]]

class SceneData(BaseModel):
    scene_id: str = Field(..., min_length=1)
    objects: List[Dict[str, Any]]

@router.post("/ar/recognize-objects")
async def recognize_objects(
    image_data: ImageData,
    user: User = Depends(oauth2_scheme),
    ar_vr_service: ARVRService = Depends(get_ar_vr_service),
):
    try:
        result = await ar_vr_service.recognize_objects(image_data.image_data)
        logger.info(f"Object recognition completed for user {user.id}")
        return {"recognized_objects": result}
    except HTTPException as e:
        logger.error(f"HTTP error in recognize_objects: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in recognize_objects: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during object recognition")

@router.post("/vr/generate-spatial-map")
async def generate_spatial_map(
    room_data: RoomData,
    user: User = Depends(oauth2_scheme),
    ar_vr_service: ARVRService = Depends(get_ar_vr_service),
):
    try:
        result = await ar_vr_service.generate_spatial_map(room_data.dict())
        logger.info(f"Spatial map generated for user {user.id}")
        return {"spatial_map": result}
    except HTTPException as e:
        logger.error(f"HTTP error in generate_spatial_map: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in generate_spatial_map: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during spatial map generation")

@router.post("/ar/recognize-gesture")
async def recognize_gesture(
    gesture_data: GestureData,
    user: User = Depends(oauth2_scheme),
    ar_vr_service: ARVRService = Depends(get_ar_vr_service),
):
    try:
        result = await ar_vr_service.recognize_gesture(gesture_data.dict())
        logger.info(f"Gesture recognition completed for user {user.id}")
        return {"recognized_gesture": result}
    except HTTPException as e:
        logger.error(f"HTTP error in recognize_gesture: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in recognize_gesture: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during gesture recognition")

@router.post("/ar/generate-overlay")
async def generate_ar_overlay(
    scene_data: SceneData,
    user: User = Depends(oauth2_scheme),
    ar_vr_service: ARVRService = Depends(get_ar_vr_service),
):
    try:
        result = await ar_vr_service.generate_ar_overlay(scene_data.dict())
        logger.info(f"AR overlay generated for user {user.id}")
        return {"ar_overlay": result}
    except HTTPException as e:
        logger.error(f"HTTP error in generate_ar_overlay: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in generate_ar_overlay: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during AR overlay generation")
