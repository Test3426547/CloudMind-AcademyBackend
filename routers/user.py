from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.user_service import UserService, get_user_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=5, max_length=100)
    age: int = Field(..., ge=0, le=120)

class UserUpdate(BaseModel):
    name: str = Field(None, min_length=1, max_length=100)
    email: str = Field(None, min_length=5, max_length=100)
    age: int = Field(None, ge=0, le=120)

class UserPreferences(BaseModel):
    learning_style: str = Field(None)
    preferred_subjects: List[str] = Field(None)
    difficulty_level: str = Field(None)

@router.post("/users")
async def create_user(
    user_data: UserCreate,
    user: User = Depends(oauth2_scheme),
    user_service: UserService = Depends(get_user_service),
):
    try:
        result = await user_service.create_user(user_data.dict())
        logger.info(f"User created successfully: {result['user_id']}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in create_user: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in create_user: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while creating the user")

@router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    user: User = Depends(oauth2_scheme),
    user_service: UserService = Depends(get_user_service),
):
    try:
        result = await user_service.get_user(user_id)
        logger.info(f"User retrieved successfully: {user_id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in get_user: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_user: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving the user")

@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    user: User = Depends(oauth2_scheme),
    user_service: UserService = Depends(get_user_service),
):
    try:
        result = await user_service.update_user(user_id, user_data.dict(exclude_unset=True))
        logger.info(f"User updated successfully: {user_id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in update_user: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in update_user: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating the user")

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    user: User = Depends(oauth2_scheme),
    user_service: UserService = Depends(get_user_service),
):
    try:
        result = await user_service.delete_user(user_id)
        logger.info(f"User deleted successfully: {user_id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in delete_user: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in delete_user: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while deleting the user")

@router.get("/users/{user_id}/recommend-courses")
async def recommend_courses(
    user_id: str,
    num_recommendations: int = 5,
    user: User = Depends(oauth2_scheme),
    user_service: UserService = Depends(get_user_service),
):
    try:
        recommendations = await user_service.recommend_courses(user_id, num_recommendations)
        logger.info(f"Course recommendations generated for user: {user_id}")
        return {"recommendations": recommendations}
    except HTTPException as e:
        logger.warning(f"HTTP error in recommend_courses: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in recommend_courses: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while recommending courses")

@router.get("/users/{user_id}/analyze-behavior")
async def analyze_user_behavior(
    user_id: str,
    user: User = Depends(oauth2_scheme),
    user_service: UserService = Depends(get_user_service),
):
    try:
        analysis = await user_service.analyze_user_behavior(user_id)
        logger.info(f"User behavior analyzed for user: {user_id}")
        return analysis
    except HTTPException as e:
        logger.warning(f"HTTP error in analyze_user_behavior: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in analyze_user_behavior: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while analyzing user behavior")

@router.put("/users/{user_id}/preferences")
async def update_user_preferences(
    user_id: str,
    preferences: UserPreferences,
    user: User = Depends(oauth2_scheme),
    user_service: UserService = Depends(get_user_service),
):
    try:
        result = await user_service.update_user_preferences(user_id, preferences.dict(exclude_unset=True))
        logger.info(f"User preferences updated for user: {user_id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in update_user_preferences: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in update_user_preferences: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating user preferences")
