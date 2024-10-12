from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.percipio_integration_service import PercipioIntegrationService, get_percipio_integration_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class UserInterests(BaseModel):
    interests: List[str] = Field(..., min_items=1)

@router.get("/percipio/courses")
async def get_percipio_courses(
    offset: int = 0,
    limit: int = 10,
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioIntegrationService = Depends(get_percipio_integration_service),
):
    try:
        courses = await percipio_service.get_courses(offset, limit)
        logger.info(f"Retrieved {len(courses)} Percipio courses for user {user.id}")
        return courses
    except Exception as e:
        logger.error(f"Error retrieving Percipio courses: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving Percipio courses")

@router.get("/percipio/user-progress")
async def get_user_progress(
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioIntegrationService = Depends(get_percipio_integration_service),
):
    try:
        progress = await percipio_service.get_user_progress(user.id)
        logger.info(f"Retrieved Percipio progress for user {user.id}")
        return progress
    except Exception as e:
        logger.error(f"Error retrieving user progress: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving user progress")

@router.post("/percipio/recommend-content")
async def recommend_content(
    user_interests: UserInterests,
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioIntegrationService = Depends(get_percipio_integration_service),
):
    try:
        recommendations = await percipio_service.recommend_content(user.id, user_interests.interests)
        logger.info(f"Generated content recommendations for user {user.id}")
        return recommendations
    except Exception as e:
        logger.error(f"Error recommending content: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while recommending content")

@router.get("/percipio/adaptive-learning-path/{target_course_id}")
async def generate_adaptive_learning_path(
    target_course_id: str,
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioIntegrationService = Depends(get_percipio_integration_service),
):
    try:
        learning_path = await percipio_service.generate_adaptive_learning_path(user.id, target_course_id)
        logger.info(f"Generated adaptive learning path for user {user.id} and target course {target_course_id}")
        return learning_path
    except Exception as e:
        logger.error(f"Error generating adaptive learning path: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the adaptive learning path")

@router.get("/percipio/estimate-difficulty/{content_id}")
async def estimate_content_difficulty(
    content_id: str,
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioIntegrationService = Depends(get_percipio_integration_service),
):
    try:
        difficulty_estimation = await percipio_service.estimate_content_difficulty(content_id)
        logger.info(f"Estimated difficulty for content {content_id}")
        return difficulty_estimation
    except Exception as e:
        logger.error(f"Error estimating content difficulty: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while estimating content difficulty")
