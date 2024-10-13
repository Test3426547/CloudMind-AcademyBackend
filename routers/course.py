from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.course_service import CourseService, get_course_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class CourseCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=10, max_length=1000)
    topics: List[str] = Field(..., min_items=1)
    duration: int = Field(..., gt=0)  # in minutes

class CourseUpdate(CourseCreate):
    pass

class ProgressUpdate(BaseModel):
    progress: float = Field(..., ge=0, le=100)

@router.post("/courses")
async def create_course(
    course_data: CourseCreate,
    user: User = Depends(oauth2_scheme),
    course_service: CourseService = Depends(get_course_service),
):
    try:
        result = await course_service.create_course(course_data.dict())
        logger.info(f"Course created successfully: {result['course_id']}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in create_course: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in create_course: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while creating the course")

@router.get("/courses/{course_id}")
async def get_course(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    course_service: CourseService = Depends(get_course_service),
):
    try:
        course = await course_service.get_course(course_id)
        logger.info(f"Course retrieved successfully: {course_id}")
        return course
    except HTTPException as e:
        logger.warning(f"HTTP error in get_course: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_course: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving the course")

@router.put("/courses/{course_id}")
async def update_course(
    course_id: str,
    course_data: CourseUpdate,
    user: User = Depends(oauth2_scheme),
    course_service: CourseService = Depends(get_course_service),
):
    try:
        result = await course_service.update_course(course_id, course_data.dict())
        logger.info(f"Course updated successfully: {course_id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in update_course: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in update_course: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating the course")

@router.delete("/courses/{course_id}")
async def delete_course(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    course_service: CourseService = Depends(get_course_service),
):
    try:
        result = await course_service.delete_course(course_id)
        logger.info(f"Course deleted successfully: {course_id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in delete_course: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in delete_course: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while deleting the course")

@router.get("/courses/recommend")
async def recommend_courses(
    num_recommendations: int = 5,
    user: User = Depends(oauth2_scheme),
    course_service: CourseService = Depends(get_course_service),
):
    try:
        recommendations = await course_service.recommend_courses(user.id, num_recommendations)
        logger.info(f"Course recommendations generated for user: {user.id}")
        return {"recommendations": recommendations}
    except HTTPException as e:
        logger.warning(f"HTTP error in recommend_courses: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in recommend_courses: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while recommending courses")

@router.put("/courses/{course_id}/progress")
async def update_progress(
    course_id: str,
    progress_data: ProgressUpdate,
    user: User = Depends(oauth2_scheme),
    course_service: CourseService = Depends(get_course_service),
):
    try:
        result = await course_service.update_user_progress(user.id, course_id, progress_data.progress)
        logger.info(f"User progress updated for course {course_id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in update_progress: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in update_progress: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating user progress")

@router.get("/courses/progress")
async def get_user_progress(
    user: User = Depends(oauth2_scheme),
    course_service: CourseService = Depends(get_course_service),
):
    try:
        progress = await course_service.get_user_progress(user.id)
        logger.info(f"User progress retrieved for user: {user.id}")
        return {"user_progress": progress}
    except HTTPException as e:
        logger.warning(f"HTTP error in get_user_progress: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_user_progress: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving user progress")
