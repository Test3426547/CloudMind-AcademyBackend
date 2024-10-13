from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.course_prerequisites_service import CoursePrerequisitesService, get_course_prerequisites_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class CourseAddRequest(BaseModel):
    course_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    prerequisites: List[str] = Field(default=[])
    topics: List[str] = Field(..., min_items=1)

class LearningPathAnalysisRequest(BaseModel):
    course_ids: List[str] = Field(..., min_items=2)

class PersonalizedLearningPathRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    target_course_id: str = Field(..., min_length=1)
    max_courses: int = Field(5, ge=1, le=10)

class UserProfileUpdateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    skills: List[str] = Field(..., min_items=1)
    completed_courses: List[str] = Field(default=[])

@router.post("/courses")
async def add_course(
    request: CourseAddRequest,
    user: User = Depends(oauth2_scheme),
    prerequisites_service: CoursePrerequisitesService = Depends(get_course_prerequisites_service),
):
    try:
        result = await prerequisites_service.add_course(
            request.course_id, request.title, request.description, request.prerequisites, request.topics
        )
        logger.info(f"Course {request.course_id} added successfully")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in add_course: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in add_course: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while adding the course")

@router.get("/courses/{course_id}/prerequisites")
async def get_prerequisites(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisites_service: CoursePrerequisitesService = Depends(get_course_prerequisites_service),
):
    try:
        prerequisites = await prerequisites_service.get_course_prerequisites(course_id)
        logger.info(f"Retrieved prerequisites for course {course_id}")
        return {"prerequisites": prerequisites}
    except HTTPException as e:
        logger.warning(f"HTTP error in get_prerequisites: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_prerequisites: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving prerequisites")

@router.get("/courses/{course_id}/suggest-prerequisites")
async def suggest_prerequisites(
    course_id: str,
    num_suggestions: int = 3,
    user: User = Depends(oauth2_scheme),
    prerequisites_service: CoursePrerequisitesService = Depends(get_course_prerequisites_service),
):
    try:
        suggestions = await prerequisites_service.suggest_prerequisites(course_id, num_suggestions)
        logger.info(f"Generated prerequisite suggestions for course {course_id}")
        return {"suggested_prerequisites": suggestions}
    except HTTPException as e:
        logger.warning(f"HTTP error in suggest_prerequisites: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in suggest_prerequisites: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while suggesting prerequisites")

@router.post("/learning-path/analyze")
async def analyze_learning_path(
    request: LearningPathAnalysisRequest,
    user: User = Depends(oauth2_scheme),
    prerequisites_service: CoursePrerequisitesService = Depends(get_course_prerequisites_service),
):
    try:
        analysis = await prerequisites_service.analyze_learning_path(request.course_ids)
        logger.info(f"Analyzed learning path for {len(request.course_ids)} courses")
        return analysis
    except HTTPException as e:
        logger.warning(f"HTTP error in analyze_learning_path: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in analyze_learning_path: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while analyzing the learning path")

@router.post("/learning-path/personalized")
async def generate_personalized_learning_path(
    request: PersonalizedLearningPathRequest,
    user: User = Depends(oauth2_scheme),
    prerequisites_service: CoursePrerequisitesService = Depends(get_course_prerequisites_service),
):
    try:
        path = await prerequisites_service.generate_personalized_learning_path(
            request.user_id, request.target_course_id, request.max_courses
        )
        logger.info(f"Generated personalized learning path for user {request.user_id}")
        return {"personalized_learning_path": path}
    except HTTPException as e:
        logger.warning(f"HTTP error in generate_personalized_learning_path: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in generate_personalized_learning_path: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the personalized learning path")

@router.post("/user-profile/update")
async def update_user_profile(
    request: UserProfileUpdateRequest,
    user: User = Depends(oauth2_scheme),
    prerequisites_service: CoursePrerequisitesService = Depends(get_course_prerequisites_service),
):
    try:
        await prerequisites_service.update_user_profile(
            request.user_id, request.skills, request.completed_courses
        )
        logger.info(f"Updated user profile for user {request.user_id}")
        return {"message": "User profile updated successfully"}
    except HTTPException as e:
        logger.warning(f"HTTP error in update_user_profile: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in update_user_profile: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating the user profile")

@router.get("/courses/recommend/{user_id}")
async def recommend_courses(
    user_id: str,
    num_recommendations: int = 5,
    user: User = Depends(oauth2_scheme),
    prerequisites_service: CoursePrerequisitesService = Depends(get_course_prerequisites_service),
):
    try:
        recommendations = await prerequisites_service.recommend_courses(user_id, num_recommendations)
        logger.info(f"Generated course recommendations for user {user_id}")
        return {"recommendations": recommendations}
    except HTTPException as e:
        logger.warning(f"HTTP error in recommend_courses: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in recommend_courses: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while recommending courses")
