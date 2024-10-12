from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.lms_integration_service import LMSIntegrationService, get_lms_integration_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class LMSIntegration(BaseModel):
    lms_type: str = Field(..., min_length=1, max_length=50)
    credentials: Dict[str, str]

class CourseContent(BaseModel):
    content: str = Field(..., min_length=10)

class UserBackground(BaseModel):
    background: str = Field(..., min_length=10)

class UserInterests(BaseModel):
    interests: List[str] = Field(..., min_items=1)
    completed_courses: List[str] = Field(default=[])

@router.post("/lms/integrate")
async def integrate_lms(
    integration: LMSIntegration,
    user: User = Depends(oauth2_scheme),
    lms_service: LMSIntegrationService = Depends(get_lms_integration_service),
):
    try:
        result = await lms_service.integrate_lms(integration.lms_type, integration.credentials)
        logger.info(f"LMS integrated successfully for user {user.id}")
        return result
    except Exception as e:
        logger.error(f"Error integrating LMS: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while integrating the LMS")

@router.get("/lms/courses/{integration_id}")
async def get_courses(
    integration_id: str,
    user: User = Depends(oauth2_scheme),
    lms_service: LMSIntegrationService = Depends(get_lms_integration_service),
):
    try:
        courses = await lms_service.get_courses(integration_id)
        logger.info(f"Courses retrieved for integration {integration_id}")
        return courses
    except HTTPException as e:
        logger.warning(f"Error retrieving courses: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error retrieving courses: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving courses")

@router.post("/lms/sync_course/{integration_id}/{course_id}")
async def sync_course(
    integration_id: str,
    course_id: str,
    user: User = Depends(oauth2_scheme),
    lms_service: LMSIntegrationService = Depends(get_lms_integration_service),
):
    try:
        result = await lms_service.sync_course(integration_id, course_id)
        logger.info(f"Course {course_id} synced for integration {integration_id}")
        return result
    except HTTPException as e:
        logger.warning(f"Error syncing course: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error syncing course: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while syncing the course")

@router.post("/lms/analyze_course/{course_id}")
async def analyze_course_content(
    course_id: str,
    content: CourseContent,
    user: User = Depends(oauth2_scheme),
    lms_service: LMSIntegrationService = Depends(get_lms_integration_service),
):
    try:
        analysis = await lms_service.analyze_course_content(course_id, content.content)
        logger.info(f"Course content analyzed for course {course_id}")
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing course content: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while analyzing course content")

@router.post("/lms/learning_path/{target_course_id}")
async def generate_learning_path(
    target_course_id: str,
    user_background: UserBackground,
    user: User = Depends(oauth2_scheme),
    lms_service: LMSIntegrationService = Depends(get_lms_integration_service),
):
    try:
        learning_path = await lms_service.generate_personalized_learning_path(user.id, target_course_id, user_background.background)
        logger.info(f"Personalized learning path generated for user {user.id} and course {target_course_id}")
        return learning_path
    except Exception as e:
        logger.error(f"Error generating learning path: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the learning path")

@router.post("/lms/recommend_courses")
async def recommend_courses(
    user_interests: UserInterests,
    user: User = Depends(oauth2_scheme),
    lms_service: LMSIntegrationService = Depends(get_lms_integration_service),
):
    try:
        recommendations = await lms_service.recommend_courses(user.id, user_interests.interests, user_interests.completed_courses)
        logger.info(f"Course recommendations generated for user {user.id}")
        return recommendations
    except Exception as e:
        logger.error(f"Error recommending courses: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while recommending courses")
