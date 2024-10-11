from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from services.percipio_service import PercipioService, get_percipio_service
from typing import List, Dict, Any, Optional
from models.user import User
import logging

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.get("/percipio/courses")
async def get_percipio_courses(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service)
):
    try:
        courses = await percipio_service.get_courses(limit, offset)
        return {"courses": courses}
    except Exception as e:
        logger.error(f"Error fetching Percipio courses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching courses: {str(e)}")

@router.get("/percipio/courses/{course_id}")
async def get_percipio_course_details(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service)
):
    try:
        course_details = await percipio_service.get_course_details(course_id)
        return course_details
    except Exception as e:
        logger.error(f"Error fetching Percipio course details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching course details: {str(e)}")

@router.post("/percipio/courses/{course_id}/start")
async def start_percipio_course(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service)
):
    try:
        result = await percipio_service.start_course(user.id, course_id)
        return {"message": "Course started successfully", "result": result}
    except Exception as e:
        logger.error(f"Error starting Percipio course: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting course: {str(e)}")

@router.get("/percipio/user/progress")
async def get_user_progress(
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service)
):
    try:
        progress = await percipio_service.get_user_progress(user.id)
        return {"user_id": user.id, "progress": progress}
    except Exception as e:
        logger.error(f"Error fetching user progress: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching user progress: {str(e)}")

@router.get("/percipio/search")
async def search_percipio_content(
    query: str,
    content_type: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service)
):
    try:
        results = await percipio_service.search_content(query, content_type, limit)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error searching Percipio content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching content: {str(e)}")

@router.get("/percipio/recommendations")
async def get_percipio_recommendations(
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service)
):
    try:
        recommendations = await percipio_service.get_recommendations(user.id)
        return {"user_id": user.id, "recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error fetching Percipio recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching recommendations: {str(e)}")
