from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from services.percipio_service import PercipioService, get_percipio_service
from typing import List, Dict, Any, Optional
from models.user import User
import logging
from fastapi_limiter.depends import RateLimiter

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
    percipio_service: PercipioService = Depends(get_percipio_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        courses = await percipio_service.get_courses(limit, offset)
        logger.info(f"Successfully fetched {len(courses)} Percipio courses for user {user.id}")
        return {"courses": courses}
    except Exception as e:
        logger.error(f"Error fetching Percipio courses for user {user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching Percipio courses. Please try again later.")

@router.get("/percipio/courses/{course_id}")
async def get_percipio_course_details(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=20, seconds=60))
):
    try:
        course_details = await percipio_service.get_course_details(course_id)
        logger.info(f"Successfully fetched details for Percipio course {course_id} for user {user.id}")
        return course_details
    except Exception as e:
        logger.error(f"Error fetching Percipio course details for course {course_id} and user {user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching course details. Please try again later.")

@router.post("/percipio/courses/{course_id}/start")
async def start_percipio_course(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        result = await percipio_service.start_course(user.id, course_id)
        logger.info(f"Successfully started Percipio course {course_id} for user {user.id}")
        return {"message": "Course started successfully", "result": result}
    except Exception as e:
        logger.error(f"Error starting Percipio course {course_id} for user {user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while starting the course. Please try again later.")

@router.get("/percipio/user/progress")
async def get_user_progress(
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        progress = await percipio_service.get_user_progress(user.id)
        logger.info(f"Successfully fetched progress for user {user.id}")
        return {"user_id": user.id, "progress": progress}
    except Exception as e:
        logger.error(f"Error fetching user progress for user {user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching user progress. Please try again later.")

@router.get("/percipio/search")
async def search_percipio_content(
    query: str,
    content_type: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=15, seconds=60))
):
    try:
        results = await percipio_service.search_content(query, content_type, limit)
        logger.info(f"Successfully searched Percipio content for user {user.id} with query: {query}")
        return {"results": results}
    except Exception as e:
        logger.error(f"Error searching Percipio content for user {user.id} with query {query}: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while searching content. Please try again later.")

@router.get("/percipio/recommendations")
async def get_percipio_recommendations(
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        recommendations = await percipio_service.get_recommendations(user.id)
        logger.info(f"Successfully fetched Percipio recommendations for user {user.id}")
        return {"user_id": user.id, "recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error fetching Percipio recommendations for user {user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching recommendations. Please try again later.")
