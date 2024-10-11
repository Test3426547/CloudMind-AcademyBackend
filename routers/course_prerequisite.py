from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from models.course_prerequisite import CoursePrerequisite, UserCourseProgress
from services.course_prerequisite_service import CoursePrerequisiteService, get_course_prerequisite_service
from typing import List
from fastapi_limiter.depends import RateLimiter
import logging
from cachetools import TTLCache, cached

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

# Initialize cache
cache = TTLCache(maxsize=100, ttl=300)  # Cache for 5 minutes

@router.post("/prerequisites")
async def add_prerequisite(
    prerequisite: CoursePrerequisite,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        if not user.is_admin:
            raise HTTPException(status_code=403, detail="Only administrators can add prerequisites")
        await prerequisite_service.add_prerequisite(prerequisite)
        logger.info(f"Prerequisite added successfully for course {prerequisite.course_id}")
        return {"message": "Prerequisite added successfully"}
    except ValueError as e:
        logger.warning(f"Invalid input for add_prerequisite: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding prerequisite: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while adding the prerequisite")

@router.get("/prerequisites/{course_id}", response_model=List[str])
@cached(cache)
async def get_prerequisites(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        prerequisites = await prerequisite_service.get_prerequisites(course_id)
        logger.info(f"Retrieved prerequisites for course {course_id}")
        return prerequisites
    except ValueError as e:
        logger.warning(f"Invalid input for get_prerequisites: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving prerequisites: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving prerequisites")

@router.post("/progress")
async def update_user_progress(
    progress: UserCourseProgress,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=20, seconds=60))
):
    try:
        if progress.user_id != user.id:
            raise HTTPException(status_code=403, detail="You can only update your own progress")
        await prerequisite_service.update_user_progress(progress)
        logger.info(f"User progress updated successfully for user {progress.user_id} and course {progress.course_id}")
        return {"message": "User progress updated successfully"}
    except ValueError as e:
        logger.warning(f"Invalid input for update_user_progress: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating user progress: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating user progress")

@router.get("/progress/{user_id}/{course_id}", response_model=UserCourseProgress)
@cached(cache)
async def get_user_progress(
    user_id: str,
    course_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        if user_id != user.id and not user.is_admin:
            raise HTTPException(status_code=403, detail="You can only view your own progress or you must be an admin")
        progress = await prerequisite_service.get_user_progress(user_id, course_id)
        logger.info(f"Retrieved user progress for user {user_id} and course {course_id}")
        return progress
    except ValueError as e:
        logger.warning(f"Invalid input for get_user_progress: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving user progress: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving user progress")

@router.get("/check-prerequisites/{user_id}/{course_id}")
@cached(cache)
async def check_prerequisites_met(
    user_id: str,
    course_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=15, seconds=60))
):
    try:
        if user_id != user.id and not user.is_admin:
            raise HTTPException(status_code=403, detail="You can only check your own prerequisites or you must be an admin")
        prerequisites_met = await prerequisite_service.check_prerequisites_met(user_id, course_id)
        logger.info(f"Checked prerequisites for user {user_id} and course {course_id}")
        return {"prerequisites_met": prerequisites_met}
    except ValueError as e:
        logger.warning(f"Invalid input for check_prerequisites_met: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error checking prerequisites: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while checking prerequisites")

@router.delete("/prerequisites/{course_id}/{prerequisite_id}")
async def remove_prerequisite(
    course_id: str,
    prerequisite_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        if not user.is_admin:
            raise HTTPException(status_code=403, detail="Only administrators can remove prerequisites")
        await prerequisite_service.remove_prerequisite(course_id, prerequisite_id)
        logger.info(f"Prerequisite {prerequisite_id} removed successfully from course {course_id}")
        return {"message": "Prerequisite removed successfully"}
    except ValueError as e:
        logger.warning(f"Invalid input for remove_prerequisite: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error removing prerequisite: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while removing the prerequisite")
