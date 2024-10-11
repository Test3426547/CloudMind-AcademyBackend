from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.course_feedback_service import CourseFeedbackService, get_course_feedback_service
from typing import List, Optional
from pydantic import BaseModel, Field, constr
import logging
from fastapi_limiter.depends import RateLimiter
from cachetools import TTLCache, cached

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

# Initialize cache
cache = TTLCache(maxsize=100, ttl=300)  # Cache for 5 minutes

class FeedbackCreate(BaseModel):
    course_id: constr(min_length=1, max_length=50)
    rating: int = Field(..., ge=1, le=5)
    comment: constr(min_length=1, max_length=1000)

class FeedbackResponse(BaseModel):
    id: str
    user_id: str
    course_id: str
    rating: int
    comment: str
    created_at: str

@router.post("/course-feedback", response_model=FeedbackResponse)
async def create_feedback(
    feedback: FeedbackCreate,
    user: User = Depends(oauth2_scheme),
    feedback_service: CourseFeedbackService = Depends(get_course_feedback_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        result = await feedback_service.create_feedback(user.id, feedback.course_id, feedback.rating, feedback.comment)
        logger.info(f"Feedback created for course {feedback.course_id} by user {user.id}")
        return result
    except ValueError as e:
        logger.warning(f"Invalid input for create_feedback: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while creating feedback")

@router.get("/course-feedback/{course_id}", response_model=List[FeedbackResponse])
@cached(cache)
async def get_course_feedback(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    feedback_service: CourseFeedbackService = Depends(get_course_feedback_service),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        feedback_list = await feedback_service.get_course_feedback(course_id, limit, offset)
        logger.info(f"Retrieved {len(feedback_list)} feedback entries for course {course_id}")
        return feedback_list
    except ValueError as e:
        logger.warning(f"Invalid input for get_course_feedback: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving course feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving course feedback")

@router.get("/course-feedback/user/{user_id}", response_model=List[FeedbackResponse])
@cached(cache)
async def get_user_feedback(
    user_id: str,
    authenticated_user: User = Depends(oauth2_scheme),
    feedback_service: CourseFeedbackService = Depends(get_course_feedback_service),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        if authenticated_user.id != user_id and not authenticated_user.is_admin:
            raise HTTPException(status_code=403, detail="Not authorized to view this user's feedback")
        
        feedback_list = await feedback_service.get_user_feedback(user_id, limit, offset)
        logger.info(f"Retrieved {len(feedback_list)} feedback entries for user {user_id}")
        return feedback_list
    except ValueError as e:
        logger.warning(f"Invalid input for get_user_feedback: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving user feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving user feedback")

@router.put("/course-feedback/{feedback_id}", response_model=FeedbackResponse)
async def update_feedback(
    feedback_id: str,
    feedback: FeedbackCreate,
    user: User = Depends(oauth2_scheme),
    feedback_service: CourseFeedbackService = Depends(get_course_feedback_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        result = await feedback_service.update_feedback(user.id, feedback_id, feedback.rating, feedback.comment)
        logger.info(f"Feedback {feedback_id} updated by user {user.id}")
        return result
    except ValueError as e:
        logger.warning(f"Invalid input for update_feedback: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error updating feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating feedback")

@router.delete("/course-feedback/{feedback_id}")
async def delete_feedback(
    feedback_id: str,
    user: User = Depends(oauth2_scheme),
    feedback_service: CourseFeedbackService = Depends(get_course_feedback_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        await feedback_service.delete_feedback(user.id, feedback_id)
        logger.info(f"Feedback {feedback_id} deleted by user {user.id}")
        return {"message": "Feedback deleted successfully"}
    except ValueError as e:
        logger.warning(f"Invalid input for delete_feedback: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while deleting feedback")

@router.get("/course-feedback/stats/{course_id}")
@cached(cache)
async def get_course_feedback_stats(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    feedback_service: CourseFeedbackService = Depends(get_course_feedback_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        stats = await feedback_service.get_course_feedback_stats(course_id)
        logger.info(f"Retrieved feedback stats for course {course_id}")
        return stats
    except ValueError as e:
        logger.warning(f"Invalid input for get_course_feedback_stats: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving course feedback stats: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving course feedback stats")
