from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.course_feedback_service import CourseFeedbackService, get_course_feedback_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class FeedbackSubmission(BaseModel):
    course_id: str = Field(..., min_length=1, max_length=50)
    rating: int = Field(..., ge=1, le=5)
    comment: str = Field(..., min_length=10, max_length=1000)

@router.post("/feedback/submit")
async def submit_feedback(
    feedback: FeedbackSubmission,
    user: User = Depends(oauth2_scheme),
    feedback_service: CourseFeedbackService = Depends(get_course_feedback_service),
):
    try:
        result = await feedback_service.submit_feedback(user.id, feedback.course_id, feedback.rating, feedback.comment)
        logger.info(f"Feedback submitted for course {feedback.course_id} by user {user.id}")
        return result
    except HTTPException as e:
        logger.error(f"HTTP error in submit_feedback: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in submit_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while submitting feedback")

@router.get("/feedback/{course_id}", response_model=List[Dict[str, Any]])
async def get_course_feedback(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    feedback_service: CourseFeedbackService = Depends(get_course_feedback_service),
):
    try:
        result = await feedback_service.get_course_feedback(course_id)
        logger.info(f"Course feedback retrieved for course {course_id}")
        return result
    except HTTPException as e:
        logger.error(f"HTTP error in get_course_feedback: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_course_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving course feedback")

@router.get("/feedback/{course_id}/summary", response_model=Dict[str, Any])
async def get_feedback_summary(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    feedback_service: CourseFeedbackService = Depends(get_course_feedback_service),
):
    try:
        result = await feedback_service.get_feedback_summary(course_id)
        logger.info(f"Feedback summary generated for course {course_id}")
        return result
    except HTTPException as e:
        logger.error(f"HTTP error in get_feedback_summary: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_feedback_summary: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating feedback summary")
