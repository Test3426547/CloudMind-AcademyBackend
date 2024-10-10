from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.course_feedback_service import CourseFeedbackService, get_course_feedback_service
from typing import List
from pydantic import BaseModel

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class FeedbackCreate(BaseModel):
    course_id: str
    rating: int
    comment: str

class FeedbackUpdate(BaseModel):
    rating: int
    comment: str

class FeedbackResponse(BaseModel):
    id: str
    user_id: str
    course_id: str
    rating: int
    comment: str
    created_at: str
    updated_at: str = None

@router.post("/course-feedback", response_model=FeedbackResponse)
async def create_feedback(
    feedback: FeedbackCreate,
    user: User = Depends(oauth2_scheme),
    feedback_service: CourseFeedbackService = Depends(get_course_feedback_service)
):
    return await feedback_service.create_feedback(user.id, feedback.dict())

@router.get("/course-feedback/{course_id}", response_model=List[FeedbackResponse])
async def get_course_feedback(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    feedback_service: CourseFeedbackService = Depends(get_course_feedback_service)
):
    return await feedback_service.get_course_feedback(course_id)

@router.get("/course-feedback/{course_id}/average-rating")
async def get_course_average_rating(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    feedback_service: CourseFeedbackService = Depends(get_course_feedback_service)
):
    average_rating = await feedback_service.get_course_average_rating(course_id)
    return {"course_id": course_id, "average_rating": average_rating}

@router.get("/course-feedback/{course_id}/user", response_model=FeedbackResponse)
async def get_user_feedback(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    feedback_service: CourseFeedbackService = Depends(get_course_feedback_service)
):
    feedback = await feedback_service.get_user_feedback(user.id, course_id)
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return feedback

@router.put("/course-feedback/{feedback_id}", response_model=FeedbackResponse)
async def update_feedback(
    feedback_id: str,
    feedback_update: FeedbackUpdate,
    user: User = Depends(oauth2_scheme),
    feedback_service: CourseFeedbackService = Depends(get_course_feedback_service)
):
    try:
        updated_feedback = await feedback_service.update_feedback(feedback_id, feedback_update.dict())
        return updated_feedback
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
