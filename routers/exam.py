from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.exam_service import ExamService, get_exam_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class ExamCreate(BaseModel):
    course_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=10, max_length=1000)
    questions: List[Dict[str, Any]] = Field(..., min_items=1)

class ExamSubmit(BaseModel):
    answers: List[str] = Field(..., min_items=1)

@router.post("/exams")
async def create_exam(
    exam_data: ExamCreate,
    user: User = Depends(oauth2_scheme),
    exam_service: ExamService = Depends(get_exam_service),
):
    try:
        result = await exam_service.create_exam(exam_data.course_id, exam_data.dict())
        logger.info(f"Exam created successfully: {result['exam_id']}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in create_exam: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in create_exam: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while creating the exam")

@router.get("/exams/{exam_id}")
async def get_exam(
    exam_id: str,
    user: User = Depends(oauth2_scheme),
    exam_service: ExamService = Depends(get_exam_service),
):
    try:
        exam = await exam_service.get_exam(exam_id)
        logger.info(f"Exam retrieved successfully: {exam_id}")
        return exam
    except HTTPException as e:
        logger.warning(f"HTTP error in get_exam: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_exam: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving the exam")

@router.post("/exams/{exam_id}/submit")
async def submit_exam(
    exam_id: str,
    exam_submit: ExamSubmit,
    user: User = Depends(oauth2_scheme),
    exam_service: ExamService = Depends(get_exam_service),
):
    try:
        result = await exam_service.submit_exam(user.id, exam_id, exam_submit.answers)
        logger.info(f"Exam submitted successfully for user {user.id}, exam {exam_id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in submit_exam: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in submit_exam: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while submitting the exam")

@router.get("/exams/results")
async def get_user_exam_results(
    user: User = Depends(oauth2_scheme),
    exam_service: ExamService = Depends(get_exam_service),
):
    try:
        results = await exam_service.get_user_exam_results(user.id)
        logger.info(f"Exam results retrieved for user: {user.id}")
        return {"user_exam_results": results}
    except HTTPException as e:
        logger.warning(f"HTTP error in get_user_exam_results: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_user_exam_results: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving user exam results")

@router.get("/exams/recommend")
async def recommend_exams(
    num_recommendations: int = 3,
    user: User = Depends(oauth2_scheme),
    exam_service: ExamService = Depends(get_exam_service),
):
    try:
        recommendations = await exam_service.recommend_exams(user.id, num_recommendations)
        logger.info(f"Exam recommendations generated for user: {user.id}")
        return {"recommendations": recommendations}
    except HTTPException as e:
        logger.warning(f"HTTP error in recommend_exams: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in recommend_exams: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while recommending exams")
