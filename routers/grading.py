from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.grading_service import GradingService, get_grading_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class AssignmentSubmission(BaseModel):
    assignment_id: str = Field(..., min_length=1, max_length=50)
    student_submission: str = Field(..., min_length=10)
    rubric: Dict[str, Any]

class QuizRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=100)
    difficulty: str = Field(..., pattern="^(easy|medium|hard)$")
    num_questions: int = Field(..., ge=1, le=50)

@router.post("/grade-assignment")
async def grade_assignment(
    submission: AssignmentSubmission,
    user: User = Depends(oauth2_scheme),
    grading_service: GradingService = Depends(get_grading_service),
):
    try:
        result = await grading_service.grade_assignment(
            submission.assignment_id,
            submission.student_submission,
            submission.rubric
        )
        logger.info(f"Assignment graded for user {user.id}")
        return result
    except HTTPException as e:
        logger.error(f"HTTP error in grade_assignment: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in grade_assignment: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while grading the assignment")

@router.post("/generate-quiz")
async def generate_quiz(
    quiz_request: QuizRequest,
    user: User = Depends(oauth2_scheme),
    grading_service: GradingService = Depends(get_grading_service),
):
    try:
        quiz = await grading_service.generate_quiz(
            quiz_request.topic,
            quiz_request.difficulty,
            quiz_request.num_questions
        )
        logger.info(f"Quiz generated for user {user.id}")
        return {"quiz": quiz}
    except HTTPException as e:
        logger.error(f"HTTP error in generate_quiz: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in generate_quiz: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the quiz")
