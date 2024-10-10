from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.grading_service import GradingService, get_grading_service
from typing import List, Dict, Any
from pydantic import BaseModel

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class AssignmentGradeRequest(BaseModel):
    assignment_text: str
    student_submission: str

class QuizGenerationRequest(BaseModel):
    course_content: str
    difficulty: str
    num_questions: int

@router.post("/grade-assignment")
async def grade_assignment(
    request: AssignmentGradeRequest,
    user: User = Depends(oauth2_scheme),
    grading_service: GradingService = Depends(get_grading_service)
):
    result = await grading_service.grade_assignment(request.assignment_text, request.student_submission)
    return result

@router.post("/generate-quiz")
async def generate_quiz(
    request: QuizGenerationRequest,
    user: User = Depends(oauth2_scheme),
    grading_service: GradingService = Depends(get_grading_service)
):
    quiz = await grading_service.generate_quiz(request.course_content, request.difficulty, request.num_questions)
    return {"quiz": quiz}
