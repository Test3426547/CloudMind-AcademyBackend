from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from services.openai_client import send_openai_request
from models.user import User
from typing import List

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/learning-path/generate")
async def generate_learning_path(goal: str, user: User = Depends(oauth2_scheme)):
    try:
        prompt = f"Generate a personalized learning path for a user with the goal: {goal}. Include recommended courses, estimated time, and milestones."
        learning_path = send_openai_request(prompt)
        return {"learning_path": learning_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/learning-path/progress")
async def get_learning_progress(user: User = Depends(oauth2_scheme)):
    # Here you would typically fetch the user's learning progress from your database
    mock_progress = {
        "completed_courses": ["Introduction to Python", "Data Structures"],
        "current_course": "Algorithms",
        "progress_percentage": 60
    }
    return mock_progress

@router.post("/learning-path/update")
async def update_learning_path(course_id: str, completed: bool, user: User = Depends(oauth2_scheme)):
    # Here you would typically update the user's learning progress in your database
    return {"message": f"Course {course_id} marked as {'completed' if completed else 'in progress'}"}
