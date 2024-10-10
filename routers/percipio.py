from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from services.percipio_service import PercipioService, get_percipio_service
from typing import List, Dict, Any
from models.user import User

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.get("/percipio/courses")
async def get_percipio_courses(
    limit: int = 10,
    offset: int = 0,
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service)
):
    try:
        courses = percipio_service.get_courses(limit, offset)
        return {"courses": courses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/percipio/courses/{course_id}")
async def get_percipio_course_details(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service)
):
    try:
        course_details = percipio_service.get_course_details(course_id)
        return course_details
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/percipio/courses/{course_id}/start")
async def start_percipio_course(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service)
):
    try:
        result = percipio_service.start_course(user.id, course_id)
        return {"message": "Course started successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/percipio/user/progress")
async def get_user_progress(
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioService = Depends(get_percipio_service)
):
    try:
        progress = percipio_service.get_user_progress(user.id)
        return {"user_id": user.id, "progress": progress}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
