from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from models.course_prerequisite import CoursePrerequisite, UserCourseProgress
from services.course_prerequisite_service import CoursePrerequisiteService, get_course_prerequisite_service
from typing import List

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/prerequisites")
async def add_prerequisite(
    prerequisite: CoursePrerequisite,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service)
):
    await prerequisite_service.add_prerequisite(prerequisite)
    return {"message": "Prerequisite added successfully"}

@router.get("/prerequisites/{course_id}", response_model=List[str])
async def get_prerequisites(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service)
):
    return await prerequisite_service.get_prerequisites(course_id)

@router.post("/progress")
async def update_user_progress(
    progress: UserCourseProgress,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service)
):
    await prerequisite_service.update_user_progress(progress)
    return {"message": "User progress updated successfully"}

@router.get("/progress/{user_id}/{course_id}", response_model=UserCourseProgress)
async def get_user_progress(
    user_id: str,
    course_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service)
):
    return await prerequisite_service.get_user_progress(user_id, course_id)

@router.get("/check-prerequisites/{user_id}/{course_id}")
async def check_prerequisites_met(
    user_id: str,
    course_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service)
):
    prerequisites_met = await prerequisite_service.check_prerequisites_met(user_id, course_id)
    return {"prerequisites_met": prerequisites_met}
