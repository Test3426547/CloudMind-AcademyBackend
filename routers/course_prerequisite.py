from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from models.course_prerequisite import CoursePrerequisite, UserCourseProgress
from services.course_prerequisite_service import CoursePrerequisiteService, get_course_prerequisite_service
from typing import List, Dict, Any
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

@router.post("/prerequisites")
async def add_prerequisite(
    prerequisite: CoursePrerequisite,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
):
    try:
        if not user.is_admin:
            raise HTTPException(status_code=403, detail="Only administrators can add prerequisites")
        await prerequisite_service.add_prerequisite(prerequisite.dict())
        logger.info(f"Prerequisite added successfully for course {prerequisite.course_id}")
        return {"message": "Prerequisite added successfully"}
    except Exception as e:
        logger.error(f"Error adding prerequisite: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while adding the prerequisite")

@router.get("/prerequisites/{course_id}", response_model=List[str])
async def get_prerequisites(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
):
    try:
        prerequisites = await prerequisite_service.get_prerequisites(course_id)
        logger.info(f"Retrieved prerequisites for course {course_id}")
        return prerequisites
    except Exception as e:
        logger.error(f"Error retrieving prerequisites: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving prerequisites")

@router.post("/progress")
async def update_user_progress(
    progress: UserCourseProgress,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
):
    try:
        if progress.user_id != user.id:
            raise HTTPException(status_code=403, detail="You can only update your own progress")
        await prerequisite_service.update_user_progress(progress.dict())
        logger.info(f"User progress updated successfully for user {progress.user_id} and course {progress.course_id}")
        return {"message": "User progress updated successfully"}
    except Exception as e:
        logger.error(f"Error updating user progress: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating user progress")

@router.get("/progress/{user_id}/{course_id}", response_model=Dict[str, Any])
async def get_user_progress(
    user_id: str,
    course_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
):
    try:
        if user_id != user.id and not user.is_admin:
            raise HTTPException(status_code=403, detail="You can only view your own progress or you must be an admin")
        progress = await prerequisite_service.get_user_progress(user_id, course_id)
        logger.info(f"Retrieved user progress for user {user_id} and course {course_id}")
        return progress
    except Exception as e:
        logger.error(f"Error retrieving user progress: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving user progress")

@router.get("/check-prerequisites/{user_id}/{course_id}")
async def check_prerequisites_met(
    user_id: str,
    course_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
):
    try:
        if user_id != user.id and not user.is_admin:
            raise HTTPException(status_code=403, detail="You can only check your own prerequisites or you must be an admin")
        prerequisites_met = await prerequisite_service.check_prerequisites_met(user_id, course_id)
        logger.info(f"Checked prerequisites for user {user_id} and course {course_id}")
        return {"prerequisites_met": prerequisites_met}
    except Exception as e:
        logger.error(f"Error checking prerequisites: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while checking prerequisites")

@router.delete("/prerequisites/{course_id}/{prerequisite_id}")
async def remove_prerequisite(
    course_id: str,
    prerequisite_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
):
    try:
        if not user.is_admin:
            raise HTTPException(status_code=403, detail="Only administrators can remove prerequisites")
        await prerequisite_service.remove_prerequisite(course_id, prerequisite_id)
        logger.info(f"Prerequisite {prerequisite_id} removed successfully from course {course_id}")
        return {"message": "Prerequisite removed successfully"}
    except Exception as e:
        logger.error(f"Error removing prerequisite: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while removing the prerequisite")

@router.get("/recommend-prerequisites/{course_id}")
async def recommend_prerequisites(
    course_id: str,
    num_recommendations: int = Query(3, ge=1, le=10),
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
):
    try:
        recommendations = await prerequisite_service.recommend_prerequisites(course_id, num_recommendations)
        logger.info(f"Generated prerequisite recommendations for course {course_id}")
        return {"recommended_prerequisites": recommendations}
    except Exception as e:
        logger.error(f"Error recommending prerequisites: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while recommending prerequisites")

@router.get("/adaptive-learning-path/{user_id}/{target_course_id}")
async def generate_adaptive_learning_path(
    user_id: str,
    target_course_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
):
    try:
        if user_id != user.id and not user.is_admin:
            raise HTTPException(status_code=403, detail="You can only generate learning paths for yourself or you must be an admin")
        learning_path = await prerequisite_service.generate_adaptive_learning_path(user_id, target_course_id)
        logger.info(f"Generated adaptive learning path for user {user_id} and target course {target_course_id}")
        return {"adaptive_learning_path": learning_path}
    except Exception as e:
        logger.error(f"Error generating adaptive learning path: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the adaptive learning path")

@router.get("/course-difficulty/{course_id}")
async def estimate_course_difficulty(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
):
    try:
        difficulty = await prerequisite_service.estimate_course_difficulty(course_id)
        logger.info(f"Estimated difficulty for course {course_id}")
        return {"course_id": course_id, "estimated_difficulty": difficulty}
    except Exception as e:
        logger.error(f"Error estimating course difficulty: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while estimating course difficulty")

@router.get("/learning-gaps/{user_id}/{target_course_id}")
async def analyze_learning_gaps(
    user_id: str,
    target_course_id: str,
    user: User = Depends(oauth2_scheme),
    prerequisite_service: CoursePrerequisiteService = Depends(get_course_prerequisite_service),
):
    try:
        if user_id != user.id and not user.is_admin:
            raise HTTPException(status_code=403, detail="You can only analyze learning gaps for yourself or you must be an admin")
        gaps_analysis = await prerequisite_service.analyze_learning_gaps(user_id, target_course_id)
        logger.info(f"Analyzed learning gaps for user {user_id} and target course {target_course_id}")
        return gaps_analysis
    except Exception as e:
        logger.error(f"Error analyzing learning gaps: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while analyzing learning gaps")
