from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.offline_learning_service import OfflineLearningService, get_offline_learning_service
from typing import Dict, List, Any
from pydantic import BaseModel

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class SyncData(BaseModel):
    user_progress: Dict[str, float]
    quiz_responses: Dict[str, List[Dict[str, Any]]]

@router.post("/offline/sync")
async def sync_offline_data(
    sync_data: SyncData,
    user: User = Depends(oauth2_scheme),
    offline_service: OfflineLearningService = Depends(get_offline_learning_service)
):
    try:
        # Update user progress
        for course_id, progress in sync_data.user_progress.items():
            offline_service.update_user_progress(user.id, course_id, progress)

        # Save quiz responses
        for quiz_id, responses in sync_data.quiz_responses.items():
            offline_service.save_quiz_response(user.id, quiz_id, responses)

        # Get the sync queue to send to the server
        sync_queue = offline_service.get_sync_queue()

        # In a real-world scenario, you would send this data to the server
        # and receive updated data from the server to merge
        # For this example, we'll just clear the sync queue
        offline_service.clear_sync_queue()

        return {"message": "Sync successful", "synced_items": len(sync_queue)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/offline/course/{course_id}")
async def get_offline_course_content(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    offline_service: OfflineLearningService = Depends(get_offline_learning_service)
):
    course_content = offline_service.get_cached_course_content(course_id)
    if not course_content:
        raise HTTPException(status_code=404, detail="Course content not found in offline cache")
    return course_content

@router.get("/offline/progress/{course_id}")
async def get_offline_user_progress(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    offline_service: OfflineLearningService = Depends(get_offline_learning_service)
):
    progress = offline_service.get_user_progress(user.id, course_id)
    return {"course_id": course_id, "progress": progress}

@router.get("/offline/quiz/{quiz_id}")
async def get_offline_quiz_responses(
    quiz_id: str,
    user: User = Depends(oauth2_scheme),
    offline_service: OfflineLearningService = Depends(get_offline_learning_service)
):
    responses = offline_service.get_quiz_responses(user.id, quiz_id)
    return {"quiz_id": quiz_id, "responses": responses}
