from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.notification_service import NotificationService, get_notification_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class NotificationCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=500)
    importance: float = Field(..., ge=0, le=1)

class UserPreferences(BaseModel):
    quiet_hours: List[List[int]] = Field(default=[])
    frequency: str = Field(default="daily")

@router.post("/notifications")
async def create_notification(
    notification: NotificationCreate,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
):
    try:
        result = await notification_service.create_notification(user.id, notification.content, notification.importance)
        logger.info(f"Notification created for user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in create_notification: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in create_notification: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while creating the notification")

@router.get("/notifications")
async def get_user_notifications(
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
):
    try:
        notifications = await notification_service.get_user_notifications(user.id)
        logger.info(f"Retrieved notifications for user {user.id}")
        return {"notifications": notifications}
    except HTTPException as e:
        logger.warning(f"HTTP error in get_user_notifications: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_user_notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving notifications")

@router.put("/notifications/preferences")
async def update_user_preferences(
    preferences: UserPreferences,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
):
    try:
        result = await notification_service.update_user_preferences(user.id, preferences.dict())
        logger.info(f"Updated notification preferences for user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in update_user_preferences: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in update_user_preferences: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating user preferences")

@router.post("/notifications/process")
async def process_notifications(
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
):
    try:
        await notification_service.process_notifications()
        logger.info("Notifications processed successfully")
        return {"message": "Notifications processed successfully"}
    except HTTPException as e:
        logger.warning(f"HTTP error in process_notifications: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in process_notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing notifications")
