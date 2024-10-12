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
    notification_type: str = Field(..., min_length=1, max_length=50)
    content: str = Field(..., min_length=1)

class NotificationUpdate(BaseModel):
    status: str = Field(..., pattern="^(read|unread|dismissed)$")

class UserPreferences(BaseModel):
    preferred_notification_times: List[str] = Field(..., min_items=1)
    notification_frequency: str = Field(..., pattern="^(low|medium|high)$")
    preferred_channels: List[str] = Field(..., min_items=1)

class NotificationResponse(BaseModel):
    user_response: str = Field(..., min_length=1)

@router.post("/notifications")
async def create_notification(
    notification: NotificationCreate,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
):
    try:
        result = await notification_service.create_notification(user.id, notification.notification_type, notification.content)
        logger.info(f"Notification created for user {user.id}")
        return result
    except Exception as e:
        logger.error(f"Error creating notification: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while creating the notification")

@router.get("/notifications")
async def get_user_notifications(
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
):
    try:
        notifications = await notification_service.get_user_notifications(user.id)
        logger.info(f"Retrieved notifications for user {user.id}")
        return notifications
    except Exception as e:
        logger.error(f"Error retrieving notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving notifications")

@router.put("/notifications/{notification_id}")
async def update_notification_status(
    notification_id: str,
    update: NotificationUpdate,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
):
    try:
        result = await notification_service.update_notification_status(notification_id, update.status)
        logger.info(f"Updated status of notification {notification_id} for user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"Error updating notification status: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error updating notification status: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating the notification status")

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
    except Exception as e:
        logger.error(f"Error updating user preferences: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while updating user preferences")

@router.post("/notifications/{notification_id}/sentiment")
async def analyze_notification_sentiment(
    notification_id: str,
    response: NotificationResponse,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
):
    try:
        result = await notification_service.analyze_notification_sentiment(notification_id, response.user_response)
        logger.info(f"Analyzed sentiment for notification {notification_id} response from user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"Error analyzing notification sentiment: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error analyzing notification sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while analyzing the notification sentiment")
