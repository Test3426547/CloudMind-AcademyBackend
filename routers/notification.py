from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from models.notification import NotificationPreference, Notification
from services.notification_service import NotificationService, get_notification_service
from typing import List, Optional
from pydantic import BaseModel, Field, validator
import logging
from fastapi_limiter.depends import RateLimiter
from cachetools import TTLCache, cached

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

# Initialize cache
cache = TTLCache(maxsize=100, ttl=300)  # Cache for 5 minutes

class NotificationPreferenceRequest(BaseModel):
    email_notifications: bool = Field(..., description="Enable/disable email notifications")
    push_notifications: bool = Field(..., description="Enable/disable push notifications")
    sms_notifications: bool = Field(..., description="Enable/disable SMS notifications")
    quiet_hours_start: Optional[int] = Field(None, ge=0, le=23, description="Start hour of quiet hours (0-23)")
    quiet_hours_end: Optional[int] = Field(None, ge=0, le=23, description="End hour of quiet hours (0-23)")

    @validator('quiet_hours_start', 'quiet_hours_end')
    def validate_quiet_hours(cls, v, values, **kwargs):
        if v is not None and not (0 <= v <= 23):
            raise ValueError("Quiet hours must be between 0 and 23")
        return v

@router.post("/notifications/preferences")
async def set_notification_preferences(
    preferences: NotificationPreferenceRequest,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        result = await notification_service.create_or_update_preferences(user.id, preferences.dict())
        logger.info(f"Notification preferences updated for user {user.id}")
        return result
    except ValueError as e:
        logger.warning(f"Invalid input for set_notification_preferences: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error setting notification preferences: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while setting notification preferences")

@router.get("/notifications/preferences")
@cached(cache)
async def get_notification_preferences(
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        preferences = await notification_service.get_user_preferences(user.id)
        logger.info(f"Notification preferences retrieved for user {user.id}")
        return preferences
    except Exception as e:
        logger.error(f"Error getting notification preferences: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving notification preferences")

@router.get("/notifications", response_model=List[Notification])
async def get_user_notifications(
    filter_read: bool = Query(False, description="Filter read notifications"),
    limit: int = Query(50, ge=1, le=100, description="Number of notifications to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=20, seconds=60))
):
    try:
        notifications = await notification_service.get_user_notifications(user.id, filter_read, limit, offset)
        logger.info(f"Retrieved {len(notifications)} notifications for user {user.id}")
        return notifications
    except ValueError as e:
        logger.warning(f"Invalid input for get_user_notifications: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting user notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving notifications")

@router.post("/notifications/mark-read/{notification_id}")
async def mark_notification_as_read(
    notification_id: str,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=20, seconds=60))
):
    try:
        success = await notification_service.mark_notification_as_read(user.id, notification_id)
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")
        logger.info(f"Notification {notification_id} marked as read for user {user.id}")
        return {"message": "Notification marked as read"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification as read: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while marking the notification as read")

@router.delete("/notifications/{notification_id}")
async def delete_notification(
    notification_id: str,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        success = await notification_service.delete_notification(user.id, notification_id)
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")
        logger.info(f"Notification {notification_id} deleted for user {user.id}")
        return {"message": "Notification deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting notification: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while deleting the notification")

@router.get("/notifications/by-type/{notification_type}", response_model=List[Notification])
async def get_notifications_by_type(
    notification_type: str,
    limit: int = Query(50, ge=1, le=100, description="Number of notifications to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        notifications = await notification_service.get_notifications_by_type(user.id, notification_type, limit, offset)
        logger.info(f"Retrieved {len(notifications)} notifications of type {notification_type} for user {user.id}")
        return notifications
    except ValueError as e:
        logger.warning(f"Invalid input for get_notifications_by_type: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting notifications by type: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving notifications")

@router.post("/notifications/bulk-delete")
async def bulk_delete_notifications(
    notification_ids: List[str],
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        if len(notification_ids) > 100:
            raise HTTPException(status_code=400, detail="Cannot delete more than 100 notifications at once")
        deleted_count = await notification_service.bulk_delete_notifications(user.id, notification_ids)
        logger.info(f"Bulk deleted {deleted_count} notifications for user {user.id}")
        return {"message": f"{deleted_count} notifications deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk deleting notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while deleting notifications")

class CreateNotificationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=500)
    notification_type: str = Field(..., min_length=1, max_length=50)
    priority: str = Field("normal", regex="^(low|normal|high)$")
    action_url: Optional[str] = Field(None, max_length=200)

@router.post("/notifications/create")
async def create_notification(
    request: CreateNotificationRequest,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        notification = await notification_service.create_notification(
            user.id, request.title, request.message, request.notification_type, request.priority, request.action_url
        )
        if notification:
            logger.info(f"Notification created for user {user.id}")
            return notification
        else:
            logger.info(f"Notification not created for user {user.id} due to user preferences")
            return {"message": "Notification not created due to user preferences"}
    except ValueError as e:
        logger.warning(f"Invalid input for create_notification: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating notification: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while creating the notification")
