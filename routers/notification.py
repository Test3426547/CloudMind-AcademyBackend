from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from models.notification import NotificationPreference, Notification
from services.notification_service import NotificationService, get_notification_service
from typing import List, Optional

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/notifications/preferences")
async def set_notification_preferences(
    preferences: NotificationPreference,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service)
):
    return await notification_service.create_or_update_preferences(user.id, preferences)

@router.get("/notifications/preferences")
async def get_notification_preferences(
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service)
):
    return await notification_service.get_user_preferences(user.id)

@router.get("/notifications", response_model=List[Notification])
async def get_user_notifications(
    filter_read: bool = False,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service)
):
    return await notification_service.get_user_notifications(user.id, filter_read)

@router.post("/notifications/mark-read/{notification_id}")
async def mark_notification_as_read(
    notification_id: str,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service)
):
    success = await notification_service.mark_notification_as_read(user.id, notification_id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"message": "Notification marked as read"}

@router.delete("/notifications/{notification_id}")
async def delete_notification(
    notification_id: str,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service)
):
    success = await notification_service.delete_notification(user.id, notification_id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"message": "Notification deleted"}

@router.get("/notifications/by-type/{notification_type}", response_model=List[Notification])
async def get_notifications_by_type(
    notification_type: str,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service)
):
    return await notification_service.get_notifications_by_type(user.id, notification_type)

@router.post("/notifications/bulk-delete")
async def bulk_delete_notifications(
    notification_ids: List[str],
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service)
):
    deleted_count = await notification_service.bulk_delete_notifications(user.id, notification_ids)
    return {"message": f"{deleted_count} notifications deleted"}

@router.post("/notifications/create")
async def create_notification(
    title: str,
    message: str,
    notification_type: str,
    priority: str = "normal",
    action_url: Optional[str] = None,
    user: User = Depends(oauth2_scheme),
    notification_service: NotificationService = Depends(get_notification_service)
):
    notification = await notification_service.create_notification(
        user.id, title, message, notification_type, priority, action_url
    )
    if notification:
        return notification
    else:
        return {"message": "Notification not created due to user preferences"}
