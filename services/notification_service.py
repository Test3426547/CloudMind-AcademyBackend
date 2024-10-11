import uuid
from datetime import datetime, time
from typing import List, Dict, Optional
from models.notification import NotificationPreference, Notification
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self):
        self.notifications: Dict[str, List[Notification]] = {}
        self.preferences: Dict[str, NotificationPreference] = {}

    async def create_or_update_preferences(self, user_id: str, preferences: Dict) -> NotificationPreference:
        try:
            updated_preferences = NotificationPreference(user_id=user_id, **preferences)
            self.preferences[user_id] = updated_preferences
            return updated_preferences
        except Exception as e:
            logger.error(f"Error creating or updating preferences: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while updating preferences")

    async def get_user_preferences(self, user_id: str) -> NotificationPreference:
        try:
            return self.preferences.get(user_id, NotificationPreference(user_id=user_id))
        except Exception as e:
            logger.error(f"Error getting user preferences: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while retrieving preferences")

    async def create_notification(self, user_id: str, title: str, message: str, notification_type: str, priority: str = "normal", action_url: Optional[str] = None) -> Optional[Notification]:
        try:
            preferences = await self.get_user_preferences(user_id)
            
            if not self._should_send_notification(preferences, notification_type):
                return None

            notification = Notification(
                id=str(uuid.uuid4()),
                user_id=user_id,
                title=title,
                message=message,
                notification_type=notification_type,
                created_at=datetime.now().isoformat(),
                priority=priority,
                action_url=action_url
            )

            if user_id not in self.notifications:
                self.notifications[user_id] = []
            self.notifications[user_id].append(notification)
            return notification
        except Exception as e:
            logger.error(f"Error creating notification: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while creating the notification")

    def _should_send_notification(self, preferences: NotificationPreference, notification_type: str) -> bool:
        if not preferences.notification_types.get(notification_type, True):
            return False

        current_time = datetime.now().time()
        if preferences.quiet_hours:
            quiet_start = time.fromisoformat(preferences.quiet_hours["start"])
            quiet_end = time.fromisoformat(preferences.quiet_hours["end"])
            if quiet_start <= current_time <= quiet_end:
                return False

        return True

    async def get_user_notifications(self, user_id: str, filter_read: bool = False, limit: int = 50, offset: int = 0) -> List[Notification]:
        try:
            notifications = self.notifications.get(user_id, [])
            if filter_read:
                notifications = [n for n in notifications if not n.is_read]
            return notifications[offset:offset+limit]
        except Exception as e:
            logger.error(f"Error getting user notifications: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while retrieving notifications")

    async def mark_notification_as_read(self, user_id: str, notification_id: str) -> bool:
        try:
            user_notifications = self.notifications.get(user_id, [])
            for notification in user_notifications:
                if notification.id == notification_id:
                    notification.is_read = True
                    return True
            return False
        except Exception as e:
            logger.error(f"Error marking notification as read: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while marking the notification as read")

    async def delete_notification(self, user_id: str, notification_id: str) -> bool:
        try:
            user_notifications = self.notifications.get(user_id, [])
            for index, notification in enumerate(user_notifications):
                if notification.id == notification_id:
                    del user_notifications[index]
                    return True
            return False
        except Exception as e:
            logger.error(f"Error deleting notification: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while deleting the notification")

    async def get_notifications_by_type(self, user_id: str, notification_type: str, limit: int = 50, offset: int = 0) -> List[Notification]:
        try:
            user_notifications = self.notifications.get(user_id, [])
            filtered_notifications = [n for n in user_notifications if n.notification_type == notification_type]
            return filtered_notifications[offset:offset+limit]
        except Exception as e:
            logger.error(f"Error getting notifications by type: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while retrieving notifications by type")

    async def bulk_delete_notifications(self, user_id: str, notification_ids: List[str]) -> int:
        try:
            user_notifications = self.notifications.get(user_id, [])
            initial_count = len(user_notifications)
            self.notifications[user_id] = [n for n in user_notifications if n.id not in notification_ids]
            return initial_count - len(self.notifications[user_id])
        except Exception as e:
            logger.error(f"Error bulk deleting notifications: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while bulk deleting notifications")

notification_service = NotificationService()

def get_notification_service() -> NotificationService:
    return notification_service
