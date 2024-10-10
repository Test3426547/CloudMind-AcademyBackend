import uuid
from datetime import datetime, time
from typing import List, Dict, Optional
from models.notification import NotificationPreference, Notification

class NotificationService:
    def __init__(self):
        self.notifications: Dict[str, List[Notification]] = {}
        self.preferences: Dict[str, NotificationPreference] = {}

    async def create_or_update_preferences(self, user_id: str, preferences: NotificationPreference) -> NotificationPreference:
        self.preferences[user_id] = preferences
        return preferences

    async def get_user_preferences(self, user_id: str) -> NotificationPreference:
        return self.preferences.get(user_id, NotificationPreference(user_id=user_id))

    async def create_notification(self, user_id: str, title: str, message: str, notification_type: str, priority: str = "normal", action_url: Optional[str] = None) -> Optional[Notification]:
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

    async def get_user_notifications(self, user_id: str, filter_read: bool = False) -> List[Notification]:
        notifications = self.notifications.get(user_id, [])
        if filter_read:
            return [n for n in notifications if not n.is_read]
        return notifications

    async def mark_notification_as_read(self, user_id: str, notification_id: str) -> bool:
        user_notifications = self.notifications.get(user_id, [])
        for notification in user_notifications:
            if notification.id == notification_id:
                notification.is_read = True
                return True
        return False

    async def delete_notification(self, user_id: str, notification_id: str) -> bool:
        user_notifications = self.notifications.get(user_id, [])
        for index, notification in enumerate(user_notifications):
            if notification.id == notification_id:
                del user_notifications[index]
                return True
        return False

    async def get_notifications_by_type(self, user_id: str, notification_type: str) -> List[Notification]:
        user_notifications = self.notifications.get(user_id, [])
        return [n for n in user_notifications if n.notification_type == notification_type]

    async def bulk_delete_notifications(self, user_id: str, notification_ids: List[str]) -> int:
        user_notifications = self.notifications.get(user_id, [])
        initial_count = len(user_notifications)
        self.notifications[user_id] = [n for n in user_notifications if n.id not in notification_ids]
        return initial_count - len(self.notifications[user_id])

notification_service = NotificationService()

def get_notification_service() -> NotificationService:
    return notification_service
