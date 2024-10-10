from pydantic import BaseModel
from typing import Dict, Optional

class NotificationPreference(BaseModel):
    user_id: str
    email_notifications: bool = True
    push_notifications: bool = True
    sms_notifications: bool = False
    notification_frequency: str = "daily"  # Options: "immediate", "daily", "weekly"
    quiet_hours: Optional[Dict[str, str]] = None  # e.g., {"start": "22:00", "end": "07:00"}
    notification_types: Dict[str, bool] = {
        "course_updates": True,
        "new_challenges": True,
        "leaderboard_changes": True,
        "achievement_unlocked": True,
        "assignment_reminders": True,
        "quiz_results": True,
        "discussion_replies": True,
        "system_announcements": True,
        "collaborative_session_invites": True,
        "personalized_study_reminders": True
    }

class Notification(BaseModel):
    id: Optional[str] = None
    user_id: str
    title: str
    message: str
    notification_type: str
    is_read: bool = False
    created_at: str
    priority: str = "normal"  # Options: "low", "normal", "high", "urgent"
    action_url: Optional[str] = None  # URL to direct users when they interact with the notification
