import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
from datetime import datetime, timedelta
import random
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self):
        self.notifications = {}
        self.user_preferences = defaultdict(lambda: {"quiet_hours": [], "frequency": "daily"})
        self.user_embeddings = defaultdict(lambda: [random.random() for _ in range(100)])  # Simulated user embeddings

    async def create_notification(self, user_id: str, content: str, importance: float) -> Dict[str, Any]:
        try:
            notification_id = f"notif_{len(self.notifications) + 1}"
            embedding = self._generate_content_embedding(content)
            scheduled_time = await self._smart_schedule_notification(user_id, importance)
            personalized_content = await self._personalize_content(user_id, content)
            
            self.notifications[notification_id] = {
                "user_id": user_id,
                "content": personalized_content,
                "importance": importance,
                "embedding": embedding,
                "scheduled_time": scheduled_time,
                "sent": False
            }
            return {"notification_id": notification_id, "scheduled_time": scheduled_time}
        except Exception as e:
            logger.error(f"Error creating notification: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create notification")

    async def get_user_notifications(self, user_id: str) -> List[Dict[str, Any]]:
        return [notif for notif in self.notifications.values() if notif["user_id"] == user_id and not notif["sent"]]

    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, str]:
        self.user_preferences[user_id].update(preferences)
        return {"message": "User preferences updated successfully"}

    async def _smart_schedule_notification(self, user_id: str, importance: float) -> datetime:
        now = datetime.now()
        user_prefs = self.user_preferences[user_id]
        quiet_hours = user_prefs["quiet_hours"]
        
        if importance > 0.8:  # High importance notifications are sent immediately
            return now
        
        # Simulated optimal time calculation using a basic time series analysis
        hour_scores = [0] * 24
        for hour in range(24):
            if any(start <= hour < end for start, end in quiet_hours):
                continue
            hour_scores[hour] = math.sin(hour * math.pi / 12) + random.uniform(0, 0.5)
        
        best_hour = hour_scores.index(max(hour_scores))
        scheduled_time = now.replace(hour=best_hour, minute=random.randint(0, 59))
        if scheduled_time < now:
            scheduled_time += timedelta(days=1)
        
        return scheduled_time

    def _generate_content_embedding(self, content: str) -> List[float]:
        # Simulated content embedding generation
        words = content.lower().split()
        embedding = [0.0] * 100
        for word in words:
            for i in range(100):
                embedding[i] += hash(word + str(i)) % 1000 / 1000
        magnitude = math.sqrt(sum(x**2 for x in embedding))
        return [x / magnitude for x in embedding]

    async def _personalize_content(self, user_id: str, content: str) -> str:
        user_embedding = self.user_embeddings[user_id]
        content_embedding = self._generate_content_embedding(content)
        
        # Simulated content personalization using cosine similarity
        similarity = sum(a * b for a, b in zip(user_embedding, content_embedding))
        personalization_factor = (similarity + 1) / 2  # Scale to 0-1
        
        # Simulated text generation for personalization
        personalized_prefix = "Based on your interests, " if personalization_factor > 0.7 else ""
        return personalized_prefix + content

    async def process_notifications(self) -> None:
        now = datetime.now()
        for notif_id, notif in self.notifications.items():
            if not notif["sent"] and notif["scheduled_time"] <= now:
                # Simulated notification sending process
                user_id = notif["user_id"]
                content = notif["content"]
                importance = notif["importance"]
                
                # Simulated sentiment analysis
                sentiment_score = self._analyze_sentiment(content)
                
                logger.info(f"Sending notification to user {user_id}: {content}")
                logger.info(f"Notification importance: {importance}, Sentiment: {sentiment_score}")
                
                notif["sent"] = True

    def _analyze_sentiment(self, text: str) -> float:
        # Simulated sentiment analysis using a basic lexicon-based approach
        positive_words = set(["good", "great", "excellent", "amazing", "wonderful", "fantastic"])
        negative_words = set(["bad", "poor", "terrible", "awful", "horrible", "disappointing"])
        
        words = text.lower().split()
        sentiment_score = sum(1 for word in words if word in positive_words) - sum(1 for word in words if word in negative_words)
        return max(min(sentiment_score / len(words), 1), -1)  # Normalize to [-1, 1]

notification_service = NotificationService()

def get_notification_service() -> NotificationService:
    return notification_service
