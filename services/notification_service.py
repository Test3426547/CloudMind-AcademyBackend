import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
from datetime import datetime, timedelta
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
import random

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.notifications = {}
        self.user_preferences = {}
        self.user_embeddings = {}

    async def create_notification(self, user_id: str, notification_type: str, content: str) -> Dict[str, Any]:
        try:
            notification_id = f"notif_{len(self.notifications) + 1}"
            personalized_content = await self._generate_personalized_content(user_id, notification_type, content)
            send_time = await self._calculate_optimal_send_time(user_id)
            
            notification = {
                "id": notification_id,
                "user_id": user_id,
                "type": notification_type,
                "content": personalized_content,
                "created_at": datetime.now(),
                "send_time": send_time,
                "status": "pending"
            }
            
            self.notifications[notification_id] = notification
            return notification
        except Exception as e:
            logger.error(f"Error creating notification: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create notification")

    async def get_user_notifications(self, user_id: str) -> List[Dict[str, Any]]:
        return [notif for notif in self.notifications.values() if notif["user_id"] == user_id]

    async def update_notification_status(self, notification_id: str, status: str) -> Dict[str, Any]:
        if notification_id not in self.notifications:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        self.notifications[notification_id]["status"] = status
        return self.notifications[notification_id]

    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        self.user_preferences[user_id] = preferences
        await self._update_user_embedding(user_id)
        return self.user_preferences[user_id]

    async def _generate_personalized_content(self, user_id: str, notification_type: str, content: str) -> str:
        user_embedding = await self._get_user_embedding(user_id)
        prompt = f"Generate a personalized notification for a user with the following context:\n\n"
        prompt += f"User embedding: {user_embedding[:10]}...\n"  # Use only a part of the embedding for brevity
        prompt += f"Notification type: {notification_type}\n"
        prompt += f"Original content: {content}\n\n"
        prompt += "Please provide a personalized version of the notification content."

        personalized_content = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI assistant specializing in creating personalized notification content."},
            {"role": "user", "content": prompt}
        ], "medium")

        return personalized_content.strip()

    async def _calculate_optimal_send_time(self, user_id: str) -> datetime:
        user_prefs = self.user_preferences.get(user_id, {})
        preferred_times = user_prefs.get("preferred_notification_times", [])
        
        if not preferred_times:
            return datetime.now() + timedelta(minutes=random.randint(5, 60))
        
        now = datetime.now()
        optimal_time = min(preferred_times, key=lambda t: abs(now.replace(hour=t.hour, minute=t.minute) - now))
        return now.replace(hour=optimal_time.hour, minute=optimal_time.minute)

    async def _get_user_embedding(self, user_id: str) -> List[float]:
        if user_id not in self.user_embeddings:
            await self._update_user_embedding(user_id)
        return self.user_embeddings[user_id]

    async def _update_user_embedding(self, user_id: str):
        user_prefs = self.user_preferences.get(user_id, {})
        user_info = f"User preferences: {user_prefs}"
        self.user_embeddings[user_id] = await self.text_embedding_service.get_embedding(user_info)

    async def analyze_notification_sentiment(self, notification_id: str, user_response: str) -> Dict[str, Any]:
        if notification_id not in self.notifications:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        prompt = f"Analyze the sentiment of the following user response to a notification:\n\n{user_response}\n\nProvide a sentiment score between -1 (very negative) and 1 (very positive), and a brief explanation."
        
        analysis = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI assistant specializing in sentiment analysis."},
            {"role": "user", "content": prompt}
        ], "low")
        
        # Parse the sentiment score and explanation from the analysis
        lines = analysis.strip().split("\n")
        sentiment_score = float(lines[0])
        explanation = "\n".join(lines[1:])
        
        return {
            "notification_id": notification_id,
            "sentiment_score": sentiment_score,
            "explanation": explanation
        }

notification_service = NotificationService(get_llm_orchestrator(), get_text_embedding_service())

def get_notification_service() -> NotificationService:
    return notification_service
