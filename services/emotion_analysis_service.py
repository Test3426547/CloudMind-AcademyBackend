import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EmotionAnalysisService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.emotion_history = {}

    async def analyze_emotion(self, user_id: str, text: str, speech_data: str = None) -> Dict[str, Any]:
        try:
            text_emotion = await self._analyze_text_emotion(text)
            speech_emotion = await self._analyze_speech_emotion(speech_data) if speech_data else None
            
            combined_emotion = self._combine_emotions(text_emotion, speech_emotion)
            sentiment_intensity = self._calculate_sentiment_intensity(combined_emotion)
            
            self._update_emotion_history(user_id, combined_emotion, sentiment_intensity)
            
            emotion_trend = await self._analyze_emotion_trend(user_id)
            contextual_understanding = await self._generate_contextual_understanding(user_id, text, combined_emotion)
            
            return {
                "emotion": combined_emotion,
                "sentiment_intensity": sentiment_intensity,
                "emotion_trend": emotion_trend,
                "contextual_understanding": contextual_understanding
            }
        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred during emotion analysis")

    async def _analyze_text_emotion(self, text: str) -> str:
        prompt = f"Analyze the emotion in the following text. Provide a single word emotion label:\n\n{text}"
        emotion = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an advanced emotion analysis AI. Provide a single-word emotion label."},
            {"role": "user", "content": prompt}
        ], "high")
        return emotion.strip().lower()

    async def _analyze_speech_emotion(self, speech_data: str) -> str:
        # Simulated speech emotion analysis
        # In a real-world scenario, this would involve processing audio data
        emotions = ["happy", "sad", "angry", "neutral", "excited"]
        return np.random.choice(emotions)

    def _combine_emotions(self, text_emotion: str, speech_emotion: str = None) -> str:
        if speech_emotion:
            return text_emotion if text_emotion == speech_emotion else "mixed"
        return text_emotion

    def _calculate_sentiment_intensity(self, emotion: str) -> float:
        intensity_map = {
            "joy": 0.8,
            "sadness": -0.6,
            "anger": -0.7,
            "fear": -0.5,
            "surprise": 0.4,
            "disgust": -0.6,
            "neutral": 0.0,
            "mixed": 0.1
        }
        return intensity_map.get(emotion, 0.0)

    def _update_emotion_history(self, user_id: str, emotion: str, intensity: float):
        if user_id not in self.emotion_history:
            self.emotion_history[user_id] = []
        self.emotion_history[user_id].append({
            "timestamp": datetime.now(),
            "emotion": emotion,
            "intensity": intensity
        })

    async def _analyze_emotion_trend(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.emotion_history:
            return {"trend": "Not enough data"}
        
        history = self.emotion_history[user_id]
        if len(history) < 5:
            return {"trend": "Not enough data"}
        
        recent_emotions = history[-5:]
        emotion_counts = {}
        total_intensity = 0
        
        for entry in recent_emotions:
            emotion_counts[entry["emotion"]] = emotion_counts.get(entry["emotion"], 0) + 1
            total_intensity += entry["intensity"]
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        avg_intensity = total_intensity / len(recent_emotions)
        
        trend_direction = "stable"
        if avg_intensity > 0.5:
            trend_direction = "improving" if recent_emotions[-1]["intensity"] > recent_emotions[0]["intensity"] else "declining"
        elif avg_intensity < -0.5:
            trend_direction = "declining" if recent_emotions[-1]["intensity"] < recent_emotions[0]["intensity"] else "improving"
        
        return {
            "trend": trend_direction,
            "dominant_emotion": dominant_emotion,
            "average_intensity": avg_intensity
        }

    async def _generate_contextual_understanding(self, user_id: str, text: str, emotion: str) -> str:
        emotion_history = self.emotion_history.get(user_id, [])
        recent_emotions = [e["emotion"] for e in emotion_history[-5:]]
        
        prompt = f"""
        Generate a contextual understanding of the user's emotional state based on the following information:
        
        Current text: {text}
        Current emotion: {emotion}
        Recent emotion history: {', '.join(recent_emotions)}
        
        Provide a brief analysis of the user's emotional context and potential factors influencing their current emotional state.
        """
        
        understanding = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an advanced emotional intelligence AI. Provide insightful contextual analysis of emotional states."},
            {"role": "user", "content": prompt}
        ], "high")
        
        return understanding.strip()

emotion_analysis_service = EmotionAnalysisService(get_llm_orchestrator(), get_text_embedding_service())

def get_emotion_analysis_service() -> EmotionAnalysisService:
    return emotion_analysis_service
