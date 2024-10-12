import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
from datetime import datetime, timedelta
import random
import statistics

logger = logging.getLogger(__name__)

class TimeTrackingService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.time_entries = {}
        self.user_embeddings = {}

    async def start_timer(self, user_id: str, task: str) -> Dict[str, Any]:
        try:
            entry_id = f"entry_{len(self.time_entries) + 1}"
            start_time = datetime.now()
            self.time_entries[entry_id] = {
                "user_id": user_id,
                "task": task,
                "start_time": start_time,
                "end_time": None,
                "duration": None
            }
            return {"entry_id": entry_id, "start_time": start_time}
        except Exception as e:
            logger.error(f"Error starting timer: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to start timer")

    async def stop_timer(self, entry_id: str) -> Dict[str, Any]:
        try:
            if entry_id not in self.time_entries:
                raise HTTPException(status_code=404, detail="Time entry not found")
            
            entry = self.time_entries[entry_id]
            if entry["end_time"] is not None:
                raise HTTPException(status_code=400, detail="Timer already stopped")
            
            end_time = datetime.now()
            duration = (end_time - entry["start_time"]).total_seconds()
            
            entry["end_time"] = end_time
            entry["duration"] = duration
            
            await self._update_user_embedding(entry["user_id"], entry["task"], duration)
            
            return {"entry_id": entry_id, "duration": duration}
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error stopping timer: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to stop timer")

    async def get_user_time_entries(self, user_id: str) -> List[Dict[str, Any]]:
        return [entry for entry in self.time_entries.values() if entry["user_id"] == user_id]

    async def analyze_productivity(self, user_id: str) -> Dict[str, Any]:
        try:
            user_entries = await self.get_user_time_entries(user_id)
            if not user_entries:
                raise HTTPException(status_code=404, detail="No time entries found for user")
            
            total_time = sum(entry["duration"] for entry in user_entries if entry["duration"] is not None)
            avg_duration = total_time / len(user_entries)
            
            task_durations = {}
            for entry in user_entries:
                task = entry["task"]
                duration = entry["duration"]
                if duration is not None:
                    if task not in task_durations:
                        task_durations[task] = []
                    task_durations[task].append(duration)
            
            task_stats = {task: {"avg_duration": sum(durations) / len(durations), "count": len(durations)} 
                          for task, durations in task_durations.items()}
            
            productivity_score = await self._calculate_productivity_score(user_id, avg_duration, task_stats)
            
            insights = await self._generate_productivity_insights(user_id, productivity_score, task_stats)
            
            return {
                "total_time": total_time,
                "avg_duration": avg_duration,
                "task_stats": task_stats,
                "productivity_score": productivity_score,
                "insights": insights
            }
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error analyzing productivity: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to analyze productivity")

    async def predict_task_duration(self, user_id: str, task: str) -> Dict[str, Any]:
        try:
            user_entries = await self.get_user_time_entries(user_id)
            task_durations = [entry["duration"] for entry in user_entries if entry["task"] == task and entry["duration"] is not None]
            
            if not task_durations:
                similar_tasks = await self._find_similar_tasks(user_id, task)
                if similar_tasks:
                    task_durations = [entry["duration"] for entry in user_entries if entry["task"] in similar_tasks and entry["duration"] is not None]
                else:
                    raise HTTPException(status_code=404, detail="No historical data found for the given task or similar tasks")
            
            avg_duration = statistics.mean(task_durations)
            std_dev = statistics.stdev(task_durations) if len(task_durations) > 1 else 0
            
            predicted_duration = avg_duration + (std_dev * random.uniform(-0.1, 0.1))  # Add some randomness
            
            confidence = 1 - (std_dev / avg_duration) if avg_duration > 0 else 0  # Simple confidence calculation
            
            return {
                "task": task,
                "predicted_duration": predicted_duration,
                "confidence": confidence
            }
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error predicting task duration: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to predict task duration")

    async def detect_time_tracking_anomalies(self, user_id: str) -> List[Dict[str, Any]]:
        try:
            user_entries = await self.get_user_time_entries(user_id)
            if len(user_entries) < 10:
                return []  # Not enough data for anomaly detection
            
            durations = [entry["duration"] for entry in user_entries if entry["duration"] is not None]
            avg_duration = statistics.mean(durations)
            std_dev = statistics.stdev(durations)
            
            anomalies = []
            for entry in user_entries:
                if entry["duration"] is not None:
                    z_score = (entry["duration"] - avg_duration) / std_dev
                    if abs(z_score) > 2:  # Consider durations more than 2 standard deviations away as anomalies
                        anomalies.append({
                            "entry_id": entry["entry_id"],
                            "task": entry["task"],
                            "duration": entry["duration"],
                            "start_time": entry["start_time"]
                        })
            
            return anomalies
        except Exception as e:
            logger.error(f"Error detecting time tracking anomalies: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to detect time tracking anomalies")

    async def _update_user_embedding(self, user_id: str, task: str, duration: float):
        task_embedding = await self.text_embedding_service.get_embedding(task)
        if user_id not in self.user_embeddings:
            self.user_embeddings[user_id] = [0] * len(task_embedding)
        self.user_embeddings[user_id] = [a + b * duration for a, b in zip(self.user_embeddings[user_id], task_embedding)]

    async def _calculate_productivity_score(self, user_id: str, avg_duration: float, task_stats: Dict[str, Dict[str, float]]) -> float:
        task_diversity = len(task_stats)
        total_tasks = sum(stat["count"] for stat in task_stats.values())
        avg_task_duration = sum(stat["avg_duration"] for stat in task_stats.values()) / len(task_stats)
        
        productivity_score = (task_diversity / 10) * 0.3 + (total_tasks / 50) * 0.3 + (3600 / avg_task_duration) * 0.4
        return min(max(productivity_score, 0), 100)  # Normalize to 0-100 scale

    async def _generate_productivity_insights(self, user_id: str, productivity_score: float, task_stats: Dict[str, Dict[str, float]]) -> str:
        prompt = f"""
        Analyze the following productivity data and provide personalized insights and recommendations:
        
        User productivity score: {productivity_score}
        Task statistics: {task_stats}
        
        Please provide:
        1. An overall assessment of the user's productivity
        2. Specific insights on task performance
        3. Actionable recommendations for improvement
        4. Suggestions for optimizing time management
        5. Potential areas for skill development based on task performance
        """
        
        insights = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI productivity coach specializing in time management and task optimization."},
            {"role": "user", "content": prompt}
        ], "high")
        
        return insights.strip()

    async def _find_similar_tasks(self, user_id: str, target_task: str) -> List[str]:
        user_entries = await self.get_user_time_entries(user_id)
        unique_tasks = list(set(entry["task"] for entry in user_entries))
        
        target_embedding = await self.text_embedding_service.get_embedding(target_task)
        task_embeddings = [await self.text_embedding_service.get_embedding(task) for task in unique_tasks]
        
        similarities = [self._cosine_similarity(target_embedding, task_embedding) for task_embedding in task_embeddings]
        
        similar_tasks = [task for task, similarity in zip(unique_tasks, similarities) if similarity > 0.8]
        return similar_tasks

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (magnitude1 * magnitude2)

time_tracking_service = TimeTrackingService(get_llm_orchestrator(), get_text_embedding_service())

def get_time_tracking_service() -> TimeTrackingService:
    return time_tracking_service
