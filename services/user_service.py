import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
import random
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

class UserService:
    def __init__(self):
        self.users = {}
        self.user_embeddings = {}
        self.user_preferences = defaultdict(dict)

    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            user_id = f"user_{len(self.users) + 1}"
            self.users[user_id] = user_data
            self.user_embeddings[user_id] = self._generate_user_embedding(user_data)
            return {"user_id": user_id, "message": "User created successfully"}
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create user")

    async def get_user(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.users:
            raise HTTPException(status_code=404, detail="User not found")
        return self.users[user_id]

    async def update_user(self, user_id: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        if user_id not in self.users:
            raise HTTPException(status_code=404, detail="User not found")
        
        try:
            self.users[user_id].update(user_data)
            self.user_embeddings[user_id] = self._generate_user_embedding(self.users[user_id])
            return {"message": "User updated successfully"}
        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to update user")

    async def delete_user(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.users:
            raise HTTPException(status_code=404, detail="User not found")
        
        del self.users[user_id]
        del self.user_embeddings[user_id]
        return {"message": "User deleted successfully"}

    def _generate_user_embedding(self, user_data: Dict[str, Any]) -> List[float]:
        # Simulated PyTorch-like embedding generation
        embedding_dim = 100
        embedding = [random.uniform(-1, 1) for _ in range(embedding_dim)]
        
        # Incorporate user data into the embedding
        for key, value in user_data.items():
            if isinstance(value, str):
                for char in value:
                    embedding[hash(char) % embedding_dim] += ord(char) / 1000
            elif isinstance(value, (int, float)):
                embedding[hash(key) % embedding_dim] += value / 100

        # Normalize the embedding (simulating PyTorch's F.normalize)
        magnitude = math.sqrt(sum(x**2 for x in embedding))
        return [x / magnitude for x in embedding]

    async def recommend_courses(self, user_id: str, num_recommendations: int = 5) -> List[Dict[str, Any]]:
        if user_id not in self.users:
            raise HTTPException(status_code=404, detail="User not found")

        user_embedding = self.user_embeddings[user_id]
        
        # Simulated course data (in a real scenario, this would come from a CourseService)
        courses = [
            {"id": f"course_{i}", "title": f"Course {i}", "embedding": [random.uniform(-1, 1) for _ in range(100)]}
            for i in range(20)
        ]

        # Calculate cosine similarity between user and courses (simulating TensorFlow operations)
        similarities = [
            (course["id"], self._cosine_similarity(user_embedding, course["embedding"]))
            for course in courses
        ]

        # Sort by similarity and get top recommendations
        similarities.sort(key=lambda x: x[1], reverse=True)
        recommendations = [
            {"course_id": course_id, "similarity": similarity}
            for course_id, similarity in similarities[:num_recommendations]
        ]

        return recommendations

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0

    async def analyze_user_behavior(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.users:
            raise HTTPException(status_code=404, detail="User not found")

        user_data = self.users[user_id]
        user_embedding = self.user_embeddings[user_id]

        # Simulated TensorFlow-like behavior analysis
        behavior_score = sum(user_embedding) / len(user_embedding)
        engagement_level = self._sigmoid(behavior_score * 10)

        # Simulated HuggingFace Transformers text generation for personalized message
        personalized_message = self._generate_personalized_message(user_data, engagement_level)

        return {
            "user_id": user_id,
            "engagement_level": engagement_level,
            "personalized_message": personalized_message
        }

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def _generate_personalized_message(self, user_data: Dict[str, Any], engagement_level: float) -> str:
        name = user_data.get("name", "there")
        if engagement_level > 0.8:
            return f"Great job, {name}! Your high engagement is impressive. Keep up the excellent work!"
        elif engagement_level > 0.5:
            return f"Hello {name}! You're making good progress. Consider exploring more courses to boost your learning."
        else:
            return f"Hi {name}! We've noticed your engagement has been low. How can we help improve your learning experience?"

    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, str]:
        if user_id not in self.users:
            raise HTTPException(status_code=404, detail="User not found")

        self.user_preferences[user_id].update(preferences)
        
        # Simulated NumPy-like preference analysis
        preference_vector = [0] * 10
        for key, value in preferences.items():
            preference_vector[hash(key) % 10] += hash(str(value)) % 100 / 100

        # Update user embedding with new preferences (simulating PyTorch operations)
        user_embedding = self.user_embeddings[user_id]
        updated_embedding = [
            (ue + pv) / 2 for ue, pv in zip(user_embedding, preference_vector * 10)
        ]
        self.user_embeddings[user_id] = self._normalize_vector(updated_embedding)

        return {"message": "User preferences updated successfully"}

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        magnitude = math.sqrt(sum(x**2 for x in vector))
        return [x / magnitude for x in vector] if magnitude > 0 else vector

user_service = UserService()

def get_user_service() -> UserService:
    return user_service
