import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
import math
from collections import defaultdict
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
import random
import statistics

logger = logging.getLogger(__name__)

class CourseService:
    def __init__(self, text_embedding_service: TextEmbeddingService):
        self.text_embedding_service = text_embedding_service
        self.courses = {}  # Simulated database of courses
        self.user_course_progress = defaultdict(dict)  # Simulated user progress

    async def create_course(self, course_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            course_id = f"course_{len(self.courses) + 1}"
            course_embedding = await self._generate_course_embedding(course_data['title'], course_data['description'], course_data['topics'])
            difficulty = await self._estimate_course_difficulty(course_data['title'], course_data['description'], course_data['topics'])
            
            self.courses[course_id] = {
                **course_data,
                "embedding": course_embedding,
                "difficulty": difficulty
            }
            return {"course_id": course_id, "message": "Course created successfully"}
        except Exception as e:
            logger.error(f"Error creating course: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create course")

    async def get_course(self, course_id: str) -> Dict[str, Any]:
        if course_id not in self.courses:
            raise HTTPException(status_code=404, detail="Course not found")
        return self.courses[course_id]

    async def update_course(self, course_id: str, course_data: Dict[str, Any]) -> Dict[str, Any]:
        if course_id not in self.courses:
            raise HTTPException(status_code=404, detail="Course not found")
        
        try:
            course_embedding = await self._generate_course_embedding(course_data['title'], course_data['description'], course_data['topics'])
            difficulty = await self._estimate_course_difficulty(course_data['title'], course_data['description'], course_data['topics'])
            
            self.courses[course_id] = {
                **course_data,
                "embedding": course_embedding,
                "difficulty": difficulty
            }
            return {"message": "Course updated successfully"}
        except Exception as e:
            logger.error(f"Error updating course: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to update course")

    async def delete_course(self, course_id: str) -> Dict[str, Any]:
        if course_id not in self.courses:
            raise HTTPException(status_code=404, detail="Course not found")
        
        del self.courses[course_id]
        return {"message": "Course deleted successfully"}

    async def recommend_courses(self, user_id: str, num_recommendations: int = 5) -> List[Dict[str, Any]]:
        if not self.user_course_progress[user_id]:
            return await self._get_popular_courses(num_recommendations)

        user_interests = await self._generate_user_interests(user_id)
        course_scores = []

        for course_id, course in self.courses.items():
            if course_id not in self.user_course_progress[user_id]:
                similarity = self._cosine_similarity(user_interests, course['embedding'])
                difficulty_score = self._calculate_difficulty_score(user_id, course['difficulty'])
                popularity_score = await self._calculate_popularity_score(course_id)
                total_score = similarity * 0.6 + difficulty_score * 0.2 + popularity_score * 0.2
                course_scores.append((course_id, total_score))

        course_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [
            {
                "course_id": course_id,
                "title": self.courses[course_id]['title'],
                "score": score
            }
            for course_id, score in course_scores[:num_recommendations]
        ]

        return recommendations

    async def update_user_progress(self, user_id: str, course_id: str, progress: float) -> Dict[str, Any]:
        if course_id not in self.courses:
            raise HTTPException(status_code=404, detail="Course not found")
        
        self.user_course_progress[user_id][course_id] = progress
        return {"message": "User progress updated successfully"}

    async def get_user_progress(self, user_id: str) -> Dict[str, float]:
        return self.user_course_progress[user_id]

    async def _generate_course_embedding(self, title: str, description: str, topics: List[str]) -> List[float]:
        combined_text = f"{title} {description} {' '.join(topics)}"
        words = combined_text.lower().split()
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        embedding = []
        total_words = len(words)
        for word, freq in word_freq.items():
            tf = freq / total_words
            idf = math.log(len(self.courses) + 1 / (sum(1 for c in self.courses.values() if word in c['title'] + ' ' + c['description'] + ' ' + ' '.join(c['topics'])) + 1))
            embedding.append(tf * idf)
        
        # Normalize the embedding
        magnitude = math.sqrt(sum(x**2 for x in embedding))
        return [x / magnitude for x in embedding] if magnitude > 0 else [0] * len(embedding)

    async def _estimate_course_difficulty(self, title: str, description: str, topics: List[str]) -> float:
        combined_text = f"{title} {description} {' '.join(topics)}"
        words = combined_text.split()
        avg_word_length = sum(len(word) for word in words) / len(words)
        unique_words = len(set(words))
        complex_words = sum(1 for word in words if len(word) > 8)
        
        difficulty = (
            avg_word_length * 0.3 +
            (unique_words / len(words)) * 0.3 +
            (complex_words / len(words)) * 0.2 +
            (len(topics) / 10) * 0.2
        ) * 10
        
        return min(max(difficulty, 1), 10)  # Scale difficulty from 1 to 10

    async def _generate_user_interests(self, user_id: str) -> List[float]:
        completed_courses = [course_id for course_id, progress in self.user_course_progress[user_id].items() if progress == 100]
        if not completed_courses:
            return [0] * 100  # Return a zero vector if no courses completed

        course_embeddings = [self.courses[course_id]['embedding'] for course_id in completed_courses]
        user_embedding = [sum(emb[i] for emb in course_embeddings) / len(course_embeddings) for i in range(len(course_embeddings[0]))]
        
        # Add some randomness to user interests
        randomness = [random.uniform(-0.1, 0.1) for _ in range(len(user_embedding))]
        user_embedding = [x + r for x, r in zip(user_embedding, randomness)]
        
        # Normalize the embedding
        magnitude = math.sqrt(sum(x**2 for x in user_embedding))
        return [x / magnitude for x in user_embedding] if magnitude > 0 else [0] * len(user_embedding)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0

    async def _get_popular_courses(self, num_courses: int) -> List[Dict[str, Any]]:
        course_popularities = [(course_id, await self._calculate_popularity_score(course_id)) 
                               for course_id in self.courses]
        course_popularities.sort(key=lambda x: x[1], reverse=True)
        popular_courses = course_popularities[:num_courses]
        return [
            {
                "course_id": course_id,
                "title": self.courses[course_id]['title'],
                "popularity_score": popularity
            }
            for course_id, popularity in popular_courses
        ]

    def _calculate_difficulty_score(self, user_id: str, course_difficulty: float) -> float:
        user_completed_courses = [course_id for course_id, progress in self.user_course_progress[user_id].items() if progress == 100]
        if not user_completed_courses:
            return 1.0  # If user has no completed courses, return max score
        
        user_avg_difficulty = statistics.mean(self.courses[course_id]['difficulty'] for course_id in user_completed_courses)
        difficulty_difference = abs(course_difficulty - user_avg_difficulty)
        
        return 1.0 / (1.0 + difficulty_difference)  # Higher score for courses closer to user's average difficulty

    async def _calculate_popularity_score(self, course_id: str) -> float:
        num_enrolled = sum(1 for progress in self.user_course_progress.values() if course_id in progress)
        total_users = len(self.user_course_progress)
        return num_enrolled / total_users if total_users > 0 else 0

course_service = CourseService(get_text_embedding_service())

def get_course_service() -> CourseService:
    return course_service
