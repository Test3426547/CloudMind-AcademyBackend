import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
import random
import math

logger = logging.getLogger(__name__)

class CoursePrerequisiteService:
    def __init__(self):
        self.prerequisites = {}
        self.user_progress = {}
        self.course_embeddings = {}

    async def add_prerequisite(self, prerequisite: Dict[str, Any]) -> None:
        course_id = prerequisite['course_id']
        if course_id not in self.prerequisites:
            self.prerequisites[course_id] = []
        self.prerequisites[course_id].append(prerequisite['prerequisite_id'])
        await self._update_course_embedding(course_id)

    async def get_prerequisites(self, course_id: str) -> List[str]:
        return self.prerequisites.get(course_id, [])

    async def update_user_progress(self, progress: Dict[str, Any]) -> None:
        user_id = progress['user_id']
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}
        self.user_progress[user_id][progress['course_id']] = progress['progress']

    async def get_user_progress(self, user_id: str, course_id: str) -> Dict[str, Any]:
        return self.user_progress.get(user_id, {}).get(course_id, {"progress": 0})

    async def check_prerequisites_met(self, user_id: str, course_id: str) -> bool:
        prerequisites = await self.get_prerequisites(course_id)
        user_courses = self.user_progress.get(user_id, {})
        return all(user_courses.get(prereq, {}).get('progress', 0) >= 100 for prereq in prerequisites)

    async def remove_prerequisite(self, course_id: str, prerequisite_id: str) -> None:
        if course_id in self.prerequisites:
            self.prerequisites[course_id] = [p for p in self.prerequisites[course_id] if p != prerequisite_id]
        await self._update_course_embedding(course_id)

    async def _update_course_embedding(self, course_id: str) -> None:
        # Simulate embedding creation using a simple hash function
        course_content = f"Course: {course_id}\nPrerequisites: {', '.join(self.prerequisites.get(course_id, []))}"
        self.course_embeddings[course_id] = self._simple_hash(course_content)

    def _simple_hash(self, text: str) -> List[float]:
        # Simple hash function to create a pseudo-embedding
        hash_value = hash(text)
        return [((hash_value >> i) & 1) * 2 - 1 for i in range(64)]  # 64-dimensional pseudo-embedding

    async def recommend_prerequisites(self, course_id: str, num_recommendations: int = 3) -> List[str]:
        if course_id not in self.course_embeddings:
            await self._update_course_embedding(course_id)

        target_embedding = self.course_embeddings[course_id]
        similarities = []

        for other_course, embedding in self.course_embeddings.items():
            if other_course != course_id:
                similarity = self._cosine_similarity(target_embedding, embedding)
                similarities.append((other_course, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [course for course, _ in similarities[:num_recommendations]]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (magnitude1 * magnitude2)

    async def generate_adaptive_learning_path(self, user_id: str, target_course_id: str) -> List[str]:
        user_courses = self.user_progress.get(user_id, {})
        completed_courses = [course for course, progress in user_courses.items() if progress['progress'] >= 100]
        
        learning_path = []
        current_course = target_course_id

        while current_course not in completed_courses:
            learning_path.append(current_course)
            prerequisites = await self.get_prerequisites(current_course)
            
            if not prerequisites:
                break

            # Find the prerequisite with the highest progress or the most relevant one
            best_prerequisite = max(prerequisites, key=lambda p: user_courses.get(p, {}).get('progress', 0))
            current_course = best_prerequisite

        learning_path.reverse()
        return learning_path

    async def estimate_course_difficulty(self, course_id: str) -> float:
        prerequisites = await self.get_prerequisites(course_id)
        num_prerequisites = len(prerequisites)
        avg_prerequisite_difficulty = 0

        if prerequisites:
            difficulties = [await self.estimate_course_difficulty(p) for p in prerequisites]
            avg_prerequisite_difficulty = sum(difficulties) / len(difficulties)

        # Enhanced difficulty calculation based on prerequisites and course content
        base_difficulty = 1.0
        content_complexity = self._estimate_content_complexity(course_id)
        difficulty = (
            base_difficulty +
            (num_prerequisites * 0.1) +
            (avg_prerequisite_difficulty * 0.2) +
            (content_complexity * 0.3)
        )
        return min(difficulty, 5.0)  # Cap difficulty at 5.0

    def _estimate_content_complexity(self, course_id: str) -> float:
        # Simulate content complexity estimation
        # In a real scenario, this could analyze course materials, topics, etc.
        return random.uniform(0.5, 1.5)

    async def analyze_learning_gaps(self, user_id: str, target_course_id: str) -> Dict[str, Any]:
        learning_path = await self.generate_adaptive_learning_path(user_id, target_course_id)
        user_courses = self.user_progress.get(user_id, {})
        
        gaps = []
        for course in learning_path:
            progress = user_courses.get(course, {}).get('progress', 0)
            if progress < 100:
                difficulty = await self.estimate_course_difficulty(course)
                estimated_time = self._estimate_completion_time(difficulty, progress)
                gaps.append({
                    "course_id": course,
                    "current_progress": progress,
                    "estimated_difficulty": difficulty,
                    "estimated_completion_time": estimated_time
                })

        return {
            "user_id": user_id,
            "target_course": target_course_id,
            "learning_gaps": gaps,
            "recommended_study_plan": self._generate_study_plan(gaps)
        }

    def _estimate_completion_time(self, difficulty: float, current_progress: float) -> int:
        # Estimate completion time in hours based on difficulty and current progress
        base_time = 10  # Base time for an average course
        remaining_percentage = (100 - current_progress) / 100
        return int(base_time * difficulty * remaining_percentage)

    def _generate_study_plan(self, gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        study_plan = []
        for gap in gaps:
            study_plan.append({
                "course_id": gap["course_id"],
                "focus_areas": self._identify_focus_areas(gap["course_id"]),
                "recommended_resources": self._recommend_resources(gap["course_id"]),
                "estimated_study_time": gap["estimated_completion_time"]
            })
        return study_plan

    def _identify_focus_areas(self, course_id: str) -> List[str]:
        # Simulate identifying focus areas for a course
        # In a real scenario, this could analyze course content and user performance
        focus_areas = ["Fundamentals", "Advanced Concepts", "Practical Applications"]
        return random.sample(focus_areas, k=2)

    def _recommend_resources(self, course_id: str) -> List[str]:
        # Simulate recommending resources for a course
        # In a real scenario, this could be based on course content and user preferences
        resources = [
            "Online tutorial: Introduction to " + course_id,
            "Interactive coding exercises for " + course_id,
            "Video lecture series on advanced " + course_id + " topics",
            "Recommended textbook: Mastering " + course_id
        ]
        return random.sample(resources, k=3)

course_prerequisite_service = CoursePrerequisiteService()

def get_course_prerequisite_service() -> CoursePrerequisiteService:
    return course_prerequisite_service
