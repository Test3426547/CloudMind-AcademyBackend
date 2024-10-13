import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
import random
import math
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)

class CoursePrerequisitesService:
    def __init__(self):
        self.courses = {}  # Simulated database of courses
        self.user_profiles = {}  # Simulated user profiles for personalization

    async def add_course(self, course_id: str, title: str, description: str, prerequisites: List[str], topics: List[str]) -> Dict[str, Any]:
        try:
            course_embedding = await self._generate_course_embedding(title, description, topics)
            difficulty = await self._estimate_course_difficulty(title, description, topics)
            self.courses[course_id] = {
                "title": title,
                "description": description,
                "prerequisites": prerequisites,
                "embedding": course_embedding,
                "difficulty": difficulty,
                "topics": topics
            }
            return {"message": f"Course {course_id} added successfully"}
        except Exception as e:
            logger.error(f"Error adding course: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to add course")

    async def get_course_prerequisites(self, course_id: str) -> List[str]:
        if course_id not in self.courses:
            raise HTTPException(status_code=404, detail="Course not found")
        return self.courses[course_id]["prerequisites"]

    async def suggest_prerequisites(self, course_id: str, num_suggestions: int = 3) -> List[str]:
        if course_id not in self.courses:
            raise HTTPException(status_code=404, detail="Course not found")

        target_course = self.courses[course_id]
        target_embedding = target_course["embedding"]

        similarities = []
        for id, course in self.courses.items():
            if id != course_id:
                similarity = self._cosine_similarity(target_embedding, course["embedding"])
                similarities.append((id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        suggested_prerequisites = [self.courses[id]["title"] for id, _ in similarities[:num_suggestions]]

        return suggested_prerequisites

    async def _generate_course_embedding(self, title: str, description: str, topics: List[str]) -> List[float]:
        # Enhanced embedding generation using TF-IDF-like approach with topic weighting
        words = (title + " " + description + " " + " ".join(topics)).lower().split()
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        embedding = []
        total_courses = len(self.courses) + 1  # Add 1 to avoid division by zero for the first course
        for word, freq in word_freq.items():
            tf = freq / len(words)
            idf = math.log(total_courses / (sum(1 for c in self.courses.values() if word in c['title'] + ' ' + c['description'] + ' ' + ' '.join(c['topics'])) + 1))
            topic_weight = 2 if word in topics else 1  # Give more weight to topic words
            embedding.append(tf * idf * topic_weight)
        
        # Normalize the embedding
        magnitude = math.sqrt(sum(x**2 for x in embedding))
        return [x / magnitude for x in embedding] if magnitude > 0 else [0] * len(embedding)

    async def _estimate_course_difficulty(self, title: str, description: str, topics: List[str]) -> float:
        # Enhanced difficulty estimation based on word complexity, length, and topics
        words = (title + " " + description).lower().split()
        avg_word_length = sum(len(word) for word in words) / len(words)
        complex_words = sum(1 for word in words if len(word) > 8)
        topic_difficulty = sum(len(topic.split()) for topic in topics) / len(topics)
        
        difficulty = (
            avg_word_length * 0.3 +
            complex_words / len(words) * 0.3 +
            topic_difficulty * 0.4
        ) * 10
        
        return min(max(difficulty, 1), 10)  # Scale difficulty from 1 to 10

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0

    async def analyze_learning_path(self, course_ids: List[str]) -> Dict[str, Any]:
        if not all(course_id in self.courses for course_id in course_ids):
            raise HTTPException(status_code=404, detail="One or more courses not found")

        course_embeddings = [self.courses[id]["embedding"] for id in course_ids]
        course_difficulties = [self.courses[id]["difficulty"] for id in course_ids]
        
        # Calculate pairwise distances and difficulty changes
        distances = [
            [self._euclidean_distance(emb1, emb2) for emb2 in course_embeddings]
            for emb1 in course_embeddings
        ]
        difficulty_changes = [
            course_difficulties[i+1] - course_difficulties[i]
            for i in range(len(course_difficulties) - 1)
        ]
        
        # Calculate the total path length and difficulty progression
        path_length = sum(distances[i][i+1] for i in range(len(distances)-1))
        difficulty_progression = sum(difficulty_changes)
        
        # Identify potential gaps and difficulty spikes in the learning path
        gap_threshold = statistics.mean([distances[i][i+1] for i in range(len(distances)-1)]) * 1.5
        difficulty_spike_threshold = statistics.mean(difficulty_changes) * 2
        gaps = [
            {"from": course_ids[i], "to": course_ids[i+1]}
            for i in range(len(distances) - 1)
            if distances[i][i+1] > gap_threshold
        ]
        difficulty_spikes = [
            {"from": course_ids[i], "to": course_ids[i+1]}
            for i in range(len(difficulty_changes))
            if difficulty_changes[i] > difficulty_spike_threshold
        ]
        
        # Calculate topic coverage
        all_topics = set()
        covered_topics = set()
        for course_id in course_ids:
            all_topics.update(self.courses[course_id]["topics"])
            covered_topics.update(self.courses[course_id]["topics"])
        
        topic_coverage = len(covered_topics) / len(all_topics) if all_topics else 1

        return {
            "path_length": path_length,
            "difficulty_progression": difficulty_progression,
            "potential_gaps": gaps,
            "difficulty_spikes": difficulty_spikes,
            "topic_coverage": topic_coverage,
            "total_topics": len(all_topics),
            "covered_topics": len(covered_topics)
        }

    async def generate_personalized_learning_path(self, user_id: str, target_course_id: str, max_courses: int = 5) -> List[str]:
        if target_course_id not in self.courses:
            raise HTTPException(status_code=404, detail="Target course not found")

        if user_id not in self.user_profiles:
            raise HTTPException(status_code=404, detail="User profile not found")

        user_profile = self.user_profiles[user_id]
        target_course = self.courses[target_course_id]
        
        # Calculate the skill gap
        user_skills = set(user_profile["skills"])
        required_skills = set(target_course["topics"])
        skill_gap = required_skills - user_skills

        # Find courses that cover the skill gap
        relevant_courses = []
        for course_id, course in self.courses.items():
            if course_id != target_course_id:
                covered_skills = set(course["topics"]) & skill_gap
                if covered_skills:
                    relevance_score = len(covered_skills) / len(skill_gap)
                    relevant_courses.append((course_id, relevance_score, course["difficulty"]))

        # Sort courses by relevance and difficulty
        relevant_courses.sort(key=lambda x: (x[1], -x[2]), reverse=True)

        # Generate the learning path
        learning_path = []
        covered_skills = set()
        for course_id, _, _ in relevant_courses:
            if len(learning_path) >= max_courses:
                break
            if not skill_gap.issubset(covered_skills):
                learning_path.append(course_id)
                covered_skills.update(self.courses[course_id]["topics"])

        learning_path.append(target_course_id)
        return learning_path

    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

    async def update_user_profile(self, user_id: str, skills: List[str], completed_courses: List[str]):
        self.user_profiles[user_id] = {
            "skills": skills,
            "completed_courses": completed_courses
        }

    async def recommend_courses(self, user_id: str, num_recommendations: int = 5) -> List[Dict[str, Any]]:
        if user_id not in self.user_profiles:
            raise HTTPException(status_code=404, detail="User profile not found")

        user_profile = self.user_profiles[user_id]
        user_skills = set(user_profile["skills"])
        completed_courses = set(user_profile["completed_courses"])

        # Calculate course scores based on relevance and novelty
        course_scores = []
        for course_id, course in self.courses.items():
            if course_id not in completed_courses:
                course_skills = set(course["topics"])
                relevance_score = len(user_skills & course_skills) / len(course_skills)
                novelty_score = len(course_skills - user_skills) / len(course_skills)
                total_score = relevance_score * 0.6 + novelty_score * 0.4
                course_scores.append((course_id, total_score))

        # Sort courses by score and return top recommendations
        course_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [
            {
                "course_id": course_id,
                "title": self.courses[course_id]["title"],
                "score": score
            }
            for course_id, score in course_scores[:num_recommendations]
        ]

        return recommendations

course_prerequisites_service = CoursePrerequisitesService()

def get_course_prerequisites_service() -> CoursePrerequisitesService:
    return course_prerequisites_service
