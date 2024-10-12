import os
import requests
import json
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class PercipioIntegrationService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.base_url = "https://api.percipio.com/content-discovery/v1"
        self.bearer_token = os.getenv("PERCIPIO_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.content_embeddings = {}

    async def get_courses(self, offset: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            url = f"{self.base_url}/courses?offset={offset}&max={limit}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            courses = response.json()
            
            # Generate embeddings for course descriptions
            for course in courses:
                course_id = course['id']
                course_description = course.get('description', '')
                self.content_embeddings[course_id] = await self.text_embedding_service.get_embedding(course_description)
            
            return courses
        except requests.RequestException as e:
            logger.error(f"Error fetching courses from Percipio: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch courses from Percipio")

    async def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        try:
            url = f"{self.base_url}/users/{user_id}/progress"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching user progress from Percipio: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch user progress from Percipio")

    async def recommend_content(self, user_id: str, user_interests: List[str]) -> List[Dict[str, Any]]:
        try:
            # Get user's current progress
            user_progress = await self.get_user_progress(user_id)
            completed_courses = [course['id'] for course in user_progress.get('completedCourses', [])]

            # Generate user interest embedding
            user_interest_text = " ".join(user_interests)
            user_embedding = await self.text_embedding_service.get_embedding(user_interest_text)

            # Find courses similar to user interests
            recommendations = []
            for course_id, course_embedding in self.content_embeddings.items():
                if course_id not in completed_courses:
                    similarity = cosine_similarity([user_embedding], [course_embedding])[0][0]
                    recommendations.append((course_id, similarity))

            # Sort recommendations by similarity
            recommendations.sort(key=lambda x: x[1], reverse=True)
            top_recommendations = recommendations[:5]

            # Get course details for top recommendations
            recommended_courses = []
            for course_id, similarity in top_recommendations:
                course_details = await self.get_course_details(course_id)
                recommended_courses.append({
                    "id": course_id,
                    "title": course_details.get('title', ''),
                    "description": course_details.get('description', ''),
                    "similarity_score": similarity
                })

            return recommended_courses
        except Exception as e:
            logger.error(f"Error recommending content: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to recommend content")

    async def get_course_details(self, course_id: str) -> Dict[str, Any]:
        try:
            url = f"{self.base_url}/courses/{course_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching course details from Percipio: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch course details from Percipio")

    async def generate_adaptive_learning_path(self, user_id: str, target_course_id: str) -> List[Dict[str, Any]]:
        try:
            user_progress = await self.get_user_progress(user_id)
            completed_courses = [course['id'] for course in user_progress.get('completedCourses', [])]
            target_course = await self.get_course_details(target_course_id)

            # Generate learning path using LLM
            prompt = f"Generate an adaptive learning path for a user with the following completed courses: {', '.join(completed_courses)}\n\n"
            prompt += f"The target course is: {target_course['title']}\n"
            prompt += "Provide a step-by-step learning path with course IDs and brief explanations."

            learning_path = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an AI tutor creating personalized learning paths."},
                {"role": "user", "content": prompt}
            ], "high")

            # Parse the learning path
            steps = learning_path.split("\n")
            parsed_path = []
            for step in steps:
                if step.strip():
                    course_id, explanation = step.split(":", 1)
                    parsed_path.append({"course_id": course_id.strip(), "explanation": explanation.strip()})

            return parsed_path
        except Exception as e:
            logger.error(f"Error generating adaptive learning path: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate adaptive learning path")

    async def estimate_content_difficulty(self, content_id: str) -> Dict[str, Any]:
        try:
            content_details = await self.get_course_details(content_id)
            content_description = content_details.get('description', '')

            prompt = f"Analyze the following course description and estimate its difficulty level (beginner, intermediate, or advanced). Provide a brief explanation for your estimation.\n\nCourse description: {content_description}"

            analysis = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an AI expert in educational content analysis."},
                {"role": "user", "content": prompt}
            ], "medium")

            # Parse the analysis
            difficulty, explanation = analysis.split("\n", 1)
            difficulty = difficulty.lower().strip()

            return {
                "content_id": content_id,
                "estimated_difficulty": difficulty,
                "explanation": explanation.strip()
            }
        except Exception as e:
            logger.error(f"Error estimating content difficulty: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to estimate content difficulty")

percipio_integration_service = PercipioIntegrationService(get_llm_orchestrator(), get_text_embedding_service())

def get_percipio_integration_service() -> PercipioIntegrationService:
    return percipio_integration_service
