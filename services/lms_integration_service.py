import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class LMSIntegrationService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.integrated_lms = {}
        self.course_embeddings = {}

    async def integrate_lms(self, lms_type: str, credentials: Dict[str, str]) -> Dict[str, Any]:
        try:
            # Simulate LMS integration
            await asyncio.sleep(1)
            integration_id = f"{lms_type}_{len(self.integrated_lms) + 1}"
            self.integrated_lms[integration_id] = {
                "type": lms_type,
                "status": "connected",
                "courses": []
            }
            return {"integration_id": integration_id, "status": "connected"}
        except Exception as e:
            logger.error(f"Error integrating LMS: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to integrate LMS")

    async def get_courses(self, integration_id: str) -> List[Dict[str, Any]]:
        if integration_id not in self.integrated_lms:
            raise HTTPException(status_code=404, detail="LMS integration not found")
        
        try:
            # Simulate fetching courses
            await asyncio.sleep(1)
            courses = [
                {"id": f"course_{i}", "title": f"Sample Course {i}", "description": f"Description for Course {i}"}
                for i in range(1, 6)
            ]
            self.integrated_lms[integration_id]["courses"] = courses
            return courses
        except Exception as e:
            logger.error(f"Error fetching courses: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch courses")

    async def sync_course(self, integration_id: str, course_id: str) -> Dict[str, Any]:
        if integration_id not in self.integrated_lms:
            raise HTTPException(status_code=404, detail="LMS integration not found")
        
        try:
            # Simulate course synchronization
            await asyncio.sleep(2)
            return {"status": "synced", "message": f"Course {course_id} synchronized successfully"}
        except Exception as e:
            logger.error(f"Error syncing course: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to sync course")

    async def analyze_course_content(self, course_id: str, content: str) -> Dict[str, Any]:
        try:
            # Generate course embedding
            embedding = await self.text_embedding_service.get_embedding(content)
            self.course_embeddings[course_id] = embedding

            # Analyze content using LLM
            analysis_prompt = f"Analyze the following course content and provide a summary, key topics, and difficulty level:\n\n{content}"
            analysis = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an expert in educational content analysis."},
                {"role": "user", "content": analysis_prompt}
            ], "high")

            return {
                "course_id": course_id,
                "summary": analysis.split("Summary:")[1].split("Key Topics:")[0].strip(),
                "key_topics": analysis.split("Key Topics:")[1].split("Difficulty Level:")[0].strip().split(", "),
                "difficulty_level": analysis.split("Difficulty Level:")[1].strip()
            }
        except Exception as e:
            logger.error(f"Error analyzing course content: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to analyze course content")

    async def generate_personalized_learning_path(self, user_id: str, target_course_id: str, user_background: str) -> List[Dict[str, Any]]:
        try:
            if target_course_id not in self.course_embeddings:
                raise HTTPException(status_code=404, detail="Target course not found or not analyzed")

            target_embedding = self.course_embeddings[target_course_id]
            
            # Find related courses
            related_courses = self._find_related_courses(target_course_id, 5)

            # Generate learning path using LLM
            path_prompt = f"Generate a personalized learning path for a user with the following background: {user_background}\n\n"
            path_prompt += f"The target course is: {target_course_id}\n"
            path_prompt += f"Related courses: {', '.join(related_courses)}\n\n"
            path_prompt += "Provide a step-by-step learning path with course IDs and brief explanations."

            learning_path = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an expert in creating personalized learning paths."},
                {"role": "user", "content": path_prompt}
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
            logger.error(f"Error generating personalized learning path: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate personalized learning path")

    async def recommend_courses(self, user_id: str, user_interests: List[str], completed_courses: List[str]) -> List[Dict[str, Any]]:
        try:
            # Generate user interest embedding
            user_interest_text = " ".join(user_interests)
            user_embedding = await self.text_embedding_service.get_embedding(user_interest_text)

            # Find courses similar to user interests
            similar_courses = self._find_similar_courses(user_embedding, 10)

            # Filter out completed courses
            recommended_courses = [course for course in similar_courses if course["id"] not in completed_courses]

            # Generate personalized recommendations using LLM
            recommendation_prompt = f"Generate personalized course recommendations for a user with the following interests: {', '.join(user_interests)}\n\n"
            recommendation_prompt += f"Recommended courses: {', '.join([course['id'] for course in recommended_courses[:5]])}\n\n"
            recommendation_prompt += "Provide brief explanations for why each course is recommended."

            recommendations = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an expert in recommending educational courses."},
                {"role": "user", "content": recommendation_prompt}
            ], "medium")

            # Parse the recommendations
            parsed_recommendations = []
            for course in recommended_courses[:5]:
                explanation = recommendations.split(course["id"] + ":")[1].split("\n")[0].strip()
                parsed_recommendations.append({
                    "course_id": course["id"],
                    "title": course["title"],
                    "explanation": explanation
                })

            return parsed_recommendations
        except Exception as e:
            logger.error(f"Error recommending courses: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to recommend courses")

    def _find_related_courses(self, course_id: str, num_courses: int) -> List[str]:
        target_embedding = self.course_embeddings[course_id]
        similarities = []
        for cid, embedding in self.course_embeddings.items():
            if cid != course_id:
                similarity = cosine_similarity([target_embedding], [embedding])[0][0]
                similarities.append((cid, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in similarities[:num_courses]]

    def _find_similar_courses(self, user_embedding: List[float], num_courses: int) -> List[Dict[str, Any]]:
        similarities = []
        for course_id, embedding in self.course_embeddings.items():
            similarity = cosine_similarity([user_embedding], [embedding])[0][0]
            similarities.append((course_id, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [{"id": cid, "title": f"Course {cid}"} for cid, _ in similarities[:num_courses]]

lms_integration_service = LMSIntegrationService(get_llm_orchestrator(), get_text_embedding_service())

def get_lms_integration_service() -> LMSIntegrationService:
    return lms_integration_service
