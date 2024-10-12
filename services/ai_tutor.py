import asyncio
from typing import List, Dict, Any, Optional
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
from fastapi import HTTPException
import logging
import time

logger = logging.getLogger(__name__)

class AITutorService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.user_contexts = {}
        self.last_interaction_time = {}

    async def chat_with_tutor(self, user_id: str, message: str) -> Dict[str, Any]:
        try:
            context = self.user_contexts.get(user_id, [])
            context.append({"role": "user", "content": message})

            # Implement adaptive response generation based on user's interaction history
            response_quality = self._determine_response_quality(user_id)

            system_message = (
                "You are an advanced AI tutor with expertise in various subjects. "
                f"Provide a {response_quality} response to help the student learn effectively."
            )

            response = await self.llm_orchestrator.process_request(
                context,
                "high",
                system_message=system_message
            )

            context.append({"role": "assistant", "content": response})
            self.user_contexts[user_id] = context[-10:]  # Keep last 10 messages for context
            self.last_interaction_time[user_id] = time.time()

            return {"response": response, "context": context}
        except Exception as e:
            logger.error(f"Error in chat_with_tutor: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred during the chat with the AI tutor")

    async def explain_concept(self, concept: str) -> Dict[str, Any]:
        try:
            context = [
                {"role": "system", "content": "You are an expert tutor. Explain the given concept in detail, providing examples and analogies to aid understanding."},
                {"role": "user", "content": f"Explain the concept of {concept} in detail."}
            ]

            explanation = await self.llm_orchestrator.process_request(context, "high")
            
            # Generate quiz questions based on the explanation
            quiz_context = [
                {"role": "system", "content": "Generate 3 multiple-choice quiz questions based on the following explanation:"},
                {"role": "user", "content": explanation}
            ]
            quiz_questions = await self.llm_orchestrator.process_request(quiz_context, "medium")

            # Generate a mind map for visual learners
            mind_map_context = [
                {"role": "system", "content": "Create a textual representation of a mind map for the following concept:"},
                {"role": "user", "content": explanation}
            ]
            mind_map = await self.llm_orchestrator.process_request(mind_map_context, "medium")

            return {
                "concept": concept,
                "explanation": explanation,
                "quiz_questions": quiz_questions,
                "mind_map": mind_map
            }
        except Exception as e:
            logger.error(f"Error in explain_concept: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while explaining the concept")

    async def generate_personalized_learning_path(self, user_id: str, subject: str, current_level: str) -> Dict[str, Any]:
        try:
            context = [
                {"role": "system", "content": "Generate a personalized learning path for the given subject and current level. Include recommended topics, resources, estimated time for each step, and potential challenges."},
                {"role": "user", "content": f"Create a detailed learning path for {subject} at {current_level} level."}
            ]

            learning_path = await self.llm_orchestrator.process_request(context, "high")
            
            # Generate milestones and progress tracking suggestions
            milestone_context = [
                {"role": "system", "content": "Based on the learning path, suggest key milestones and ways to track progress:"},
                {"role": "user", "content": learning_path}
            ]
            milestones = await self.llm_orchestrator.process_request(milestone_context, "medium")

            return {
                "user_id": user_id,
                "subject": subject,
                "current_level": current_level,
                "learning_path": learning_path,
                "milestones": milestones
            }
        except Exception as e:
            logger.error(f"Error in generate_personalized_learning_path: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while generating the personalized learning path")

    async def analyze_student_performance(self, user_id: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            context = [
                {"role": "system", "content": "Analyze the given student performance data and provide insights, recommendations, and areas for improvement. Include specific strategies for enhancing weak areas and leveraging strengths."},
                {"role": "user", "content": f"Analyze the following student performance data:\n{performance_data}"}
            ]

            analysis = await self.llm_orchestrator.process_request(context, "high")
            
            # Generate a study plan based on the analysis
            study_plan_context = [
                {"role": "system", "content": "Based on the performance analysis, create a tailored study plan:"},
                {"role": "user", "content": analysis}
            ]
            study_plan = await self.llm_orchestrator.process_request(study_plan_context, "medium")

            return {
                "user_id": user_id,
                "performance_data": performance_data,
                "analysis": analysis,
                "study_plan": study_plan
            }
        except Exception as e:
            logger.error(f"Error in analyze_student_performance: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while analyzing student performance")

    def _determine_response_quality(self, user_id: str) -> str:
        current_time = time.time()
        last_interaction = self.last_interaction_time.get(user_id, 0)
        time_since_last_interaction = current_time - last_interaction

        if time_since_last_interaction < 300:  # Less than 5 minutes
            return "concise"
        elif time_since_last_interaction < 3600:  # Less than 1 hour
            return "detailed"
        else:
            return "comprehensive"

ai_tutor_service = AITutorService(get_llm_orchestrator(), get_text_embedding_service())

def get_ai_tutor_service() -> AITutorService:
    return ai_tutor_service
