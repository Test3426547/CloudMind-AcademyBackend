from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.ai_tutor import AITutorService, get_ai_tutor_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)

class ConceptExplanationRequest(BaseModel):
    concept: str = Field(..., min_length=1, max_length=100)

class LearningPathRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=100)
    current_level: str = Field(..., min_length=1, max_length=50)

class PerformanceData(BaseModel):
    grades: Dict[str, float]
    completed_courses: List[str]
    time_spent: Dict[str, int]

@router.post("/ai-tutor/chat")
async def chat_with_tutor(
    chat_message: ChatMessage,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service),
):
    try:
        result = await ai_tutor_service.chat_with_tutor(user.id, chat_message.message)
        logger.info(f"AI Tutor chat completed for user {user.id}")
        return result
    except HTTPException as e:
        logger.error(f"HTTP error in chat_with_tutor: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in chat_with_tutor: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during the chat")

@router.post("/ai-tutor/explain-concept")
async def explain_concept(
    request: ConceptExplanationRequest,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service),
):
    try:
        result = await ai_tutor_service.explain_concept(request.concept)
        logger.info(f"Concept explanation completed for user {user.id}")
        return result
    except HTTPException as e:
        logger.error(f"HTTP error in explain_concept: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in explain_concept: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while explaining the concept")

@router.post("/ai-tutor/generate-learning-path")
async def generate_learning_path(
    request: LearningPathRequest,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service),
):
    try:
        result = await ai_tutor_service.generate_personalized_learning_path(user.id, request.subject, request.current_level)
        logger.info(f"Personalized learning path generated for user {user.id}")
        return result
    except HTTPException as e:
        logger.error(f"HTTP error in generate_learning_path: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in generate_learning_path: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the learning path")

@router.post("/ai-tutor/analyze-performance")
async def analyze_performance(
    performance_data: PerformanceData,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service),
):
    try:
        result = await ai_tutor_service.analyze_student_performance(user.id, performance_data.dict())
        logger.info(f"Performance analysis completed for user {user.id}")
        return result
    except HTTPException as e:
        logger.error(f"HTTP error in analyze_performance: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in analyze_performance: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while analyzing performance")
