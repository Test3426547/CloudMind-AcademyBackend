from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.ai_tutor import AITutorService, get_ai_tutor_service
from typing import List, Dict
from pydantic import BaseModel, Field
from fastapi_limiter.depends import RateLimiter
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=5, max_length=1000)

class ConceptExplanationRequest(BaseModel):
    concept: str = Field(..., min_length=5, max_length=100)

class CollaborationRequest(BaseModel):
    message: str = Field(..., min_length=5, max_length=1000)
    context: List[Dict[str, str]] = []

class CollaborationSummaryRequest(BaseModel):
    messages: List[str] = Field(..., min_items=1, max_items=100)

@router.post("/ai-tutor/chat")
async def chat_with_tutor(
    chat_message: ChatMessage,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        result = await ai_tutor_service.chat_with_tutor(chat_message.message)
        logger.info(f"AI Tutor chat completed for user {user.id}")
        return result
    except ValueError as e:
        logger.warning(f"Invalid input for chat_with_tutor: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in chat_with_tutor: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during the chat. Please try again later.")

@router.post("/ai-tutor/explain-concept")
async def explain_concept(
    request: ConceptExplanationRequest,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        result = await ai_tutor_service.explain_concept(request.concept)
        logger.info(f"Concept explanation completed for user {user.id}")
        return result
    except ValueError as e:
        logger.warning(f"Invalid input for explain_concept: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in explain_concept: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while explaining the concept. Please try again later.")

@router.post("/ai-tutor/collaboration-assistance")
async def get_collaboration_assistance(
    request: CollaborationRequest,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        result = await ai_tutor_service.get_collaboration_assistance(request.message, request.context)
        logger.info(f"Collaboration assistance provided for user {user.id}")
        return {"response": result}
    except ValueError as e:
        logger.warning(f"Invalid input for get_collaboration_assistance: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in get_collaboration_assistance: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during collaboration assistance. Please try again later.")

@router.post("/ai-tutor/summarize-collaboration")
async def summarize_collaboration(
    request: CollaborationSummaryRequest,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=3, seconds=60))
):
    try:
        result = await ai_tutor_service.summarize_collaboration(request.messages)
        logger.info(f"Collaboration summary generated for user {user.id}")
        return {"summary": result}
    except ValueError as e:
        logger.warning(f"Invalid input for summarize_collaboration: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in summarize_collaboration: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while summarizing the collaboration. Please try again later.")
