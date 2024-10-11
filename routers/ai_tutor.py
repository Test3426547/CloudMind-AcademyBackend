from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.ai_tutor import AITutorService, get_ai_tutor_service
from typing import List, Dict
from pydantic import BaseModel, Field

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service)
):
    try:
        result = await ai_tutor_service.chat_with_tutor(chat_message.message)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred during the chat")

@router.post("/ai-tutor/explain-concept")
async def explain_concept(
    request: ConceptExplanationRequest,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service)
):
    try:
        result = await ai_tutor_service.explain_concept(request.concept)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred while explaining the concept")

@router.post("/ai-tutor/collaboration-assistance")
async def get_collaboration_assistance(
    request: CollaborationRequest,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service)
):
    try:
        result = await ai_tutor_service.get_collaboration_assistance(request.message, request.context)
        return {"response": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred during collaboration assistance")

@router.post("/ai-tutor/summarize-collaboration")
async def summarize_collaboration(
    request: CollaborationSummaryRequest,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service)
):
    try:
        result = await ai_tutor_service.summarize_collaboration(request.messages)
        return {"summary": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred while summarizing the collaboration")
