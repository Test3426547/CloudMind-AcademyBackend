from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.openrouter_service import OpenRouterService, get_openrouter_service
from typing import List, Dict, Any
from pydantic import BaseModel

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class CompletionRequest(BaseModel):
    messages: List[Dict[str, str]]

@router.get("/openrouter/models")
async def get_openrouter_models(
    user: User = Depends(oauth2_scheme),
    openrouter_service: OpenRouterService = Depends(get_openrouter_service)
):
    try:
        models = await openrouter_service.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/openrouter/completion")
async def generate_openrouter_completion(
    request: CompletionRequest,
    model: str,
    user: User = Depends(oauth2_scheme),
    openrouter_service: OpenRouterService = Depends(get_openrouter_service)
):
    try:
        completion = await openrouter_service.generate_completion(model, request.messages)
        return completion
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/openrouter/completion/gpt4")
async def generate_gpt4_completion(
    request: CompletionRequest,
    user: User = Depends(oauth2_scheme),
    openrouter_service: OpenRouterService = Depends(get_openrouter_service)
):
    try:
        completion = await openrouter_service.generate_completion_gpt4(request.messages)
        return completion
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/openrouter/completion/claude3")
async def generate_claude3_completion(
    request: CompletionRequest,
    user: User = Depends(oauth2_scheme),
    openrouter_service: OpenRouterService = Depends(get_openrouter_service)
):
    try:
        completion = await openrouter_service.generate_completion_claude3(request.messages)
        return completion
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/openrouter/completion/mistral")
async def generate_mistral_completion(
    request: CompletionRequest,
    user: User = Depends(oauth2_scheme),
    openrouter_service: OpenRouterService = Depends(get_openrouter_service)
):
    try:
        completion = await openrouter_service.generate_completion_mistral(request.messages)
        return completion
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/openrouter/completion/llama")
async def generate_llama_completion(
    request: CompletionRequest,
    user: User = Depends(oauth2_scheme),
    openrouter_service: OpenRouterService = Depends(get_openrouter_service)
):
    try:
        completion = await openrouter_service.generate_completion_llama(request.messages)
        return completion
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
