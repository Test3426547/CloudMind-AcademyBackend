from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.openrouter_service import OpenRouterService, get_openrouter_service, OpenRouterException
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging
from fastapi.responses import StreamingResponse

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

logger = logging.getLogger(__name__)

class CompletionRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., min_items=1)

class BatchCompletionRequest(BaseModel):
    requests: List[Dict[str, Any]] = Field(..., min_items=1, max_items=10)

@router.get("/openrouter/models")
async def get_openrouter_models(
    user: User = Depends(oauth2_scheme),
    openrouter_service: OpenRouterService = Depends(get_openrouter_service)
):
    try:
        models = await openrouter_service.get_available_models()
        return {"models": models}
    except OpenRouterException as e:
        logger.error(f"Error fetching OpenRouter models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/openrouter/models/{model_id}")
async def get_model_details(
    model_id: str,
    user: User = Depends(oauth2_scheme),
    openrouter_service: OpenRouterService = Depends(get_openrouter_service)
):
    try:
        model_details = await openrouter_service.get_model_details(model_id)
        return model_details
    except OpenRouterException as e:
        logger.error(f"Error fetching model details: {str(e)}")
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
    except OpenRouterException as e:
        logger.error(f"Error generating OpenRouter completion: {str(e)}")
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
    except OpenRouterException as e:
        logger.error(f"Error generating GPT-4 completion: {str(e)}")
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
    except OpenRouterException as e:
        logger.error(f"Error generating Claude 3 completion: {str(e)}")
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
    except OpenRouterException as e:
        logger.error(f"Error generating Mistral completion: {str(e)}")
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
    except OpenRouterException as e:
        logger.error(f"Error generating Llama completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/openrouter/batch-completion")
async def batch_generate_completions(
    request: BatchCompletionRequest,
    user: User = Depends(oauth2_scheme),
    openrouter_service: OpenRouterService = Depends(get_openrouter_service)
):
    try:
        completions = await openrouter_service.batch_generate_completions(request.requests)
        return {"completions": completions}
    except OpenRouterException as e:
        logger.error(f"Error generating batch completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/openrouter/usage")
async def get_usage_stats(
    user: User = Depends(oauth2_scheme),
    openrouter_service: OpenRouterService = Depends(get_openrouter_service)
):
    try:
        usage_stats = await openrouter_service.get_usage_stats()
        return usage_stats
    except OpenRouterException as e:
        logger.error(f"Error fetching usage stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/openrouter/stream-completion")
async def stream_completion(
    request: CompletionRequest,
    model: str,
    user: User = Depends(oauth2_scheme),
    openrouter_service: OpenRouterService = Depends(get_openrouter_service)
):
    try:
        return StreamingResponse(openrouter_service.stream_completion(model, request.messages), media_type="text/event-stream")
    except OpenRouterException as e:
        logger.error(f"Error streaming completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/openrouter/cancel-request/{request_id}")
async def cancel_request(
    request_id: str,
    user: User = Depends(oauth2_scheme),
    openrouter_service: OpenRouterService = Depends(get_openrouter_service)
):
    try:
        result = await openrouter_service.cancel_request(request_id)
        return result
    except OpenRouterException as e:
        logger.error(f"Error cancelling request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
