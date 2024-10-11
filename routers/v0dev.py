from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.v0dev_service import V0DevService, get_v0dev_service
from typing import Dict
from pydantic import BaseModel, Field, validator
import logging
from fastapi_limiter.depends import RateLimiter
from cachetools import TTLCache, cached

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

# Initialize cache
cache = TTLCache(maxsize=100, ttl=3600)  # Cache for 1 hour

class UIGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=1000)

    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty or just whitespace")
        return v

@router.post("/v0dev/generate-ui")
@cached(cache)
async def generate_ui_component(
    request: UIGenerationRequest,
    user: User = Depends(oauth2_scheme),
    v0dev_service: V0DevService = Depends(get_v0dev_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        logger.info(f"Generating UI component for user {user.id}")
        result = await v0dev_service.generate_ui_component(request.prompt)
        logger.info(f"Successfully generated UI component for user {user.id}")
        return result
    except ValueError as e:
        logger.warning(f"Invalid input for generate_ui_component: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        logger.error(f"HTTP error in generate_ui_component: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in generate_ui_component: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the UI component")

@router.get("/v0dev/health")
async def health_check(
    v0dev_service: V0DevService = Depends(get_v0dev_service)
):
    try:
        health_status = await v0dev_service.health_check()
        if health_status["status"] == "healthy":
            return health_status
        else:
            raise HTTPException(status_code=503, detail=health_status["message"])
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service is unhealthy: {str(e)}")
