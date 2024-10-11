from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.ai_model_training_service import AIModelTrainingService, get_ai_model_training_service, HuggingFaceTrainingRequest
from typing import Dict, Any
import logging
from fastapi_limiter.depends import RateLimiter

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

logger = logging.getLogger(__name__)

@router.post("/ai-model/train-huggingface", response_model=Dict[str, Any])
async def train_with_huggingface(
    request: HuggingFaceTrainingRequest,
    user: User = Depends(oauth2_scheme),
    training_service: AIModelTrainingService = Depends(get_ai_model_training_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=1, seconds=3600))  # Limit to 1 request per hour
):
    try:
        result = await training_service.train_with_huggingface(request)
        logger.info(f"Hugging Face model training completed for user {user.id}")
        return result
    except HTTPException as e:
        logger.error(f"HTTP error in train_with_huggingface: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in train_with_huggingface: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during model training")

@router.get("/ai-model/run-ml-flows", response_model=Dict[str, str])
async def run_ml_flows(
    user: User = Depends(oauth2_scheme),
    training_service: AIModelTrainingService = Depends(get_ai_model_training_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=3600))  # Limit to 5 requests per hour
):
    try:
        results = await training_service.run_ml_flows()
        logger.info(f"ML flows executed successfully for user {user.id}")
        return {"results": results}
    except HTTPException as e:
        logger.error(f"HTTP error in run_ml_flows: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in run_ml_flows: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while running ML flows")
