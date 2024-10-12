from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.ai_model_training_service import AIModelTrainingService, get_ai_model_training_service, HuggingFaceTrainingRequest
from typing import Dict, Any
import logging

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

logger = logging.getLogger(__name__)

@router.post("/ai-model/train-huggingface", response_model=Dict[str, Any])
async def train_with_huggingface(
    request: HuggingFaceTrainingRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(oauth2_scheme),
    training_service: AIModelTrainingService = Depends(get_ai_model_training_service),
):
    try:
        # Start the training process in the background
        background_tasks.add_task(training_service.train_with_huggingface, request)
        logger.info(f"Hugging Face model training started for user {user.id}")
        return {"message": "Training process started", "status": "pending"}
    except HTTPException as e:
        logger.error(f"HTTP error in train_with_huggingface: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in train_with_huggingface: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during model training")

@router.get("/ai-model/training-progress/{training_id}", response_model=Dict[str, Any])
async def get_training_progress(
    training_id: str,
    user: User = Depends(oauth2_scheme),
    training_service: AIModelTrainingService = Depends(get_ai_model_training_service),
):
    try:
        progress = await training_service.get_training_progress(training_id)
        logger.info(f"Training progress retrieved for training_id {training_id} by user {user.id}")
        return progress
    except HTTPException as e:
        logger.error(f"HTTP error in get_training_progress: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_training_progress: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving training progress")
