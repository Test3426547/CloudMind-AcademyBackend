from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.ai_model_training_service import AIModelTrainingService, get_ai_model_training_service
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class HuggingFaceTrainingRequest(BaseModel):
    model_name: str
    dataset_name: str
    num_labels: int

@router.post("/ai-model/train-huggingface")
async def train_with_huggingface(
    request: HuggingFaceTrainingRequest,
    user: User = Depends(oauth2_scheme),
    training_service: AIModelTrainingService = Depends(get_ai_model_training_service)
):
    try:
        result = await training_service.train_with_huggingface(
            request.model_name,
            request.dataset_name,
            request.num_labels
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ai-model/run-ml-flows")
async def run_ml_flows(
    user: User = Depends(oauth2_scheme),
    training_service: AIModelTrainingService = Depends(get_ai_model_training_service)
):
    try:
        results = await training_service.run_ml_flows()
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
