from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.ai_model_training_service import AIModelTrainingService, get_ai_model_training_service
from typing import List, Optional
from pydantic import BaseModel
import os

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ... (keep all existing code)

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
