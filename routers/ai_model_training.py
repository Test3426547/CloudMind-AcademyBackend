from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.ai_model_training_service import AIModelTrainingService, get_ai_model_training_service
from typing import List, Optional
from pydantic import BaseModel
import os

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TrainingRequest(BaseModel):
    dataset_name: Optional[str] = None
    num_labels: int
    model_name: str = "distilbert-base-uncased"
    epochs: int = 3

class TensorFlowTrainingRequest(BaseModel):
    dataset_name: Optional[str] = None
    num_labels: int
    epochs: int = 3

class PredictionRequest(BaseModel):
    text: str

class AdvancedTrainingRequest(BaseModel):
    dataset_name: Optional[str] = None
    num_labels: int
    epochs: int = 3

@router.post("/ai-model/train")
async def train_model(
    request: TrainingRequest,
    user: User = Depends(oauth2_scheme),
    training_service: AIModelTrainingService = Depends(get_ai_model_training_service)
):
    try:
        result = await training_service.train_model(
            request.dataset_name,
            request.num_labels,
            request.model_name,
            request.epochs
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai-model/train-tensorflow")
async def train_tensorflow_model(
    request: TensorFlowTrainingRequest,
    user: User = Depends(oauth2_scheme),
    training_service: AIModelTrainingService = Depends(get_ai_model_training_service)
):
    try:
        result = await training_service.train_tensorflow_model(
            request.dataset_name,
            request.num_labels,
            request.epochs
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai-model/train-local")
async def train_model_local(
    num_labels: int,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    file: UploadFile = File(...),
    user: User = Depends(oauth2_scheme),
    training_service: AIModelTrainingService = Depends(get_ai_model_training_service)
):
    try:
        # Save the uploaded file
        file_path = f"./uploads/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        result = await training_service.train_model(
            num_labels=num_labels,
            model_name=model_name,
            epochs=epochs,
            local_dataset_path=file_path
        )

        # Clean up the uploaded file
        os.remove(file_path)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai-model/predict")
async def predict(
    request: PredictionRequest,
    user: User = Depends(oauth2_scheme),
    training_service: AIModelTrainingService = Depends(get_ai_model_training_service)
):
    try:
        prediction = await training_service.predict(request.text)
        return {"prediction": prediction}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai-model/train-advanced")
async def train_advanced_model(
    request: AdvancedTrainingRequest,
    user: User = Depends(oauth2_scheme),
    training_service: AIModelTrainingService = Depends(get_ai_model_training_service)
):
    try:
        result = await training_service.train_advanced_model(
            request.dataset_name,
            request.num_labels,
            request.epochs
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai-model/predict-advanced")
async def predict_advanced(
    request: PredictionRequest,
    user: User = Depends(oauth2_scheme),
    training_service: AIModelTrainingService = Depends(get_ai_model_training_service)
):
    try:
        prediction = await training_service.predict_advanced(request.text)
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
