from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.emotion_analysis_service import EmotionAnalysisService, get_emotion_analysis_service
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class EmotionAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    speech_data: Optional[str] = Field(None, max_length=1000)

@router.post("/emotion-analysis", response_model=Dict[str, Any])
async def analyze_emotion(
    request: EmotionAnalysisRequest,
    user: User = Depends(oauth2_scheme),
    emotion_service: EmotionAnalysisService = Depends(get_emotion_analysis_service),
):
    try:
        result = await emotion_service.analyze_emotion(user.id, request.text, request.speech_data)
        logger.info(f"Emotion analysis completed for user {user.id}")
        return result
    except HTTPException as e:
        logger.error(f"HTTP error in analyze_emotion: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in analyze_emotion: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during emotion analysis")

@router.get("/emotion-trend", response_model=Dict[str, Any])
async def get_emotion_trend(
    user: User = Depends(oauth2_scheme),
    emotion_service: EmotionAnalysisService = Depends(get_emotion_analysis_service),
):
    try:
        trend = await emotion_service._analyze_emotion_trend(user.id)
        logger.info(f"Emotion trend analysis completed for user {user.id}")
        return trend
    except HTTPException as e:
        logger.error(f"HTTP error in get_emotion_trend: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_emotion_trend: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving emotion trend")

@router.post("/contextual-emotion", response_model=Dict[str, str])
async def get_contextual_emotion(
    request: EmotionAnalysisRequest,
    user: User = Depends(oauth2_scheme),
    emotion_service: EmotionAnalysisService = Depends(get_emotion_analysis_service),
):
    try:
        emotion_result = await emotion_service.analyze_emotion(user.id, request.text, request.speech_data)
        context = await emotion_service._generate_contextual_understanding(user.id, request.text, emotion_result["emotion"])
        logger.info(f"Contextual emotion analysis completed for user {user.id}")
        return {"contextual_understanding": context}
    except HTTPException as e:
        logger.error(f"HTTP error in get_contextual_emotion: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_contextual_emotion: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during contextual emotion analysis")
