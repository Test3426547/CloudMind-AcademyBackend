from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.voice_recognition_service import VoiceRecognitionService, get_voice_recognition_service
from typing import List, Dict, Any
from pydantic import BaseModel
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class VoiceAnalysisResponse(BaseModel):
    transcription: str
    analysis: Dict[str, Any]
    speaker_id: str

class VoiceVerificationResponse(BaseModel):
    user_id: str
    is_verified: bool
    similarity_score: float

@router.post("/voice/analyze", response_model=VoiceAnalysisResponse)
async def analyze_voice(
    audio: UploadFile = File(...),
    user: User = Depends(oauth2_scheme),
    voice_service: VoiceRecognitionService = Depends(get_voice_recognition_service),
):
    try:
        audio_data = await audio.read()
        result = await voice_service.analyze_voice(audio_data)
        logger.info(f"Voice analysis completed for user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in analyze_voice: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in analyze_voice: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during voice analysis")

@router.post("/voice/train")
async def train_voice_model(
    audio_samples: List[UploadFile] = File(...),
    user: User = Depends(oauth2_scheme),
    voice_service: VoiceRecognitionService = Depends(get_voice_recognition_service),
):
    try:
        audio_data_list = [await audio.read() for audio in audio_samples]
        result = await voice_service.train_voice_model(user.id, audio_data_list)
        logger.info(f"Voice model trained for user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in train_voice_model: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in train_voice_model: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during voice model training")

@router.post("/voice/verify", response_model=VoiceVerificationResponse)
async def verify_voice(
    audio: UploadFile = File(...),
    user: User = Depends(oauth2_scheme),
    voice_service: VoiceRecognitionService = Depends(get_voice_recognition_service),
):
    try:
        audio_data = await audio.read()
        result = await voice_service.verify_voice(user.id, audio_data)
        logger.info(f"Voice verification completed for user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in verify_voice: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in verify_voice: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during voice verification")
