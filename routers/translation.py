from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.translation import TranslationService, get_translation_service
from typing import Dict
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    target_language: str = Field(..., min_length=2, max_length=5)
    source_language: str = Field('auto', min_length=2, max_length=5)

class LanguageDetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

@router.post("/translate")
async def translate_text(
    request: TranslationRequest,
    user: User = Depends(oauth2_scheme),
    translation_service: TranslationService = Depends(get_translation_service),
):
    try:
        result = await translation_service.translate_text(request.text, request.target_language, request.source_language)
        logger.info(f"Translation completed for user {user.id}")
        return result
    except ValueError as e:
        logger.warning(f"Invalid input for translate_text: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in translate_text: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during translation")

@router.post("/detect-language")
async def detect_language(
    request: LanguageDetectionRequest,
    user: User = Depends(oauth2_scheme),
    translation_service: TranslationService = Depends(get_translation_service),
):
    try:
        result = await translation_service.detect_language_async(request.text)
        logger.info(f"Language detection completed for user {user.id}")
        return result
    except ValueError as e:
        logger.warning(f"Invalid input for detect_language: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in detect_language: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during language detection")
