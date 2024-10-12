from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.translation_service import TranslationService, get_translation_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    source_lang: str = Field(..., min_length=2, max_length=2)
    target_lang: str = Field(..., min_length=2, max_length=2)

class DetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class SummarizationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    target_length: int = Field(100, ge=50, le=500)

class KeyPhraseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    num_phrases: int = Field(5, ge=1, le=20)

class SentimentAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

@router.post("/translate")
async def translate_text(
    request: TranslationRequest,
    user: User = Depends(oauth2_scheme),
    translation_service: TranslationService = Depends(get_translation_service),
):
    try:
        translation = await translation_service.translate(request.text, request.source_lang, request.target_lang)
        logger.info(f"Translation completed for user {user.id}")
        return {"translation": translation}
    except HTTPException as e:
        logger.warning(f"HTTP error in translate_text: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in translate_text: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during translation")

@router.post("/detect-language")
async def detect_language(
    request: DetectionRequest,
    user: User = Depends(oauth2_scheme),
    translation_service: TranslationService = Depends(get_translation_service),
):
    try:
        detected_lang = await translation_service.detect_language(request.text)
        logger.info(f"Language detection completed for user {user.id}")
        return {"detected_language": detected_lang}
    except HTTPException as e:
        logger.warning(f"HTTP error in detect_language: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in detect_language: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during language detection")

@router.post("/summarize")
async def summarize_text(
    request: SummarizationRequest,
    user: User = Depends(oauth2_scheme),
    translation_service: TranslationService = Depends(get_translation_service),
):
    try:
        summary = await translation_service.summarize_text(request.text, request.target_length)
        logger.info(f"Text summarization completed for user {user.id}")
        return {"summary": summary}
    except HTTPException as e:
        logger.warning(f"HTTP error in summarize_text: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in summarize_text: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during text summarization")

@router.post("/analyze-complexity")
async def analyze_text_complexity(
    request: DetectionRequest,
    user: User = Depends(oauth2_scheme),
    translation_service: TranslationService = Depends(get_translation_service),
):
    try:
        complexity_analysis = await translation_service.analyze_text_complexity(request.text)
        logger.info(f"Text complexity analysis completed for user {user.id}")
        return complexity_analysis
    except HTTPException as e:
        logger.warning(f"HTTP error in analyze_text_complexity: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in analyze_text_complexity: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during text complexity analysis")

@router.post("/identify-key-phrases")
async def identify_key_phrases(
    request: KeyPhraseRequest,
    user: User = Depends(oauth2_scheme),
    translation_service: TranslationService = Depends(get_translation_service),
):
    try:
        key_phrases = await translation_service.identify_key_phrases(request.text, request.num_phrases)
        logger.info(f"Key phrase identification completed for user {user.id}")
        return {"key_phrases": key_phrases}
    except HTTPException as e:
        logger.warning(f"HTTP error in identify_key_phrases: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in identify_key_phrases: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during key phrase identification")

@router.post("/sentiment-analysis")
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    user: User = Depends(oauth2_scheme),
    translation_service: TranslationService = Depends(get_translation_service),
):
    try:
        sentiment_analysis = await translation_service.sentiment_analysis(request.text)
        logger.info(f"Sentiment analysis completed for user {user.id}")
        return sentiment_analysis
    except HTTPException as e:
        logger.warning(f"HTTP error in analyze_sentiment: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in analyze_sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during sentiment analysis")
