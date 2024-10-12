from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.sentiment_analysis_service import SentimentAnalysisService, get_sentiment_analysis_service
from typing import Dict, Any, List
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=10)

@router.post("/analyze-sentiment")
async def analyze_sentiment(
    text_input: TextInput,
    user: User = Depends(oauth2_scheme),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_analysis_service),
):
    try:
        result = await sentiment_service.analyze_sentiment(text_input.text)
        logger.info(f"Sentiment analysis completed for user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in analyze_sentiment: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in analyze_sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during sentiment analysis")

@router.post("/batch-analyze-sentiment")
async def batch_analyze_sentiment(
    batch_input: BatchTextInput,
    user: User = Depends(oauth2_scheme),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_analysis_service),
):
    try:
        results = await sentiment_service.batch_analyze_sentiment(batch_input.texts)
        logger.info(f"Batch sentiment analysis completed for user {user.id}")
        return results
    except HTTPException as e:
        logger.warning(f"HTTP error in batch_analyze_sentiment: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in batch_analyze_sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during batch sentiment analysis")
