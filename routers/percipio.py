from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.percipio_integration_service import PercipioIntegrationService, get_percipio_integration_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class UserInterests(BaseModel):
    interests: List[str] = Field(..., min_items=1)

# ... (keep existing endpoints)

@router.get("/percipio/analyze-sentiment/{content_id}")
async def analyze_content_sentiment(
    content_id: str,
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioIntegrationService = Depends(get_percipio_integration_service),
):
    try:
        sentiment_analysis = await percipio_service.analyze_content_sentiment(content_id)
        logger.info(f"Analyzed sentiment for content {content_id}")
        return sentiment_analysis
    except Exception as e:
        logger.error(f"Error analyzing content sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while analyzing content sentiment")

@router.get("/percipio/generate-summary/{content_id}")
async def generate_content_summary(
    content_id: str,
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioIntegrationService = Depends(get_percipio_integration_service),
):
    try:
        content_summary = await percipio_service.generate_content_summary(content_id)
        logger.info(f"Generated summary for content {content_id}")
        return content_summary
    except Exception as e:
        logger.error(f"Error generating content summary: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating content summary")

@router.get("/percipio/predict-performance/{course_id}")
async def predict_user_performance(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    percipio_service: PercipioIntegrationService = Depends(get_percipio_integration_service),
):
    try:
        performance_prediction = await percipio_service.predict_user_performance(user.id, course_id)
        logger.info(f"Predicted performance for user {user.id} in course {course_id}")
        return performance_prediction
    except Exception as e:
        logger.error(f"Error predicting user performance: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while predicting user performance")
