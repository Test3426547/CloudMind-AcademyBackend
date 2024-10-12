from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.data_visualization_service import DataVisualizationService, get_data_visualization_service, ChartData
from typing import Dict, Any
from pydantic import BaseModel
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class DateRange(BaseModel):
    start_date: str
    end_date: str

@router.post("/productivity-chart", response_model=ChartData)
async def generate_productivity_chart(
    date_range: DateRange,
    user: User = Depends(oauth2_scheme),
    visualization_service: DataVisualizationService = Depends(get_data_visualization_service),
):
    try:
        chart_data = await visualization_service.generate_productivity_chart(user.id, date_range.start_date, date_range.end_date)
        logger.info(f"Generated productivity chart for user {user.id}")
        return chart_data
    except ValueError as e:
        logger.warning(f"Invalid input for generate_productivity_chart: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating productivity chart: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate productivity chart")

@router.post("/user-engagement-chart", response_model=ChartData)
async def generate_user_engagement_chart(
    date_range: DateRange,
    user: User = Depends(oauth2_scheme),
    visualization_service: DataVisualizationService = Depends(get_data_visualization_service),
):
    try:
        chart_data = await visualization_service.generate_user_engagement_chart(user.id, date_range.start_date, date_range.end_date)
        logger.info(f"Generated user engagement chart for user {user.id}")
        return chart_data
    except ValueError as e:
        logger.warning(f"Invalid input for generate_user_engagement_chart: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating user engagement chart: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate user engagement chart")

@router.post("/user-behavior-analysis", response_model=ChartData)
async def analyze_user_behavior(
    date_range: DateRange,
    user: User = Depends(oauth2_scheme),
    visualization_service: DataVisualizationService = Depends(get_data_visualization_service),
):
    try:
        chart_data = await visualization_service.analyze_user_behavior(date_range.start_date, date_range.end_date)
        logger.info(f"Analyzed user behavior for date range: {date_range.start_date} to {date_range.end_date}")
        return chart_data
    except ValueError as e:
        logger.warning(f"Invalid input for analyze_user_behavior: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing user behavior: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze user behavior")

@router.get("/predict-learning-outcome", response_model=Dict[str, Any])
async def predict_learning_outcome(
    course_id: str = Query(..., min_length=1),
    user: User = Depends(oauth2_scheme),
    visualization_service: DataVisualizationService = Depends(get_data_visualization_service),
):
    try:
        prediction = await visualization_service.predict_learning_outcomes(user.id, course_id)
        logger.info(f"Predicted learning outcome for user {user.id} in course {course_id}")
        return prediction
    except ValueError as e:
        logger.warning(f"Invalid input for predict_learning_outcome: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting learning outcome: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to predict learning outcome")
