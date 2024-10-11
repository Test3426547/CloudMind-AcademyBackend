from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.data_visualization_service import DataVisualizationService, get_data_visualization_service
from typing import List, Optional
from pydantic import BaseModel, Field
import logging
from fastapi_limiter.depends import RateLimiter
from cachetools import TTLCache, cached

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

# Initialize cache
cache = TTLCache(maxsize=100, ttl=300)  # Cache for 5 minutes

class ChartRequest(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate")
    data: dict = Field(..., description="Data for the chart")
    options: Optional[dict] = Field(None, description="Additional options for the chart")

class ChartResponse(BaseModel):
    chart_url: str
    metadata: dict

@router.post("/visualize/productivity", response_model=ChartResponse)
@cached(cache)
async def visualize_productivity(
    start_date: str = Query(..., description="Start date for the data range"),
    end_date: str = Query(..., description="End date for the data range"),
    user: User = Depends(oauth2_scheme),
    visualization_service: DataVisualizationService = Depends(get_data_visualization_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        chart_data = await visualization_service.generate_productivity_chart(user.id, start_date, end_date)
        logger.info(f"Generated productivity chart for user {user.id}")
        return chart_data
    except ValueError as e:
        logger.warning(f"Invalid input for visualize_productivity: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating productivity chart: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the productivity chart")

@router.post("/visualize/learning-progress", response_model=ChartResponse)
@cached(cache)
async def visualize_learning_progress(
    course_id: str = Query(..., description="ID of the course to visualize progress for"),
    user: User = Depends(oauth2_scheme),
    visualization_service: DataVisualizationService = Depends(get_data_visualization_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        chart_data = await visualization_service.generate_learning_progress_chart(user.id, course_id)
        logger.info(f"Generated learning progress chart for user {user.id} and course {course_id}")
        return chart_data
    except ValueError as e:
        logger.warning(f"Invalid input for visualize_learning_progress: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating learning progress chart: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the learning progress chart")

@router.post("/visualize/user-engagement", response_model=ChartResponse)
@cached(cache)
async def visualize_user_engagement(
    start_date: str = Query(..., description="Start date for the data range"),
    end_date: str = Query(..., description="End date for the data range"),
    user: User = Depends(oauth2_scheme),
    visualization_service: DataVisualizationService = Depends(get_data_visualization_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        chart_data = await visualization_service.generate_user_engagement_chart(user.id, start_date, end_date)
        logger.info(f"Generated user engagement chart for user {user.id}")
        return chart_data
    except ValueError as e:
        logger.warning(f"Invalid input for visualize_user_engagement: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating user engagement chart: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the user engagement chart")

@router.post("/visualize/custom", response_model=ChartResponse)
async def create_custom_chart(
    chart_request: ChartRequest,
    user: User = Depends(oauth2_scheme),
    visualization_service: DataVisualizationService = Depends(get_data_visualization_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=3, seconds=60))
):
    try:
        chart_data = await visualization_service.generate_custom_chart(chart_request.chart_type, chart_request.data, chart_request.options)
        logger.info(f"Generated custom chart for user {user.id}")
        return chart_data
    except ValueError as e:
        logger.warning(f"Invalid input for create_custom_chart: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating custom chart: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the custom chart")
