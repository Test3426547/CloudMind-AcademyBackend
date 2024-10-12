from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.time_tracking_service import TimeTrackingService, get_time_tracking_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class TimerStart(BaseModel):
    task: str = Field(..., min_length=1, max_length=200)

class ProductivityAnalysis(BaseModel):
    total_time: float
    avg_duration: float
    task_stats: Dict[str, Dict[str, float]]
    productivity_score: float
    insights: str

class TaskDurationPrediction(BaseModel):
    task: str
    predicted_duration: float
    confidence: float

class TimeTrackingAnomaly(BaseModel):
    entry_id: str
    task: str
    duration: float
    start_time: Any

@router.post("/time-tracking/start")
async def start_timer(
    timer_start: TimerStart,
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service),
):
    try:
        result = await time_tracking_service.start_timer(user.id, timer_start.task)
        logger.info(f"Timer started for user {user.id}")
        return result
    except Exception as e:
        logger.error(f"Error starting timer: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while starting the timer")

@router.post("/time-tracking/stop/{entry_id}")
async def stop_timer(
    entry_id: str,
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service),
):
    try:
        result = await time_tracking_service.stop_timer(entry_id)
        logger.info(f"Timer stopped for entry {entry_id}")
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error stopping timer: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while stopping the timer")

@router.get("/time-tracking/entries")
async def get_user_time_entries(
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service),
):
    try:
        entries = await time_tracking_service.get_user_time_entries(user.id)
        logger.info(f"Retrieved time entries for user {user.id}")
        return entries
    except Exception as e:
        logger.error(f"Error retrieving time entries: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving time entries")

@router.get("/time-tracking/analyze-productivity", response_model=ProductivityAnalysis)
async def analyze_productivity(
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service),
):
    try:
        analysis = await time_tracking_service.analyze_productivity(user.id)
        logger.info(f"Analyzed productivity for user {user.id}")
        return analysis
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error analyzing productivity: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while analyzing productivity")

@router.get("/time-tracking/predict-duration/{task}", response_model=TaskDurationPrediction)
async def predict_task_duration(
    task: str,
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service),
):
    try:
        prediction = await time_tracking_service.predict_task_duration(user.id, task)
        logger.info(f"Predicted task duration for user {user.id} and task {task}")
        return prediction
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error predicting task duration: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while predicting task duration")

@router.get("/time-tracking/detect-anomalies", response_model=List[TimeTrackingAnomaly])
async def detect_time_tracking_anomalies(
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service),
):
    try:
        anomalies = await time_tracking_service.detect_time_tracking_anomalies(user.id)
        logger.info(f"Detected time tracking anomalies for user {user.id}")
        return anomalies
    except Exception as e:
        logger.error(f"Error detecting time tracking anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while detecting time tracking anomalies")
