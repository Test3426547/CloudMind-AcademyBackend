from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.time_tracking_service import TimeTrackingService, get_time_tracking_service
from typing import List, Dict, Any
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TimeEntryStart(BaseModel):
    task_id: str
    description: str

class TimeEntryResponse(BaseModel):
    id: str
    user_id: str
    task_id: str
    description: str
    start_time: datetime
    end_time: datetime = None
    duration: float = None

class ProductivityAnalytics(BaseModel):
    total_time: float
    task_breakdown: Dict[str, float]
    num_entries: int
    avg_entry_duration: float
    productivity_score: float
    productivity_trend: str
    task_efficiency: Dict[str, float]
    insights: str

@router.post("/time-tracking/start", response_model=TimeEntryResponse)
async def start_timer(
    entry: TimeEntryStart,
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service)
):
    result = await time_tracking_service.start_timer(user.id, entry.task_id, entry.description)
    return TimeEntryResponse(**result)

@router.post("/time-tracking/{entry_id}/stop", response_model=TimeEntryResponse)
async def stop_timer(
    entry_id: str,
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service)
):
    try:
        result = await time_tracking_service.stop_timer(entry_id)
        return TimeEntryResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/time-tracking/entries", response_model=List[TimeEntryResponse])
async def get_time_entries(
    start_date: datetime,
    end_date: datetime,
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service)
):
    entries = await time_tracking_service.get_user_time_entries(user.id, start_date, end_date)
    return [TimeEntryResponse(**entry) for entry in entries]

@router.get("/time-tracking/analytics", response_model=ProductivityAnalytics)
async def get_productivity_analytics(
    start_date: datetime,
    end_date: datetime,
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service)
):
    analytics = await time_tracking_service.get_productivity_analytics(user.id, start_date, end_date)
    return ProductivityAnalytics(**analytics)

@router.get("/time-tracking/dashboard")
async def get_productivity_dashboard(
    start_date: datetime,
    end_date: datetime,
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service)
):
    analytics = await time_tracking_service.get_productivity_analytics(user.id, start_date, end_date)
    entries = await time_tracking_service.get_user_time_entries(user.id, start_date, end_date)
    
    daily_productivity = {}
    for entry in entries:
        date = entry["start_time"].date()
        if date not in daily_productivity:
            daily_productivity[date] = 0
        daily_productivity[date] += entry["duration"] or 0
    
    return {
        "analytics": ProductivityAnalytics(**analytics),
        "daily_productivity": daily_productivity,
        "recent_entries": [TimeEntryResponse(**entry) for entry in entries[-5:]]  # Last 5 entries
    }
