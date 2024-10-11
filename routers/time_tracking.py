from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.time_tracking_service import TimeTrackingService, get_time_tracking_service
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging
from fastapi_limiter.depends import RateLimiter
from cachetools import TTLCache, cached

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

# Initialize cache
cache = TTLCache(maxsize=100, ttl=300)  # Cache for 5 minutes

class TimeEntry(BaseModel):
    task_id: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)

class TimeEntryResponse(BaseModel):
    id: str
    user_id: str
    task_id: str
    description: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None

class ProductivityAnalytics(BaseModel):
    total_time: float
    productivity_score: float
    task_breakdown: dict
    most_productive_hour: str

@router.post("/time-tracking/start", response_model=TimeEntryResponse)
async def start_timer(
    time_entry: TimeEntry,
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        result = await time_tracking_service.start_timer(user.id, time_entry.task_id, time_entry.description)
        logger.info(f"Timer started for user {user.id}, task {time_entry.task_id}")
        return result
    except ValueError as e:
        logger.warning(f"Invalid input for start_timer: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting timer: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while starting the timer")

@router.post("/time-tracking/{entry_id}/stop", response_model=TimeEntryResponse)
async def stop_timer(
    entry_id: str,
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        result = await time_tracking_service.stop_timer(user.id, entry_id)
        logger.info(f"Timer stopped for user {user.id}, entry {entry_id}")
        return result
    except ValueError as e:
        logger.warning(f"Invalid input for stop_timer: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error stopping timer: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while stopping the timer")

@router.get("/time-tracking/entries", response_model=List[TimeEntryResponse])
@cached(cache)
async def get_time_entries(
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=20, seconds=60))
):
    try:
        entries = await time_tracking_service.get_time_entries(user.id, start_date, end_date, limit, offset)
        logger.info(f"Retrieved {len(entries)} time entries for user {user.id}")
        return entries
    except ValueError as e:
        logger.warning(f"Invalid input for get_time_entries: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving time entries: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving time entries")

@router.get("/time-tracking/productivity", response_model=ProductivityAnalytics)
@cached(cache)
async def get_productivity_analytics(
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service),
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        if end_date <= start_date:
            raise ValueError("End date must be after start date")
        if end_date - start_date > timedelta(days=30):
            raise ValueError("Date range cannot exceed 30 days")

        analytics = await time_tracking_service.get_productivity_analytics(user.id, start_date, end_date)
        logger.info(f"Retrieved productivity analytics for user {user.id}")
        return analytics
    except ValueError as e:
        logger.warning(f"Invalid input for get_productivity_analytics: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving productivity analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving productivity analytics")

@router.delete("/time-tracking/{entry_id}")
async def delete_time_entry(
    entry_id: str,
    user: User = Depends(oauth2_scheme),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        await time_tracking_service.delete_time_entry(user.id, entry_id)
        logger.info(f"Deleted time entry {entry_id} for user {user.id}")
        return {"message": "Time entry deleted successfully"}
    except ValueError as e:
        logger.warning(f"Invalid input for delete_time_entry: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting time entry: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while deleting the time entry")
