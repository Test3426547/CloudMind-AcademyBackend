from typing import List, Optional, Dict
from datetime import datetime, timedelta
from fastapi import HTTPException
from models.user import User
from models.time_entry import TimeEntry
from auth_config import get_supabase_client
import logging

logger = logging.getLogger(__name__)

class TimeTrackingService:
    def __init__(self):
        self.supabase = get_supabase_client()

    async def start_timer(self, user_id: str, task_id: str, description: str) -> Dict:
        try:
            new_entry = TimeEntry(
                user_id=user_id,
                task_id=task_id,
                description=description,
                start_time=datetime.utcnow()
            )
            result = self.supabase.table('time_entries').insert(new_entry.dict()).execute()
            return result.data[0]
        except Exception as e:
            logger.error(f"Error starting timer: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to start timer")

    async def stop_timer(self, user_id: str, entry_id: str) -> Dict:
        try:
            entry = self.supabase.table('time_entries').select('*').eq('id', entry_id).eq('user_id', user_id).execute()
            if not entry.data:
                raise HTTPException(status_code=404, detail="Time entry not found")
            
            end_time = datetime.utcnow()
            duration = (end_time - entry.data[0]['start_time']).total_seconds() / 3600  # Duration in hours
            
            result = self.supabase.table('time_entries').update({
                'end_time': end_time,
                'duration': duration
            }).eq('id', entry_id).execute()
            
            return result.data[0]
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error stopping timer: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to stop timer")

    async def get_time_entries(self, user_id: str, start_date: Optional[datetime], end_date: Optional[datetime], limit: int, offset: int) -> List[Dict]:
        try:
            query = self.supabase.table('time_entries').select('*').eq('user_id', user_id).order('start_time', desc=True)
            
            if start_date:
                query = query.gte('start_time', start_date)
            if end_date:
                query = query.lte('start_time', end_date)
            
            result = query.range(offset, offset + limit - 1).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error retrieving time entries: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to retrieve time entries")

    async def get_productivity_analytics(self, user_id: str, start_date: datetime, end_date: datetime) -> Dict:
        try:
            entries = self.supabase.table('time_entries').select('*').eq('user_id', user_id).gte('start_time', start_date).lte('start_time', end_date).execute()
            
            total_time = sum(entry['duration'] or 0 for entry in entries.data)
            task_breakdown = {}
            hourly_productivity = {i: 0 for i in range(24)}
            
            for entry in entries.data:
                task_breakdown[entry['task_id']] = task_breakdown.get(entry['task_id'], 0) + (entry['duration'] or 0)
                hour = entry['start_time'].hour
                hourly_productivity[hour] += entry['duration'] or 0
            
            most_productive_hour = max(hourly_productivity, key=hourly_productivity.get)
            
            # Calculate productivity score (example algorithm, can be adjusted)
            productivity_score = min(100, (total_time / ((end_date - start_date).days + 1)) * 10)
            
            return {
                "total_time": total_time,
                "productivity_score": productivity_score,
                "task_breakdown": task_breakdown,
                "most_productive_hour": f"{most_productive_hour:02d}:00"
            }
        except Exception as e:
            logger.error(f"Error calculating productivity analytics: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to calculate productivity analytics")

    async def delete_time_entry(self, user_id: str, entry_id: str) -> None:
        try:
            result = self.supabase.table('time_entries').delete().eq('id', entry_id).eq('user_id', user_id).execute()
            if not result.data:
                raise HTTPException(status_code=404, detail="Time entry not found")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting time entry: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to delete time entry")

time_tracking_service = TimeTrackingService()

def get_time_tracking_service() -> TimeTrackingService:
    return time_tracking_service
