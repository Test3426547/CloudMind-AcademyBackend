from typing import Dict, Any, Optional
from fastapi import HTTPException
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from auth_config import get_supabase_client
import logging
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

class ChartData(BaseModel):
    chart_url: str
    metadata: Dict[str, Any]

class DataVisualizationService:
    def __init__(self):
        self.supabase = get_supabase_client()

    @staticmethod
    def _validate_date_range(start_date: str, end_date: str) -> tuple:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            if end <= start:
                raise ValueError("End date must be after start date")
            if end - start > timedelta(days=365):
                raise ValueError("Date range cannot exceed one year")
            return start, end
        except ValueError as e:
            logger.error(f"Date validation error: {str(e)}")
            raise ValueError(f"Invalid date format or range: {str(e)}")

    async def generate_productivity_chart(self, user_id: str, start_date: str, end_date: str) -> ChartData:
        try:
            start, end = self._validate_date_range(start_date, end_date)
            
            # Fetch productivity data from Supabase
            result = self.supabase.table('time_entries').select('date, duration').eq('user_id', user_id).gte('date', start).lte('date', end).execute()
            
            if not result.data:
                raise ValueError("No productivity data found for the given date range")

            # Process data for visualization
            dates = [entry['date'] for entry in result.data]
            durations = [entry['duration'] for entry in result.data]

            # Create chart
            plt.figure(figsize=(12, 6))
            sns.lineplot(x=dates, y=durations)
            plt.title('Productivity Over Time')
            plt.xlabel('Date')
            plt.ylabel('Duration (hours)')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save chart to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            chart_url = base64.b64encode(buffer.getvalue()).decode()

            return ChartData(
                chart_url=f"data:image/png;base64,{chart_url}",
                metadata={
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_entries": len(result.data),
                    "average_duration": sum(durations) / len(durations) if durations else 0
                }
            )
        except ValueError as e:
            logger.warning(f"Invalid input for generate_productivity_chart: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating productivity chart: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate productivity chart")

    async def generate_learning_progress_chart(self, user_id: str, course_id: str) -> ChartData:
        try:
            # Fetch learning progress data from Supabase
            result = self.supabase.table('course_progress').select('date, progress_percentage').eq('user_id', user_id).eq('course_id', course_id).execute()
            
            if not result.data:
                raise ValueError("No learning progress data found for the given course")

            # Process data for visualization
            dates = [entry['date'] for entry in result.data]
            progress = [entry['progress_percentage'] for entry in result.data]

            # Create chart
            plt.figure(figsize=(12, 6))
            sns.lineplot(x=dates, y=progress)
            plt.title('Learning Progress Over Time')
            plt.xlabel('Date')
            plt.ylabel('Progress (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save chart to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            chart_url = base64.b64encode(buffer.getvalue()).decode()

            return ChartData(
                chart_url=f"data:image/png;base64,{chart_url}",
                metadata={
                    "course_id": course_id,
                    "total_entries": len(result.data),
                    "current_progress": progress[-1] if progress else 0
                }
            )
        except ValueError as e:
            logger.warning(f"Invalid input for generate_learning_progress_chart: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating learning progress chart: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate learning progress chart")

    async def generate_user_engagement_chart(self, user_id: str, start_date: str, end_date: str) -> ChartData:
        try:
            start, end = self._validate_date_range(start_date, end_date)
            
            # Fetch user engagement data from Supabase
            result = self.supabase.table('user_activities').select('date, activity_type, count').eq('user_id', user_id).gte('date', start).lte('date', end).execute()
            
            if not result.data:
                raise ValueError("No user engagement data found for the given date range")

            # Process data for visualization
            activity_types = set(entry['activity_type'] for entry in result.data)
            data = {activity: [] for activity in activity_types}
            dates = sorted(set(entry['date'] for entry in result.data))

            for date in dates:
                for activity in activity_types:
                    count = next((entry['count'] for entry in result.data if entry['date'] == date and entry['activity_type'] == activity), 0)
                    data[activity].append(count)

            # Create chart
            plt.figure(figsize=(12, 6))
            for activity, counts in data.items():
                plt.plot(dates, counts, label=activity)

            plt.title('User Engagement Over Time')
            plt.xlabel('Date')
            plt.ylabel('Activity Count')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save chart to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            chart_url = base64.b64encode(buffer.getvalue()).decode()

            return ChartData(
                chart_url=f"data:image/png;base64,{chart_url}",
                metadata={
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_entries": len(result.data),
                    "activity_types": list(activity_types)
                }
            )
        except ValueError as e:
            logger.warning(f"Invalid input for generate_user_engagement_chart: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating user engagement chart: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate user engagement chart")

    async def generate_custom_chart(self, chart_type: str, data: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> ChartData:
        try:
            if chart_type not in ['bar', 'line', 'scatter', 'pie']:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            if not data or not isinstance(data, dict):
                raise ValueError("Invalid data format")

            plt.figure(figsize=(12, 6))

            if chart_type == 'bar':
                plt.bar(data.keys(), data.values())
            elif chart_type == 'line':
                plt.plot(list(data.keys()), list(data.values()))
            elif chart_type == 'scatter':
                plt.scatter(list(data.keys()), list(data.values()))
            elif chart_type == 'pie':
                plt.pie(list(data.values()), labels=list(data.keys()), autopct='%1.1f%%')

            if options:
                if 'title' in options:
                    plt.title(options['title'])
                if 'xlabel' in options:
                    plt.xlabel(options['xlabel'])
                if 'ylabel' in options:
                    plt.ylabel(options['ylabel'])

            plt.tight_layout()

            # Save chart to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            chart_url = base64.b64encode(buffer.getvalue()).decode()

            return ChartData(
                chart_url=f"data:image/png;base64,{chart_url}",
                metadata={
                    "chart_type": chart_type,
                    "data_points": len(data),
                    "options": options or {}
                }
            )
        except ValueError as e:
            logger.warning(f"Invalid input for generate_custom_chart: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating custom chart: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate custom chart")

data_visualization_service = DataVisualizationService()

def get_data_visualization_service() -> DataVisualizationService:
    return data_visualization_service
