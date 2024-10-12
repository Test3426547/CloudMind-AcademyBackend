from typing import Dict, Any, Optional, List
from fastapi import HTTPException
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from auth_config import get_supabase_client
import logging
from pydantic import BaseModel, Field, validator
import numpy as np
from collections import defaultdict

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
            dates = [datetime.strptime(entry['date'], "%Y-%m-%d") for entry in result.data]
            durations = [entry['duration'] for entry in result.data]

            # Perform trend analysis using moving average
            window_size = 7
            moving_average = self._calculate_moving_average(durations, window_size)

            # Create chart
            plt.figure(figsize=(12, 6))
            sns.lineplot(x=dates, y=durations, label='Actual')
            sns.lineplot(x=dates[window_size-1:], y=moving_average, label=f'{window_size}-day Moving Average')
            plt.title('Productivity Over Time with Trend Analysis')
            plt.xlabel('Date')
            plt.ylabel('Duration (hours)')
            plt.xticks(rotation=45)
            plt.legend()
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
                    "average_duration": sum(durations) / len(durations) if durations else 0,
                    "trend": "Increasing" if moving_average[-1] > moving_average[0] else "Decreasing"
                }
            )
        except ValueError as e:
            logger.warning(f"Invalid input for generate_productivity_chart: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating productivity chart: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate productivity chart")

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

            # Perform anomaly detection
            anomalies = self._detect_anomalies(data)

            # Create chart
            plt.figure(figsize=(12, 6))
            for activity, counts in data.items():
                plt.plot(dates, counts, label=activity)
                if activity in anomalies:
                    plt.scatter([dates[i] for i in anomalies[activity]], [counts[i] for i in anomalies[activity]], color='red', s=50)

            plt.title('User Engagement Over Time with Anomaly Detection')
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
                    "activity_types": list(activity_types),
                    "anomalies_detected": {k: len(v) for k, v in anomalies.items()}
                }
            )
        except ValueError as e:
            logger.warning(f"Invalid input for generate_user_engagement_chart: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating user engagement chart: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate user engagement chart")

    async def analyze_user_behavior(self, start_date: str, end_date: str) -> ChartData:
        try:
            start, end = self._validate_date_range(start_date, end_date)
            
            # Fetch user behavior data from Supabase
            result = self.supabase.table('user_activities').select('user_id, activity_type, count').gte('date', start).lte('date', end).execute()
            
            if not result.data:
                raise ValueError("No user behavior data found for the given date range")

            # Process data for clustering
            user_data = defaultdict(lambda: defaultdict(int))
            for entry in result.data:
                user_data[entry['user_id']][entry['activity_type']] += entry['count']

            # Perform clustering
            feature_names = sorted(set(activity_type for user in user_data.values() for activity_type in user))
            X = np.array([[user[feature] for feature in feature_names] for user in user_data.values()])
            labels = self._simple_kmeans(X, n_clusters=3)

            # Create chart
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
            plt.title('User Behavior Clustering')
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
            plt.colorbar(scatter, label='Cluster')
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
                    "total_users": len(user_data),
                    "features_analyzed": feature_names,
                    "num_clusters": 3
                }
            )
        except ValueError as e:
            logger.warning(f"Invalid input for analyze_user_behavior: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error analyzing user behavior: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to analyze user behavior")

    async def predict_learning_outcomes(self, user_id: str, course_id: str) -> Dict[str, Any]:
        try:
            # Fetch user's course progress and engagement data
            progress_result = self.supabase.table('course_progress').select('progress_percentage, date').eq('user_id', user_id).eq('course_id', course_id).execute()
            engagement_result = self.supabase.table('user_activities').select('count, date').eq('user_id', user_id).eq('course_id', course_id).execute()

            if not progress_result.data or not engagement_result.data:
                raise ValueError("Insufficient data for prediction")

            # Process data
            progress_data = sorted([(datetime.strptime(entry['date'], "%Y-%m-%d"), entry['progress_percentage']) for entry in progress_result.data])
            engagement_data = sorted([(datetime.strptime(entry['date'], "%Y-%m-%d"), entry['count']) for entry in engagement_result.data])

            # Simple prediction based on recent progress and engagement
            recent_progress_rate = (progress_data[-1][1] - progress_data[0][1]) / (progress_data[-1][0] - progress_data[0][0]).days
            recent_engagement_rate = sum(entry[1] for entry in engagement_data[-7:]) / 7  # Average engagement for the last 7 days

            # Predict completion date and final score
            days_to_completion = (100 - progress_data[-1][1]) / recent_progress_rate if recent_progress_rate > 0 else float('inf')
            predicted_completion_date = progress_data[-1][0] + timedelta(days=days_to_completion)
            predicted_final_score = min(100, progress_data[-1][1] + recent_engagement_rate * days_to_completion / 2)

            return {
                "user_id": user_id,
                "course_id": course_id,
                "predicted_completion_date": predicted_completion_date.strftime("%Y-%m-%d"),
                "predicted_final_score": round(predicted_final_score, 2),
                "confidence": "Medium"  # Since this is a simple prediction, we use a fixed confidence level
            }
        except ValueError as e:
            logger.warning(f"Invalid input for predict_learning_outcomes: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error predicting learning outcomes: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to predict learning outcomes")

    @staticmethod
    def _calculate_moving_average(data: List[float], window_size: int) -> List[float]:
        return [sum(data[i:i+window_size]) / window_size for i in range(len(data) - window_size + 1)]

    @staticmethod
    def _detect_anomalies(data: Dict[str, List[float]], threshold: float = 3.0) -> Dict[str, List[int]]:
        anomalies = {}
        for activity, counts in data.items():
            mean = np.mean(counts)
            std = np.std(counts)
            z_scores = [(count - mean) / std for count in counts]
            anomalies[activity] = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
        return anomalies

    @staticmethod
    def _simple_kmeans(X: np.ndarray, n_clusters: int, max_iters: int = 100) -> np.ndarray:
        # Initialize centroids randomly
        centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
        
        for _ in range(max_iters):
            # Assign points to nearest centroid
            labels = np.argmin(np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2), axis=1)
            
            # Update centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])
            
            # Check for convergence
            if np.all(centroids == new_centroids):
                break
            
            centroids = new_centroids
        
        return labels

data_visualization_service = DataVisualizationService()

def get_data_visualization_service() -> DataVisualizationService:
    return data_visualization_service
