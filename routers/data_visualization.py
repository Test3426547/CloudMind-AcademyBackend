from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.data_visualization_service import DataVisualizationService, get_data_visualization_service
from services.time_tracking_service import TimeTrackingService, get_time_tracking_service
from typing import Dict, Any
from datetime import datetime, timedelta

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.get("/visualize/productivity")
async def visualize_productivity(
    user: User = Depends(oauth2_scheme),
    data_viz_service: DataVisualizationService = Depends(get_data_visualization_service),
    time_tracking_service: TimeTrackingService = Depends(get_time_tracking_service)
):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    analytics = await time_tracking_service.get_productivity_analytics(user.id, start_date, end_date)
    
    productivity_trend = data_viz_service.create_line_chart(
        {"Productivity Score": analytics["productivity_score"]},
        "Productivity Trend",
        "Days",
        "Productivity Score"
    )
    
    task_breakdown = data_viz_service.create_pie_chart(
        analytics["task_breakdown"],
        "Task Time Allocation"
    )
    
    task_efficiency = data_viz_service.create_bar_chart(
        analytics["task_efficiency"],
        "Task Efficiency",
        "Tasks",
        "Efficiency (%)"
    )
    
    return {
        "productivity_trend": productivity_trend,
        "task_breakdown": task_breakdown,
        "task_efficiency": task_efficiency
    }

@router.get("/visualize/learning-progress")
async def visualize_learning_progress(
    user: User = Depends(oauth2_scheme),
    data_viz_service: DataVisualizationService = Depends(get_data_visualization_service)
):
    # Mock data for learning progress
    course_progress = {
        "Python Basics": 80,
        "Data Structures": 65,
        "Algorithms": 50,
        "Machine Learning": 30,
        "Web Development": 90
    }
    
    progress_chart = data_viz_service.create_bar_chart(
        course_progress,
        "Learning Progress by Course",
        "Courses",
        "Progress (%)"
    )
    
    return {"learning_progress": progress_chart}

@router.get("/visualize/engagement")
async def visualize_engagement(
    user: User = Depends(oauth2_scheme),
    data_viz_service: DataVisualizationService = Depends(get_data_visualization_service)
):
    # Mock data for user engagement
    engagement_data = {
        "Course Completion Rate": [75, 80, 85, 82, 88, 90, 92],
        "Quiz Participation": [60, 65, 70, 75, 80, 85, 90],
        "Forum Activity": [40, 45, 50, 55, 60, 65, 70]
    }
    
    engagement_chart = data_viz_service.create_line_chart(
        engagement_data,
        "User Engagement Metrics",
        "Weeks",
        "Engagement Score"
    )
    
    return {"engagement_metrics": engagement_chart}
