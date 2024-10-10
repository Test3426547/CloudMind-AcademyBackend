from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from services.emotion_analysis import analyze_emotion
from services.sentiment_analysis import analyze_sentiment
from models.user import User
from typing import Dict, Any

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/analytics/emotion")
async def analyze_user_emotion(text: str, user: User = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        emotion_analysis = analyze_emotion(text)
        return emotion_analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytics/sentiment")
async def analyze_user_sentiment(text: str, user: User = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        sentiment_result = analyze_sentiment(text)
        return {"sentiment_analysis": sentiment_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytics/content-insights")
async def get_content_insights(text: str, user: User = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        emotion_analysis = analyze_emotion(text)
        sentiment_result = analyze_sentiment(text)
        
        return {
            "emotion_analysis": emotion_analysis,
            "sentiment_analysis": sentiment_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/user-performance")
async def get_user_performance(user: User = Depends(oauth2_scheme)) -> Dict[str, Any]:
    # Here you would typically fetch and analyze the user's performance data
    mock_performance = {
        "courses_completed": 5,
        "average_quiz_score": 85,
        "total_study_time": 120,  # in hours
        "strengths": ["Python", "Data Structures"],
        "areas_for_improvement": ["Algorithms", "Machine Learning"]
    }
    return mock_performance

@router.get("/analytics/course-engagement")
async def get_course_engagement(course_id: str, user: User = Depends(oauth2_scheme)) -> Dict[str, Any]:
    # Here you would typically fetch and analyze the engagement data for a specific course
    mock_engagement = {
        "total_students": 100,
        "average_completion_rate": 75,
        "average_quiz_score": 80,
        "most_engaging_module": "Introduction to Neural Networks"
    }
    return mock_engagement
