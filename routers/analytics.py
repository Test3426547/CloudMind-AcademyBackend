from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from services.emotion_analysis import analyze_emotion
from services.sentiment_analysis import analyze_sentiment
from services.llm_orchestrator import LLMOrchestrator
from models.user import User
from typing import Dict, Any

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
llm_orchestrator = LLMOrchestrator()

@router.post("/analytics/emotion")
async def analyze_user_emotion(text: str, user: User = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        emotion_analysis = analyze_emotion(text)
        
        # Use LLMOrchestrator for advanced emotion analysis
        prompt = f"Provide a detailed interpretation of the following emotion analysis:\n\n{emotion_analysis}"
        advanced_analysis = llm_orchestrator.process_request([
            {"role": "system", "content": "You are an expert in emotion analysis."},
            {"role": "user", "content": prompt}
        ], "medium")
        
        emotion_analysis["advanced_analysis"] = advanced_analysis
        return emotion_analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytics/sentiment")
async def analyze_user_sentiment(text: str, user: User = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        sentiment_result = analyze_sentiment(text)
        
        # Use LLMOrchestrator for advanced sentiment analysis
        prompt = f"Provide a detailed interpretation of the following sentiment analysis:\n\n{sentiment_result}"
        advanced_analysis = llm_orchestrator.process_request([
            {"role": "system", "content": "You are an expert in sentiment analysis."},
            {"role": "user", "content": prompt}
        ], "medium")
        
        return {
            "sentiment_analysis": sentiment_result,
            "advanced_analysis": advanced_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytics/content-insights")
async def get_content_insights(text: str, user: User = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        emotion_analysis = analyze_emotion(text)
        sentiment_result = analyze_sentiment(text)
        
        # Use LLMOrchestrator for comprehensive content insights
        prompt = f"""
        Provide comprehensive content insights based on the following emotion and sentiment analysis:
        
        Emotion Analysis: {emotion_analysis}
        Sentiment Analysis: {sentiment_result}
        
        Consider the following aspects in your analysis:
        1. Overall emotional tone of the content
        2. Potential impact on the audience
        3. Suggestions for improvement or optimization
        4. Any potential red flags or areas of concern
        5. Alignment with educational goals and values
        """
        
        insights = llm_orchestrator.process_request([
            {"role": "system", "content": "You are an expert in content analysis for educational platforms."},
            {"role": "user", "content": prompt}
        ], "high")
        
        return {
            "emotion_analysis": emotion_analysis,
            "sentiment_analysis": sentiment_result,
            "comprehensive_insights": insights
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
    
    # Use LLMOrchestrator for personalized performance analysis
    prompt = f"""
    Provide a personalized performance analysis and recommendations based on the following user data:
    
    {mock_performance}
    
    Include the following in your analysis:
    1. Overall performance assessment
    2. Specific strengths and how to leverage them
    3. Areas for improvement and strategies to address them
    4. Personalized study plan recommendations
    5. Suggested resources or courses for further improvement
    """
    
    analysis = llm_orchestrator.process_request([
        {"role": "system", "content": "You are an expert in educational performance analysis and personalized learning."},
        {"role": "user", "content": prompt}
    ], "high")
    
    return {
        "performance_data": mock_performance,
        "personalized_analysis": analysis
    }

@router.get("/analytics/course-engagement")
async def get_course_engagement(course_id: str, user: User = Depends(oauth2_scheme)) -> Dict[str, Any]:
    # Here you would typically fetch and analyze the engagement data for a specific course
    mock_engagement = {
        "total_students": 100,
        "average_completion_rate": 75,
        "average_quiz_score": 80,
        "most_engaging_module": "Introduction to Neural Networks"
    }
    
    # Use LLMOrchestrator for course engagement analysis and recommendations
    prompt = f"""
    Analyze the following course engagement data and provide insights and recommendations:
    
    Course ID: {course_id}
    {mock_engagement}
    
    Include the following in your analysis:
    1. Overall engagement assessment
    2. Factors contributing to high/low engagement
    3. Recommendations for improving course engagement
    4. Strategies for increasing completion rates and quiz scores
    5. Suggestions for replicating the success of the most engaging module
    """
    
    analysis = llm_orchestrator.process_request([
        {"role": "system", "content": "You are an expert in educational engagement analysis and course optimization."},
        {"role": "user", "content": prompt}
    ], "high")
    
    return {
        "engagement_data": mock_engagement,
        "engagement_analysis": analysis
    }
