from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.supabase_client import EnhancedSupabaseClient, get_enhanced_supabase_client
from typing import List, Dict, Any
from pydantic import BaseModel
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class ClusteringResponse(BaseModel):
    users: List[Dict[str, Any]]
    num_clusters: int

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]

class CourseAnalysisResponse(BaseModel):
    course_id: str
    difficulty: str
    prerequisites: str
    estimated_time: int
    key_topics: List[str]

class SimilarCoursesResponse(BaseModel):
    similar_courses: List[Dict[str, Any]]

@router.get("/cluster-users", response_model=ClusteringResponse)
async def cluster_users(
    num_clusters: int = 3,
    user: User = Depends(oauth2_scheme),
    supabase_client: EnhancedSupabaseClient = Depends(get_enhanced_supabase_client),
):
    try:
        result = await supabase_client.fetch_and_cluster_users(num_clusters)
        logger.info(f"User clustering completed with {num_clusters} clusters")
        return result
    except Exception as e:
        logger.error(f"Error in cluster_users: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during user clustering")

@router.get("/recommend-courses/{user_id}", response_model=RecommendationResponse)
async def recommend_courses(
    user_id: str,
    user: User = Depends(oauth2_scheme),
    supabase_client: EnhancedSupabaseClient = Depends(get_enhanced_supabase_client),
):
    try:
        recommendations = await supabase_client.generate_user_recommendations(user_id)
        logger.info(f"Course recommendations generated for user {user_id}")
        return {"recommendations": recommendations}
    except ValueError as e:
        logger.warning(f"User not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in recommend_courses: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating course recommendations")

@router.get("/analyze-course/{course_id}", response_model=CourseAnalysisResponse)
async def analyze_course(
    course_id: str,
    user: User = Depends(oauth2_scheme),
    supabase_client: EnhancedSupabaseClient = Depends(get_enhanced_supabase_client),
):
    try:
        analysis = await supabase_client.analyze_course_difficulty(course_id)
        logger.info(f"Course analysis completed for course {course_id}")
        return analysis
    except ValueError as e:
        logger.warning(f"Course not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze_course: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while analyzing the course")

@router.get("/similar-courses/{course_id}", response_model=SimilarCoursesResponse)
async def find_similar_courses(
    course_id: str,
    num_similar: int = 5,
    user: User = Depends(oauth2_scheme),
    supabase_client: EnhancedSupabaseClient = Depends(get_enhanced_supabase_client),
):
    try:
        similar_courses = await supabase_client.find_similar_courses(course_id, num_similar)
        logger.info(f"Found {len(similar_courses)} similar courses for course {course_id}")
        return {"similar_courses": similar_courses}
    except ValueError as e:
        logger.warning(f"Course not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in find_similar_courses: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while finding similar courses")
