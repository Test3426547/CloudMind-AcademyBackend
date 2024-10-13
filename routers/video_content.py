from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.video_content_service import VideoContentService, get_video_content_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class VideoUpload(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    url: str = Field(..., min_length=1)

class VideoRecommendation(BaseModel):
    video_id: str
    score: float
    title: str

class VideoAnalysis(BaseModel):
    tags: List[str]
    main_topic: str
    target_audience: str
    difficulty_level: str
    estimated_duration: int
    key_concepts: List[str]
    sentiment: str
    named_entities: List[str]

@router.post("/videos/upload", response_model=Dict[str, Any])
async def upload_video(
    video_data: VideoUpload,
    user: User = Depends(oauth2_scheme),
    video_service: VideoContentService = Depends(get_video_content_service),
):
    try:
        result = await video_service.upload_video(video_data.dict())
        logger.info(f"Video uploaded successfully by user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in upload_video: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in upload_video: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while uploading the video")

@router.get("/videos/{video_id}", response_model=Dict[str, Any])
async def get_video(
    video_id: str,
    user: User = Depends(oauth2_scheme),
    video_service: VideoContentService = Depends(get_video_content_service),
):
    try:
        video = await video_service.get_video(video_id)
        logger.info(f"Video {video_id} retrieved for user {user.id}")
        return video
    except HTTPException as e:
        logger.warning(f"HTTP error in get_video: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_video: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving the video")

@router.get("/videos/recommend", response_model=List[VideoRecommendation])
async def recommend_videos(
    num_recommendations: int = 5,
    user: User = Depends(oauth2_scheme),
    video_service: VideoContentService = Depends(get_video_content_service),
):
    try:
        recommendations = await video_service.recommend_videos(user.id, num_recommendations)
        logger.info(f"Video recommendations generated for user {user.id}")
        return recommendations
    except HTTPException as e:
        logger.warning(f"HTTP error in recommend_videos: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in recommend_videos: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating video recommendations")

@router.get("/videos/{video_id}/summary", response_model=Dict[str, str])
async def get_video_summary(
    video_id: str,
    user: User = Depends(oauth2_scheme),
    video_service: VideoContentService = Depends(get_video_content_service),
):
    try:
        summary = await video_service.generate_video_summary(video_id)
        logger.info(f"Summary generated for video {video_id}")
        return {"video_id": video_id, "summary": summary}
    except HTTPException as e:
        logger.warning(f"HTTP error in get_video_summary: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_video_summary: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the video summary")

@router.get("/videos/cluster", response_model=Dict[str, List[str]])
async def cluster_videos(
    num_clusters: int = 5,
    user: User = Depends(oauth2_scheme),
    video_service: VideoContentService = Depends(get_video_content_service),
):
    try:
        clusters = await video_service.cluster_videos(num_clusters)
        logger.info(f"Videos clustered into {num_clusters} groups")
        return {"clusters": clusters}
    except HTTPException as e:
        logger.warning(f"HTTP error in cluster_videos: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in cluster_videos: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while clustering videos")

@router.get("/videos/{video_id}/analyze", response_model=VideoAnalysis)
async def analyze_video(
    video_id: str,
    user: User = Depends(oauth2_scheme),
    video_service: VideoContentService = Depends(get_video_content_service),
):
    try:
        video = await video_service.get_video(video_id)
        analysis = await video_service.analyze_video_content(video['title'], video['description'])
        logger.info(f"Analysis generated for video {video_id}")
        return VideoAnalysis(**analysis)
    except HTTPException as e:
        logger.warning(f"HTTP error in analyze_video: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in analyze_video: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while analyzing the video")
