from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from typing import List, Dict
from pydantic import BaseModel, Field
from fastapi_limiter.depends import RateLimiter
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class VideoMetadata(BaseModel):
    id: str = Field(..., min_length=1, max_length=50)
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    duration: int = Field(..., gt=0)  # in seconds
    thumbnail_url: str = Field(..., min_length=1, max_length=500)

class VideoProgress(BaseModel):
    video_id: str = Field(..., min_length=1, max_length=50)
    progress: float = Field(..., ge=0, le=1)  # 0 to 1

# Mock data for demonstration
mock_videos = [
    {
        "id": "1",
        "title": "Introduction to Python",
        "description": "Learn the basics of Python programming",
        "duration": 600,
        "thumbnail_url": "https://example.com/thumbnail1.jpg",
        "hls_url": "https://example.com/video1.m3u8"
    },
    {
        "id": "2",
        "title": "Advanced Python Concepts",
        "description": "Dive deeper into Python with advanced topics",
        "duration": 900,
        "thumbnail_url": "https://example.com/thumbnail2.jpg",
        "hls_url": "https://example.com/video2.m3u8"
    }
]

@router.get("/videos", response_model=List[VideoMetadata])
async def get_videos(
    user: User = Depends(oauth2_scheme),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        logger.info(f"User {user.id} requested video list")
        return [VideoMetadata(**video) for video in mock_videos]
    except Exception as e:
        logger.error(f"Error fetching videos: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching videos")

@router.get("/videos/{video_id}", response_model=VideoMetadata)
async def get_video_metadata(
    video_id: str,
    user: User = Depends(oauth2_scheme),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=20, seconds=60))
):
    try:
        logger.info(f"User {user.id} requested metadata for video {video_id}")
        video = next((v for v in mock_videos if v["id"] == video_id), None)
        if not video:
            logger.warning(f"Video {video_id} not found")
            raise HTTPException(status_code=404, detail="Video not found")
        return VideoMetadata(**video)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching video metadata: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching video metadata")

@router.get("/videos/{video_id}/stream")
async def get_video_stream_url(
    video_id: str,
    user: User = Depends(oauth2_scheme),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=30, seconds=60))
):
    try:
        logger.info(f"User {user.id} requested stream URL for video {video_id}")
        video = next((v for v in mock_videos if v["id"] == video_id), None)
        if not video:
            logger.warning(f"Video {video_id} not found")
            raise HTTPException(status_code=404, detail="Video not found")
        return {"hls_url": video["hls_url"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching video stream URL: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching video stream URL")

@router.post("/videos/{video_id}/progress")
async def update_video_progress(
    video_id: str,
    progress: VideoProgress,
    user: User = Depends(oauth2_scheme),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=50, seconds=60))
):
    try:
        logger.info(f"User {user.id} updated progress for video {video_id}: {progress.progress}")
        # In a real implementation, you would update the user's progress in a database
        return {"message": f"Progress updated for video {video_id}"}
    except Exception as e:
        logger.error(f"Error updating video progress: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while updating video progress")

@router.get("/videos/{video_id}/related")
async def get_related_videos(
    video_id: str,
    user: User = Depends(oauth2_scheme),
    limit: int = Query(5, ge=1, le=20),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        logger.info(f"User {user.id} requested related videos for video {video_id}")
        # In a real implementation, you would use a recommendation system
        # For now, we'll just return all other videos
        related_videos = [v for v in mock_videos if v["id"] != video_id][:limit]
        return [VideoMetadata(**video) for video in related_videos]
    except Exception as e:
        logger.error(f"Error fetching related videos: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching related videos")
