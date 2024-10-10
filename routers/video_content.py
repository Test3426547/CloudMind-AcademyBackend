from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from typing import List, Dict
from pydantic import BaseModel

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class VideoMetadata(BaseModel):
    id: str
    title: str
    description: str
    duration: int  # in seconds
    thumbnail_url: str

class VideoProgress(BaseModel):
    video_id: str
    progress: float  # 0 to 1

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
async def get_videos(user: User = Depends(oauth2_scheme)):
    return [VideoMetadata(**video) for video in mock_videos]

@router.get("/videos/{video_id}", response_model=VideoMetadata)
async def get_video_metadata(video_id: str, user: User = Depends(oauth2_scheme)):
    video = next((v for v in mock_videos if v["id"] == video_id), None)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoMetadata(**video)

@router.get("/videos/{video_id}/stream")
async def get_video_stream_url(video_id: str, user: User = Depends(oauth2_scheme)):
    video = next((v for v in mock_videos if v["id"] == video_id), None)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return {"hls_url": video["hls_url"]}

@router.post("/videos/{video_id}/progress")
async def update_video_progress(video_id: str, progress: VideoProgress, user: User = Depends(oauth2_scheme)):
    # In a real implementation, you would update the user's progress in a database
    return {"message": f"Progress updated for video {video_id}"}

@router.get("/videos/{video_id}/related")
async def get_related_videos(video_id: str, user: User = Depends(oauth2_scheme)):
    # In a real implementation, you would use a recommendation system
    # For now, we'll just return all other videos
    related_videos = [v for v in mock_videos if v["id"] != video_id]
    return [VideoMetadata(**video) for video in related_videos]
