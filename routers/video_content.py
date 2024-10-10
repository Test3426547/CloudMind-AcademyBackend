from fastapi import APIRouter, Depends, HTTPException
from typing import List
from models.video_content import VideoContent, VideoContentCreate
from services.video_content_service import get_video_content_service, VideoContentService

router = APIRouter()

@router.post("/video-content", response_model=VideoContent)
async def create_video_content(video_content: VideoContentCreate, service: VideoContentService = Depends(get_video_content_service)):
    return await service.create_video_content(video_content)

@router.get("/video-content/{video_id}", response_model=VideoContent)
async def get_video_content(video_id: str, service: VideoContentService = Depends(get_video_content_service)):
    video_content = await service.get_video_content(video_id)
    if video_content is None:
        raise HTTPException(status_code=404, detail="Video content not found")
    return video_content

@router.get("/video-content", response_model=List[VideoContent])
async def list_video_contents(skip: int = 0, limit: int = 10, service: VideoContentService = Depends(get_video_content_service)):
    return await service.list_video_contents(skip, limit)

@router.get("/video-content/search", response_model=List[VideoContent])
async def search_video_contents(query: str, service: VideoContentService = Depends(get_video_content_service)):
    return await service.search_video_contents(query)

@router.put("/video-content/{video_id}", response_model=VideoContent)
async def update_video_content(video_id: str, video_content: VideoContentCreate, service: VideoContentService = Depends(get_video_content_service)):
    updated_video = await service.update_video_content(video_id, video_content)
    if updated_video is None:
        raise HTTPException(status_code=404, detail="Video content not found")
    return updated_video

@router.delete("/video-content/{video_id}", response_model=bool)
async def delete_video_content(video_id: str, service: VideoContentService = Depends(get_video_content_service)):
    deleted = await service.delete_video_content(video_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Video content not found")
    return True
