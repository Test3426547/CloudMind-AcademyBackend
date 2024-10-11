from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.live_streaming_service import LiveStreamingService, get_live_streaming_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from fastapi_limiter.depends import RateLimiter
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class StreamInfo(BaseModel):
    stream_id: str = Field(..., min_length=1, max_length=50)
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    instructor_id: str = Field(..., min_length=1, max_length=50)

@router.post("/create-stream")
async def create_stream(
    stream_info: StreamInfo,
    user: User = Depends(oauth2_scheme),
    live_streaming_service: LiveStreamingService = Depends(get_live_streaming_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        stream = await live_streaming_service.create_stream(stream_info.dict(), user.id)
        logger.info(f"Stream created successfully by user {user.id}")
        return stream
    except ValueError as e:
        logger.warning(f"Invalid input for create_stream: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating stream: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while creating the stream")

@router.get("/streams")
async def list_streams(
    user: User = Depends(oauth2_scheme),
    live_streaming_service: LiveStreamingService = Depends(get_live_streaming_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60)),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    try:
        streams = await live_streaming_service.list_streams(limit, offset)
        logger.info(f"Streams listed successfully for user {user.id}")
        return streams
    except Exception as e:
        logger.error(f"Error listing streams: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while listing streams")

@router.websocket("/ws/stream/{stream_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    stream_id: str,
    live_streaming_service: LiveStreamingService = Depends(get_live_streaming_service)
):
    try:
        await websocket.accept()
        await live_streaming_service.join_stream(websocket, stream_id)
    except WebSocketDisconnect:
        await live_streaming_service.leave_stream(websocket, stream_id)
    except Exception as e:
        logger.error(f"Error in websocket connection: {str(e)}")
        await websocket.close(code=1000)

@router.post("/streams/{stream_id}/end")
async def end_stream(
    stream_id: str,
    user: User = Depends(oauth2_scheme),
    live_streaming_service: LiveStreamingService = Depends(get_live_streaming_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        await live_streaming_service.end_stream(stream_id, user.id)
        logger.info(f"Stream {stream_id} ended successfully by user {user.id}")
        return {"message": "Stream ended successfully"}
    except ValueError as e:
        logger.warning(f"Invalid input for end_stream: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error ending stream: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while ending the stream")

@router.get("/streams/{stream_id}")
async def get_stream_details(
    stream_id: str,
    user: User = Depends(oauth2_scheme),
    live_streaming_service: LiveStreamingService = Depends(get_live_streaming_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=10, seconds=60))
):
    try:
        stream_details = await live_streaming_service.get_stream_details(stream_id)
        logger.info(f"Stream details retrieved successfully for stream {stream_id} by user {user.id}")
        return stream_details
    except ValueError as e:
        logger.warning(f"Invalid input for get_stream_details: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving stream details: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving stream details")
