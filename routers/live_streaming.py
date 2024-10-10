from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.live_streaming_service import LiveStreamingService, get_live_streaming_service
from typing import List, Dict, Any
from pydantic import BaseModel

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class StreamInfo(BaseModel):
    stream_id: str
    title: str
    description: str
    instructor_id: str

@router.post("/create-stream")
async def create_stream(
    stream_info: StreamInfo,
    user: User = Depends(oauth2_scheme),
    live_streaming_service: LiveStreamingService = Depends(get_live_streaming_service)
):
    stream = await live_streaming_service.create_stream(stream_info.dict(), user.id)
    return stream

@router.get("/streams")
async def list_streams(
    user: User = Depends(oauth2_scheme),
    live_streaming_service: LiveStreamingService = Depends(get_live_streaming_service)
):
    streams = await live_streaming_service.list_streams()
    return streams

@router.websocket("/ws/stream/{stream_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    stream_id: str,
    live_streaming_service: LiveStreamingService = Depends(get_live_streaming_service)
):
    await websocket.accept()
    try:
        await live_streaming_service.join_stream(websocket, stream_id)
    except WebSocketDisconnect:
        await live_streaming_service.leave_stream(websocket, stream_id)

@router.post("/streams/{stream_id}/end")
async def end_stream(
    stream_id: str,
    user: User = Depends(oauth2_scheme),
    live_streaming_service: LiveStreamingService = Depends(get_live_streaming_service)
):
    await live_streaming_service.end_stream(stream_id, user.id)
    return {"message": "Stream ended successfully"}
