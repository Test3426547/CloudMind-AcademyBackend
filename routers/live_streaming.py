from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.live_streaming_service import LiveStreamingService, get_live_streaming_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class StreamCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)

class TranscriptionInput(BaseModel):
    text: str = Field(..., min_length=1)

@router.post("/streams")
async def create_stream(
    stream: StreamCreate,
    user: User = Depends(oauth2_scheme),
    streaming_service: LiveStreamingService = Depends(get_live_streaming_service),
):
    try:
        result = await streaming_service.create_stream(user.id, stream.title)
        logger.info(f"Stream created by user {user.id}")
        return result
    except Exception as e:
        logger.error(f"Error creating stream: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while creating the stream")

@router.post("/streams/{stream_id}/end")
async def end_stream(
    stream_id: str,
    user: User = Depends(oauth2_scheme),
    streaming_service: LiveStreamingService = Depends(get_live_streaming_service),
):
    try:
        result = await streaming_service.end_stream(stream_id)
        logger.info(f"Stream {stream_id} ended by user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"Error ending stream: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error ending stream: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while ending the stream")

@router.get("/streams/{stream_id}")
async def get_stream_info(
    stream_id: str,
    user: User = Depends(oauth2_scheme),
    streaming_service: LiveStreamingService = Depends(get_live_streaming_service),
):
    try:
        result = await streaming_service.get_stream_info(stream_id)
        logger.info(f"Stream info retrieved for stream {stream_id}")
        return result
    except HTTPException as e:
        logger.warning(f"Error retrieving stream info: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error retrieving stream info: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving stream info")

@router.post("/streams/{stream_id}/viewers")
async def update_viewer_count(
    stream_id: str,
    count: int,
    user: User = Depends(oauth2_scheme),
    streaming_service: LiveStreamingService = Depends(get_live_streaming_service),
):
    try:
        result = await streaming_service.update_viewer_count(stream_id, count)
        logger.info(f"Viewer count updated for stream {stream_id}")
        return result
    except HTTPException as e:
        logger.warning(f"Error updating viewer count: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error updating viewer count: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating viewer count")

@router.post("/streams/{stream_id}/chat")
async def add_chat_message(
    stream_id: str,
    chat_message: ChatMessage,
    user: User = Depends(oauth2_scheme),
    streaming_service: LiveStreamingService = Depends(get_live_streaming_service),
):
    try:
        result = await streaming_service.add_chat_message(stream_id, user.id, chat_message.message)
        logger.info(f"Chat message added to stream {stream_id} by user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"Error adding chat message: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error adding chat message: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while adding chat message")

@router.get("/streams/{stream_id}/chat")
async def get_chat_history(
    stream_id: str,
    user: User = Depends(oauth2_scheme),
    streaming_service: LiveStreamingService = Depends(get_live_streaming_service),
):
    try:
        result = await streaming_service.get_chat_history(stream_id)
        logger.info(f"Chat history retrieved for stream {stream_id}")
        return result
    except HTTPException as e:
        logger.warning(f"Error retrieving chat history: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error retrieving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving chat history")

@router.post("/streams/{stream_id}/transcription")
async def add_transcription(
    stream_id: str,
    transcription: TranscriptionInput,
    user: User = Depends(oauth2_scheme),
    streaming_service: LiveStreamingService = Depends(get_live_streaming_service),
):
    try:
        result = await streaming_service.add_transcription(stream_id, transcription.text)
        logger.info(f"Transcription added to stream {stream_id}")
        return result
    except HTTPException as e:
        logger.warning(f"Error adding transcription: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error adding transcription: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while adding transcription")

@router.get("/streams/{stream_id}/transcription")
async def get_transcriptions(
    stream_id: str,
    user: User = Depends(oauth2_scheme),
    streaming_service: LiveStreamingService = Depends(get_live_streaming_service),
):
    try:
        result = await streaming_service.get_transcriptions(stream_id)
        logger.info(f"Transcriptions retrieved for stream {stream_id}")
        return result
    except HTTPException as e:
        logger.warning(f"Error retrieving transcriptions: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error retrieving transcriptions: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving transcriptions")
