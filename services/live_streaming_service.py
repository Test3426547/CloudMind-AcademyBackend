import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
import random
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service

logger = logging.getLogger(__name__)

class LiveStreamingService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.active_streams = {}
        self.chat_history = {}
        self.transcriptions = {}

    async def create_stream(self, user_id: str, stream_title: str) -> Dict[str, Any]:
        stream_id = f"stream_{random.randint(1000, 9999)}"
        self.active_streams[stream_id] = {
            "user_id": user_id,
            "title": stream_title,
            "viewers": 0,
            "status": "live"
        }
        self.chat_history[stream_id] = []
        self.transcriptions[stream_id] = []
        return {"stream_id": stream_id, "stream_url": f"https://example.com/stream/{stream_id}"}

    async def end_stream(self, stream_id: str) -> Dict[str, str]:
        if stream_id not in self.active_streams:
            raise HTTPException(status_code=404, detail="Stream not found")
        self.active_streams[stream_id]["status"] = "ended"
        return {"message": "Stream ended successfully"}

    async def get_stream_info(self, stream_id: str) -> Dict[str, Any]:
        if stream_id not in self.active_streams:
            raise HTTPException(status_code=404, detail="Stream not found")
        return self.active_streams[stream_id]

    async def update_viewer_count(self, stream_id: str, count: int) -> Dict[str, int]:
        if stream_id not in self.active_streams:
            raise HTTPException(status_code=404, detail="Stream not found")
        self.active_streams[stream_id]["viewers"] = count
        return {"viewers": count}

    async def add_chat_message(self, stream_id: str, user_id: str, message: str) -> Dict[str, Any]:
        if stream_id not in self.active_streams:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        sentiment = await self._analyze_sentiment(message)
        is_appropriate = await self._moderate_content(message)
        
        chat_entry = {
            "user_id": user_id,
            "message": message,
            "sentiment": sentiment,
            "is_appropriate": is_appropriate
        }
        
        self.chat_history[stream_id].append(chat_entry)
        return chat_entry

    async def get_chat_history(self, stream_id: str) -> List[Dict[str, Any]]:
        if stream_id not in self.chat_history:
            raise HTTPException(status_code=404, detail="Chat history not found")
        return self.chat_history[stream_id]

    async def add_transcription(self, stream_id: str, text: str) -> Dict[str, str]:
        if stream_id not in self.active_streams:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        transcription = await self._process_transcription(text)
        self.transcriptions[stream_id].append(transcription)
        return {"transcription": transcription}

    async def get_transcriptions(self, stream_id: str) -> List[str]:
        if stream_id not in self.transcriptions:
            raise HTTPException(status_code=404, detail="Transcriptions not found")
        return self.transcriptions[stream_id]

    async def _analyze_sentiment(self, text: str) -> str:
        prompt = f"Analyze the sentiment of the following text and respond with either 'positive', 'neutral', or 'negative':\n\n{text}"
        sentiment = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are a sentiment analysis expert. Provide a single-word response: 'positive', 'neutral', or 'negative'."},
            {"role": "user", "content": prompt}
        ], "low")
        return sentiment.strip().lower()

    async def _moderate_content(self, text: str) -> bool:
        prompt = f"Determine if the following text is appropriate for a public chat. Respond with 'yes' if it's appropriate, or 'no' if it contains inappropriate content:\n\n{text}"
        response = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are a content moderation expert. Provide a single-word response: 'yes' or 'no'."},
            {"role": "user", "content": prompt}
        ], "low")
        return response.strip().lower() == "yes"

    async def _process_transcription(self, text: str) -> str:
        # Simulate advanced processing (e.g., punctuation, capitalization)
        prompt = f"Process the following speech-to-text transcription, adding proper punctuation and capitalization:\n\n{text}"
        processed_text = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an expert in processing speech-to-text transcriptions. Improve the text with proper punctuation and capitalization."},
            {"role": "user", "content": prompt}
        ], "low")
        return processed_text.strip()

live_streaming_service = LiveStreamingService(get_llm_orchestrator(), get_text_embedding_service())

def get_live_streaming_service() -> LiveStreamingService:
    return live_streaming_service
