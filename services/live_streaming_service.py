from fastapi import WebSocket
from typing import List, Dict, Any
import asyncio
import uuid

class LiveStreamingService:
    def __init__(self):
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.connected_clients: Dict[str, List[WebSocket]] = {}

    async def create_stream(self, stream_info: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        stream_id = str(uuid.uuid4())
        stream = {
            "id": stream_id,
            "title": stream_info["title"],
            "description": stream_info["description"],
            "instructor_id": user_id,
            "status": "active"
        }
        self.active_streams[stream_id] = stream
        self.connected_clients[stream_id] = []
        return stream

    async def list_streams(self) -> List[Dict[str, Any]]:
        return list(self.active_streams.values())

    async def join_stream(self, websocket: WebSocket, stream_id: str):
        if stream_id not in self.active_streams:
            await websocket.close(code=4004)
            return

        self.connected_clients[stream_id].append(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await self.broadcast(stream_id, f"Client says: {data}")
        except Exception as e:
            print(f"Error in websocket connection: {e}")

    async def leave_stream(self, websocket: WebSocket, stream_id: str):
        if stream_id in self.connected_clients:
            self.connected_clients[stream_id].remove(websocket)

    async def end_stream(self, stream_id: str, user_id: str):
        if stream_id in self.active_streams:
            if self.active_streams[stream_id]["instructor_id"] == user_id:
                self.active_streams[stream_id]["status"] = "ended"
                await self.broadcast(stream_id, "Stream has ended")
                for client in self.connected_clients[stream_id]:
                    await client.close(code=4000)
                del self.connected_clients[stream_id]
            else:
                raise ValueError("Only the instructor can end the stream")
        else:
            raise ValueError("Stream not found")

    async def broadcast(self, stream_id: str, message: str):
        if stream_id in self.connected_clients:
            for client in self.connected_clients[stream_id]:
                await client.send_text(message)

live_streaming_service = LiveStreamingService()

def get_live_streaming_service() -> LiveStreamingService:
    return live_streaming_service
