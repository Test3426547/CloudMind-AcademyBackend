from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import uuid

class ARVRContent(BaseModel):
    id: str
    title: str
    description: str
    content_type: str
    file_url: str
    interaction_type: str
    duration: Optional[int] = None
    complexity_level: str

class ARVRContentCreate(BaseModel):
    title: str
    description: str
    content_type: str
    file_url: str
    interaction_type: str
    duration: Optional[int] = None
    complexity_level: str

class ARVRSession(BaseModel):
    session_id: str
    content_id: str
    user_id: str
    start_time: str
    end_time: Optional[str] = None
    progress: float

class ARVRService:
    def __init__(self):
        self.contents = {}
        self.sessions = {}
        self.next_id = 1

    async def create_content(self, content: ARVRContentCreate) -> ARVRContent:
        content_id = str(self.next_id)
        self.next_id += 1
        new_content = ARVRContent(id=content_id, **content.dict())
        self.contents[content_id] = new_content
        return new_content

    async def get_content(self, content_id: str) -> Optional[ARVRContent]:
        return self.contents.get(content_id)

    async def list_content(self) -> List[ARVRContent]:
        return list(self.contents.values())

    async def update_content(self, content_id: str, content: ARVRContentCreate) -> Optional[ARVRContent]:
        if content_id not in self.contents:
            return None
        updated_content = ARVRContent(id=content_id, **content.dict())
        self.contents[content_id] = updated_content
        return updated_content

    async def delete_content(self, content_id: str) -> bool:
        if content_id not in self.contents:
            return False
        del self.contents[content_id]
        return True

    async def start_session(self, content_id: str, user_id: str) -> Optional[ARVRSession]:
        if content_id not in self.contents:
            return None
        session_id = str(uuid.uuid4())
        start_time = datetime.now().isoformat()
        session = ARVRSession(
            session_id=session_id,
            content_id=content_id,
            user_id=user_id,
            start_time=start_time,
            progress=0.0
        )
        self.sessions[session_id] = session
        return session

    async def update_session(self, session_id: str, progress: float) -> Optional[ARVRSession]:
        if session_id not in self.sessions:
            return None
        session = self.sessions[session_id]
        session.progress = progress
        if progress >= 100:
            session.end_time = datetime.now().isoformat()
        return session

    async def get_session(self, session_id: str) -> Optional[ARVRSession]:
        return self.sessions.get(session_id)

ar_vr_service = ARVRService()

def get_ar_vr_service() -> ARVRService:
    return ar_vr_service
