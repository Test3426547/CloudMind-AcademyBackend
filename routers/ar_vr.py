from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Optional
from services.ar_vr_service import ARVRService, get_ar_vr_service

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class ARVRContent(BaseModel):
    id: Optional[str] = None
    title: str
    description: str
    content_type: str  # e.g., "3D_model", "360_video", "VR_scene"
    file_url: str
    interaction_type: str  # e.g., "view_only", "interactive", "educational"
    duration: Optional[int] = None  # Duration in seconds, if applicable
    complexity_level: str  # e.g., "beginner", "intermediate", "advanced"

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
    progress: float  # Progress percentage

@router.post("/ar-vr/content", response_model=ARVRContent)
async def create_ar_vr_content(content: ARVRContentCreate, token: str = Depends(oauth2_scheme), ar_vr_service: ARVRService = Depends(get_ar_vr_service)):
    return await ar_vr_service.create_content(content)

@router.get("/ar-vr/content/{content_id}", response_model=ARVRContent)
async def get_ar_vr_content(content_id: str, token: str = Depends(oauth2_scheme), ar_vr_service: ARVRService = Depends(get_ar_vr_service)):
    content = await ar_vr_service.get_content(content_id)
    if content is None:
        raise HTTPException(status_code=404, detail="AR/VR content not found")
    return content

@router.get("/ar-vr/content", response_model=List[ARVRContent])
async def list_ar_vr_content(token: str = Depends(oauth2_scheme), ar_vr_service: ARVRService = Depends(get_ar_vr_service)):
    return await ar_vr_service.list_content()

@router.put("/ar-vr/content/{content_id}", response_model=ARVRContent)
async def update_ar_vr_content(content_id: str, content: ARVRContentCreate, token: str = Depends(oauth2_scheme), ar_vr_service: ARVRService = Depends(get_ar_vr_service)):
    updated_content = await ar_vr_service.update_content(content_id, content)
    if updated_content is None:
        raise HTTPException(status_code=404, detail="AR/VR content not found")
    return updated_content

@router.delete("/ar-vr/content/{content_id}", response_model=bool)
async def delete_ar_vr_content(content_id: str, token: str = Depends(oauth2_scheme), ar_vr_service: ARVRService = Depends(get_ar_vr_service)):
    success = await ar_vr_service.delete_content(content_id)
    if not success:
        raise HTTPException(status_code=404, detail="AR/VR content not found")
    return True

@router.post("/ar-vr/session", response_model=ARVRSession)
async def start_ar_vr_session(content_id: str, user_id: str, token: str = Depends(oauth2_scheme), ar_vr_service: ARVRService = Depends(get_ar_vr_service)):
    session = await ar_vr_service.start_session(content_id, user_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Failed to start AR/VR session")
    return session

@router.put("/ar-vr/session/{session_id}", response_model=ARVRSession)
async def update_ar_vr_session(session_id: str, progress: float, token: str = Depends(oauth2_scheme), ar_vr_service: ARVRService = Depends(get_ar_vr_service)):
    updated_session = await ar_vr_service.update_session(session_id, progress)
    if updated_session is None:
        raise HTTPException(status_code=404, detail="AR/VR session not found")
    return updated_session

@router.get("/ar-vr/session/{session_id}", response_model=ARVRSession)
async def get_ar_vr_session(session_id: str, token: str = Depends(oauth2_scheme), ar_vr_service: ARVRService = Depends(get_ar_vr_service)):
    session = await ar_vr_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="AR/VR session not found")
    return session
