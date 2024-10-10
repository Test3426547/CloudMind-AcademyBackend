from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.ai_tutor import get_ai_tutor_service, AITutorService
from typing import List, Dict
from pydantic import BaseModel
from datetime import datetime
import uuid

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class CollaborationMessage(BaseModel):
    user_id: str
    content: str
    timestamp: str

class CollaborationSession(BaseModel):
    session_id: str
    participants: List[str]
    messages: List[CollaborationMessage]

collaboration_sessions: Dict[str, CollaborationSession] = {}

@router.post("/collaboration/create-session")
async def create_collaboration_session(participants: List[str], user: User = Depends(oauth2_scheme)):
    session_id = str(uuid.uuid4())
    collaboration_sessions[session_id] = CollaborationSession(
        session_id=session_id,
        participants=participants,
        messages=[]
    )
    return {"session_id": session_id}

@router.post("/collaboration/{session_id}/send-message")
async def send_collaboration_message(
    session_id: str,
    message: str,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service)
):
    if session_id not in collaboration_sessions:
        raise HTTPException(status_code=404, detail="Collaboration session not found")
    
    session = collaboration_sessions[session_id]
    if user.id not in session.participants:
        raise HTTPException(status_code=403, detail="User is not a participant in this session")
    
    # Add user message to the session
    user_message = CollaborationMessage(user_id=user.id, content=message, timestamp=str(datetime.now()))
    session.messages.append(user_message)
    
    # Get AI chatbot assistance using LLMOrchestrator
    ai_response = await ai_tutor_service.get_collaboration_assistance(message)
    ai_message = CollaborationMessage(user_id="AI_Assistant", content=ai_response, timestamp=str(datetime.now()))
    session.messages.append(ai_message)
    
    return {"user_message": user_message, "ai_response": ai_message}

@router.get("/collaboration/{session_id}/messages")
async def get_collaboration_messages(session_id: str, user: User = Depends(oauth2_scheme)):
    if session_id not in collaboration_sessions:
        raise HTTPException(status_code=404, detail="Collaboration session not found")
    
    session = collaboration_sessions[session_id]
    if user.id not in session.participants:
        raise HTTPException(status_code=403, detail="User is not a participant in this session")
    
    return {"messages": session.messages}

@router.post("/collaboration/{session_id}/invite")
async def invite_participant(session_id: str, new_participant: str, user: User = Depends(oauth2_scheme)):
    if session_id not in collaboration_sessions:
        raise HTTPException(status_code=404, detail="Collaboration session not found")
    
    session = collaboration_sessions[session_id]
    if user.id not in session.participants:
        raise HTTPException(status_code=403, detail="User is not a participant in this session")
    
    if new_participant not in session.participants:
        session.participants.append(new_participant)
        return {"message": f"User {new_participant} has been invited to the collaboration session"}
    else:
        return {"message": f"User {new_participant} is already a participant in this session"}

@router.delete("/collaboration/{session_id}")
async def end_collaboration_session(session_id: str, user: User = Depends(oauth2_scheme)):
    if session_id not in collaboration_sessions:
        raise HTTPException(status_code=404, detail="Collaboration session not found")
    
    session = collaboration_sessions[session_id]
    if user.id not in session.participants:
        raise HTTPException(status_code=403, detail="User is not a participant in this session")
    
    del collaboration_sessions[session_id]
    return {"message": "Collaboration session ended successfully"}

@router.post("/collaboration/{session_id}/summarize")
async def summarize_collaboration_session(
    session_id: str,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service)
):
    if session_id not in collaboration_sessions:
        raise HTTPException(status_code=404, detail="Collaboration session not found")
    
    session = collaboration_sessions[session_id]
    if user.id not in session.participants:
        raise HTTPException(status_code=403, detail="User is not a participant in this session")
    
    messages = [f"{msg.user_id}: {msg.content}" for msg in session.messages]
    summary = await ai_tutor_service.summarize_collaboration(messages)
    
    return {"summary": summary}
