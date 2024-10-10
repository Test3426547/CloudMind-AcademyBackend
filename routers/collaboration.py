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
    created_at: str
    last_activity: str

collaboration_sessions: Dict[str, CollaborationSession] = {}

@router.post("/collaboration/create-session")
async def create_collaboration_session(participants: List[str], user: User = Depends(oauth2_scheme)):
    session_id = str(uuid.uuid4())
    current_time = str(datetime.now())
    collaboration_sessions[session_id] = CollaborationSession(
        session_id=session_id,
        participants=participants,
        messages=[],
        created_at=current_time,
        last_activity=current_time
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
    current_time = str(datetime.now())
    user_message = CollaborationMessage(user_id=user.id, content=message, timestamp=current_time)
    session.messages.append(user_message)
    session.last_activity = current_time
    
    # Get AI chatbot assistance using LLMOrchestrator
    context = [{"role": "user" if msg.user_id != "AI_Assistant" else "assistant", "content": msg.content} for msg in session.messages[-5:]]
    ai_response = await ai_tutor_service.get_collaboration_assistance(message, context)
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

@router.get("/collaboration/{session_id}/info")
async def get_collaboration_session_info(session_id: str, user: User = Depends(oauth2_scheme)):
    if session_id not in collaboration_sessions:
        raise HTTPException(status_code=404, detail="Collaboration session not found")
    
    session = collaboration_sessions[session_id]
    if user.id not in session.participants:
        raise HTTPException(status_code=403, detail="User is not a participant in this session")
    
    return {
        "session_id": session.session_id,
        "participants": session.participants,
        "created_at": session.created_at,
        "last_activity": session.last_activity,
        "message_count": len(session.messages)
    }

@router.post("/collaboration/{session_id}/leave")
async def leave_collaboration_session(session_id: str, user: User = Depends(oauth2_scheme)):
    if session_id not in collaboration_sessions:
        raise HTTPException(status_code=404, detail="Collaboration session not found")
    
    session = collaboration_sessions[session_id]
    if user.id not in session.participants:
        raise HTTPException(status_code=403, detail="User is not a participant in this session")
    
    session.participants.remove(user.id)
    
    if len(session.participants) == 0:
        del collaboration_sessions[session_id]
        return {"message": "You have left the session. The session has been closed as there are no participants left."}
    
    return {"message": "You have successfully left the collaboration session"}
