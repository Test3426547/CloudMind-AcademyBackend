from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from services.openai_client import send_openai_request
from services.emotion_analysis import analyze_emotion
from models.user import User
from typing import Dict

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/tutor/chat")
async def chat_with_tutor(message: str, user: User = Depends(oauth2_scheme)) -> Dict[str, str]:
    try:
        user_emotion = analyze_emotion(message)
        prompt = f"User emotion: {user_emotion}\nUser message: {message}\nRespond as an empathetic AI tutor, addressing the user's emotional state:"
        response = send_openai_request(prompt)
        
        # Analyze the emotion of the AI's response
        ai_emotion = analyze_emotion(response)
        
        return {
            "user_emotion": user_emotion,
            "ai_response": response,
            "ai_emotion": ai_emotion
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tutor/explain")
async def explain_concept(concept: str, user: User = Depends(oauth2_scheme)) -> Dict[str, str]:
    try:
        user_emotion = analyze_emotion(f"Explain {concept}")
        prompt = f"User emotion: {user_emotion}\nExplain the following concept in simple terms, considering the user's emotional state: {concept}"
        explanation = send_openai_request(prompt)
        
        return {
            "user_emotion": user_emotion,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
