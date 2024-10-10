from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from services.openai_client import send_openai_request
from services.emotion_analysis import analyze_emotion
from models.user import User

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/tutor/chat")
async def chat_with_tutor(message: str, user: User = Depends(oauth2_scheme)):
    try:
        emotion = analyze_emotion(message)
        prompt = f"User emotion: {emotion}\nUser message: {message}\nRespond as an empathetic AI tutor:"
        response = send_openai_request(prompt)
        return {"response": response, "emotion": emotion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tutor/explain")
async def explain_concept(concept: str, user: User = Depends(oauth2_scheme)):
    try:
        prompt = f"Explain the following concept in simple terms: {concept}"
        explanation = send_openai_request(prompt)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
