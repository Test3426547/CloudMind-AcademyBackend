from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from services.openai_client import send_openai_request
from models.user import User

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/code/analyze")
async def analyze_code(code: str, language: str, user: User = Depends(oauth2_scheme)):
    try:
        prompt = f"Analyze the following {language} code and provide feedback:\n\n{code}"
        analysis = send_openai_request(prompt)
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/code/suggest")
async def suggest_improvements(code: str, language: str, user: User = Depends(oauth2_scheme)):
    try:
        prompt = f"Suggest improvements for the following {language} code:\n\n{code}"
        suggestions = send_openai_request(prompt)
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
