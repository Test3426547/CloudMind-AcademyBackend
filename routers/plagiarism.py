from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List
from services.plagiarism_detection import detect_plagiarism
from models.user import User

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class PlagiarismCheckRequest(BaseModel):
    submitted_text: str
    original_texts: List[str]

class PlagiarismCheckResponse(BaseModel):
    is_plagiarized: bool
    overall_similarity: float
    detailed_results: List[dict]

@router.post("/plagiarism/check", response_model=PlagiarismCheckResponse)
async def check_plagiarism(
    request: PlagiarismCheckRequest,
    user: User = Depends(oauth2_scheme)
):
    try:
        result = detect_plagiarism(request.submitted_text, request.original_texts)
        return PlagiarismCheckResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
