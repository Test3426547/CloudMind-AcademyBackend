from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List
from services.plagiarism_detection import PlagiarismDetectionService, get_plagiarism_detection_service
from models.user import User

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class PlagiarismCheckRequest(BaseModel):
    submitted_text: str
    original_texts: List[str]

class BatchPlagiarismCheckRequest(BaseModel):
    submitted_texts: List[str]
    original_texts: List[str]

class PlagiarismCheckResponse(BaseModel):
    is_plagiarized: bool
    overall_similarity: float
    detailed_results: List[dict]

@router.post("/plagiarism/check", response_model=PlagiarismCheckResponse)
async def check_plagiarism(
    request: PlagiarismCheckRequest,
    user: User = Depends(oauth2_scheme),
    plagiarism_service: PlagiarismDetectionService = Depends(get_plagiarism_detection_service)
):
    try:
        result = plagiarism_service.detect_plagiarism(request.submitted_text, request.original_texts)
        return PlagiarismCheckResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/plagiarism/batch-check", response_model=List[PlagiarismCheckResponse])
async def batch_check_plagiarism(
    request: BatchPlagiarismCheckRequest,
    user: User = Depends(oauth2_scheme),
    plagiarism_service: PlagiarismDetectionService = Depends(get_plagiarism_detection_service)
):
    try:
        results = await plagiarism_service.batch_plagiarism_check(request.submitted_texts, request.original_texts)
        return [PlagiarismCheckResponse(**result) for result in results]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
