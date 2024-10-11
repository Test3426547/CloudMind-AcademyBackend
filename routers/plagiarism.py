from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from typing import List
from services.plagiarism_detection import PlagiarismDetectionService, get_plagiarism_detection_service
from models.user import User
import logging
from fastapi_limiter.depends import RateLimiter

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class PlagiarismCheckRequest(BaseModel):
    submitted_text: str = Field(..., min_length=10, max_length=10000)
    original_texts: List[str] = Field(..., min_items=1, max_items=10)

class BatchPlagiarismCheckRequest(BaseModel):
    submitted_texts: List[str] = Field(..., min_items=1, max_items=10)
    original_texts: List[str] = Field(..., min_items=1, max_items=10)

class PlagiarismCheckResponse(BaseModel):
    is_plagiarized: bool
    overall_similarity: float
    detailed_results: List[dict]

@router.post("/plagiarism/check", response_model=PlagiarismCheckResponse)
async def check_plagiarism(
    request: PlagiarismCheckRequest,
    user: User = Depends(oauth2_scheme),
    plagiarism_service: PlagiarismDetectionService = Depends(get_plagiarism_detection_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        result = await plagiarism_service.detect_plagiarism(request.submitted_text, request.original_texts)
        return PlagiarismCheckResponse(**result)
    except ValueError as e:
        logger.warning(f"Invalid input for plagiarism check: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Error during plagiarism check: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during plagiarism detection. Please try again later.")

@router.post("/plagiarism/batch-check", response_model=List[PlagiarismCheckResponse])
async def batch_check_plagiarism(
    request: BatchPlagiarismCheckRequest,
    user: User = Depends(oauth2_scheme),
    plagiarism_service: PlagiarismDetectionService = Depends(get_plagiarism_detection_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=2, seconds=60))
):
    try:
        results = await plagiarism_service.batch_plagiarism_check(request.submitted_texts, request.original_texts)
        return [PlagiarismCheckResponse(**result) for result in results]
    except ValueError as e:
        logger.warning(f"Invalid input for batch plagiarism check: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Error during batch plagiarism check: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during batch plagiarism detection. Please try again later.")
