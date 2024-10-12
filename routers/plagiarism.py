from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.plagiarism_detection_service import PlagiarismDetectionService, get_plagiarism_detection_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class PlagiarismCheckRequest(BaseModel):
    text: str = Field(..., min_length=10)
    original_sources: List[str] = Field(..., min_items=1)

class BatchPlagiarismCheckRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)
    original_sources: List[str] = Field(..., min_items=1)

@router.post("/check-plagiarism")
async def check_plagiarism(
    request: PlagiarismCheckRequest,
    user: User = Depends(oauth2_scheme),
    plagiarism_service: PlagiarismDetectionService = Depends(get_plagiarism_detection_service),
):
    try:
        result = await plagiarism_service.check_plagiarism(request.text, request.original_sources)
        logger.info(f"Plagiarism check completed for user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in check_plagiarism: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in check_plagiarism: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during plagiarism check")

@router.post("/batch-check-plagiarism")
async def batch_check_plagiarism(
    request: BatchPlagiarismCheckRequest,
    user: User = Depends(oauth2_scheme),
    plagiarism_service: PlagiarismDetectionService = Depends(get_plagiarism_detection_service),
):
    try:
        results = await plagiarism_service.batch_check_plagiarism(request.texts, request.original_sources)
        logger.info(f"Batch plagiarism check completed for user {user.id}")
        return results
    except HTTPException as e:
        logger.warning(f"HTTP error in batch_check_plagiarism: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in batch_check_plagiarism: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during batch plagiarism check")

@router.get("/plagiarism-threshold")
async def get_plagiarism_threshold(
    text: str = Query(..., min_length=10),
    user: User = Depends(oauth2_scheme),
    plagiarism_service: PlagiarismDetectionService = Depends(get_plagiarism_detection_service),
):
    try:
        threshold = await plagiarism_service._determine_threshold(text)
        logger.info(f"Plagiarism threshold determined for user {user.id}")
        return {"threshold": threshold}
    except Exception as e:
        logger.error(f"Error determining plagiarism threshold: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while determining the plagiarism threshold")
