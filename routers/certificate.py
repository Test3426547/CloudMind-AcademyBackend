from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.certificate_service import CertificateService, get_certificate_service
from typing import Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class CertificateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    course_id: str = Field(..., min_length=1)
    completion_date: str = Field(..., min_length=1)

class CertificateVerificationRequest(BaseModel):
    certificate_hash: str = Field(..., min_length=64, max_length=64)

@router.post("/certificates/create")
async def create_certificate(
    request: CertificateRequest,
    user: User = Depends(oauth2_scheme),
    certificate_service: CertificateService = Depends(get_certificate_service),
):
    try:
        result = await certificate_service.create_certificate(request.user_id, request.course_id, request.completion_date)
        logger.info(f"Certificate created for user {request.user_id}, course {request.course_id}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in create_certificate: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in create_certificate: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while creating the certificate")

@router.post("/certificates/verify")
async def verify_certificate(
    request: CertificateVerificationRequest,
    user: User = Depends(oauth2_scheme),
    certificate_service: CertificateService = Depends(get_certificate_service),
):
    try:
        is_valid = await certificate_service.verify_certificate(request.certificate_hash)
        logger.info(f"Certificate verification completed for hash {request.certificate_hash}")
        return {"is_valid": is_valid}
    except HTTPException as e:
        logger.warning(f"HTTP error in verify_certificate: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in verify_certificate: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while verifying the certificate")
