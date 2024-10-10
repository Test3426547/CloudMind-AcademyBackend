from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from models.certificate import Certificate, CertificateCreate
from services.certificate_service import CertificateService, get_certificate_service
from services.blockchain_service import BlockchainService, get_blockchain_service
from typing import List
from pydantic import BaseModel

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class VerificationRequest(BaseModel):
    certificate_hash: str

@router.post("/certificates", response_model=Certificate)
async def create_certificate(
    certificate: CertificateCreate,
    token: str = Depends(oauth2_scheme),
    cert_service: CertificateService = Depends(get_certificate_service),
    blockchain_service: BlockchainService = Depends(get_blockchain_service)
):
    # For simplicity, we're not validating the token here. In a real-world scenario, you should validate the token and get the user ID from it.
    user_id = "user456"  # Hardcoded for testing purposes
    new_certificate = await cert_service.create_certificate(certificate, user_id)
    return new_certificate

@router.get("/certificates/{certificate_id}", response_model=Certificate)
async def get_certificate(
    certificate_id: str,
    token: str = Depends(oauth2_scheme),
    cert_service: CertificateService = Depends(get_certificate_service)
):
    certificate = await cert_service.get_certificate(certificate_id)
    if certificate is None:
        raise HTTPException(status_code=404, detail="Certificate not found")
    return certificate

@router.get("/certificates", response_model=List[Certificate])
async def list_certificates(
    token: str = Depends(oauth2_scheme),
    cert_service: CertificateService = Depends(get_certificate_service)
):
    # For simplicity, we're not validating the token here. In a real-world scenario, you should validate the token and get the user ID from it.
    user_id = "user456"  # Hardcoded for testing purposes
    return await cert_service.list_certificates(user_id)

@router.post("/certificates/{certificate_id}/verify")
async def verify_certificate(
    certificate_id: str,
    verification_request: VerificationRequest,
    token: str = Depends(oauth2_scheme),
    cert_service: CertificateService = Depends(get_certificate_service)
):
    is_valid = await cert_service.verify_certificate(certificate_id, verification_request.certificate_hash)
    return {"is_valid": is_valid}

@router.post("/certificates/{certificate_id}/revoke")
async def revoke_certificate(
    certificate_id: str,
    token: str = Depends(oauth2_scheme),
    cert_service: CertificateService = Depends(get_certificate_service)
):
    certificate = await cert_service.get_certificate(certificate_id)
    if certificate is None:
        raise HTTPException(status_code=404, detail="Certificate not found")
    await cert_service.revoke_certificate(certificate_id)
    return {"message": "Certificate revoked successfully"}
