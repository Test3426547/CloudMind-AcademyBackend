from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from models.certificate import Certificate, CertificateCreate
from services.certificate_service import CertificateService, get_certificate_service
from typing import List

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/certificates", response_model=Certificate)
async def create_certificate(
    certificate: CertificateCreate,
    user: User = Depends(oauth2_scheme),
    cert_service: CertificateService = Depends(get_certificate_service)
):
    new_certificate = await cert_service.create_certificate(certificate, user.id)
    return new_certificate

@router.get("/certificates/{certificate_id}", response_model=Certificate)
async def get_certificate(
    certificate_id: str,
    user: User = Depends(oauth2_scheme),
    cert_service: CertificateService = Depends(get_certificate_service)
):
    certificate = await cert_service.get_certificate(certificate_id)
    if certificate is None:
        raise HTTPException(status_code=404, detail="Certificate not found")
    return certificate

@router.get("/certificates", response_model=List[Certificate])
async def list_certificates(
    user: User = Depends(oauth2_scheme),
    cert_service: CertificateService = Depends(get_certificate_service)
):
    return await cert_service.list_certificates(user.id)

@router.post("/certificates/{certificate_id}/verify")
async def verify_certificate(
    certificate_id: str,
    certificate_hash: str,
    user: User = Depends(oauth2_scheme),
    cert_service: CertificateService = Depends(get_certificate_service)
):
    is_valid = await cert_service.verify_certificate(certificate_id, certificate_hash)
    return {"is_valid": is_valid}

@router.post("/certificates/{certificate_id}/revoke")
async def revoke_certificate(
    certificate_id: str,
    user: User = Depends(oauth2_scheme),
    cert_service: CertificateService = Depends(get_certificate_service)
):
    await cert_service.revoke_certificate(certificate_id)
    return {"message": "Certificate revoked successfully"}
