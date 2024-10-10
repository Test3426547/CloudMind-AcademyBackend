from models.certificate import Certificate, CertificateCreate
from typing import List, Optional
import uuid
import hashlib
from datetime import datetime

class CertificateService:
    def __init__(self):
        self.certificates = {}

    async def create_certificate(self, certificate: CertificateCreate, user_id: str) -> Certificate:
        certificate_id = str(uuid.uuid4())
        certificate_hash = self._generate_certificate_hash(certificate_id, certificate.course_id, user_id)
        new_certificate = Certificate(
            id=certificate_id,
            hash=certificate_hash,
            **certificate.dict(),
            user_id=user_id,
            issue_date=datetime.now()
        )
        self.certificates[certificate_id] = new_certificate
        return new_certificate

    async def get_certificate(self, certificate_id: str) -> Optional[Certificate]:
        return self.certificates.get(certificate_id)

    async def list_certificates(self, user_id: str) -> List[Certificate]:
        return [cert for cert in self.certificates.values() if cert.user_id == user_id]

    async def revoke_certificate(self, certificate_id: str) -> None:
        if certificate_id in self.certificates:
            self.certificates[certificate_id].revoked = True

    def _generate_certificate_hash(self, certificate_id: str, course_id: str, user_id: str) -> str:
        data = f"{certificate_id}:{course_id}:{user_id}:{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()

    async def verify_certificate(self, certificate_id: str, certificate_hash: str) -> bool:
        certificate = self.certificates.get(certificate_id)
        if certificate and certificate.hash == certificate_hash and not certificate.revoked:
            return True
        return False

certificate_service = CertificateService()

def get_certificate_service() -> CertificateService:
    return certificate_service
