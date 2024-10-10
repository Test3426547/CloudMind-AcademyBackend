from models.certificate import Certificate, CertificateCreate
from typing import List, Optional
import uuid
import hashlib
from datetime import datetime
from services.blockchain_service import get_blockchain_service

class CertificateService:
    def __init__(self):
        self.certificates = {}
        self.blockchain_service = get_blockchain_service()

    async def create_certificate(self, certificate: CertificateCreate, user_id: str) -> Certificate:
        certificate_id = str(uuid.uuid4())
        certificate_hash = self._generate_certificate_hash(certificate_id, certificate.course_id, user_id)
        new_certificate = Certificate(
            id=certificate_id,
            hash=certificate_hash,
            user_id=user_id,
            **certificate.dict()
        )
        self.certificates[certificate_id] = new_certificate
        
        # Store certificate on blockchain
        tx_hash = await self.blockchain_service.store_certificate(certificate_id, certificate_hash)
        print(f"Certificate stored on blockchain. Transaction hash: {tx_hash}")
        
        return new_certificate

    async def get_certificate(self, certificate_id: str) -> Optional[Certificate]:
        return self.certificates.get(certificate_id)

    async def list_certificates(self, user_id: str) -> List[Certificate]:
        return [cert for cert in self.certificates.values() if cert.user_id == user_id]

    async def revoke_certificate(self, certificate_id: str) -> None:
        if certificate_id in self.certificates:
            self.certificates[certificate_id].revoked = True
            tx_hash = await self.blockchain_service.revoke_certificate(certificate_id)
            print(f"Certificate revoked on blockchain. Transaction hash: {tx_hash}")

    def _generate_certificate_hash(self, certificate_id: str, course_id: str, user_id: str) -> str:
        data = f"{certificate_id}:{course_id}:{user_id}:{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()

    async def verify_certificate(self, certificate_id: str, certificate_hash: str) -> bool:
        certificate = self.certificates.get(certificate_id)
        if certificate and certificate.hash == certificate_hash and not certificate.revoked:
            # Verify on blockchain
            return await self.blockchain_service.verify_certificate(certificate_id, certificate_hash)
        return False

certificate_service = CertificateService()

def get_certificate_service() -> CertificateService:
    return certificate_service
