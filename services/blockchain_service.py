class BlockchainService:
    def __init__(self):
        self.certificates = {}

    async def store_certificate(self, certificate_id: str, certificate_hash: str) -> str:
        self.certificates[certificate_id] = {
            'hash': certificate_hash,
            'revoked': False
        }
        return f"mock_transaction_hash_{certificate_id}"

    async def verify_certificate(self, certificate_id: str, certificate_hash: str) -> bool:
        if certificate_id in self.certificates:
            return (self.certificates[certificate_id]['hash'] == certificate_hash and
                    not self.certificates[certificate_id]['revoked'])
        return False

    async def revoke_certificate(self, certificate_id: str) -> str:
        if certificate_id in self.certificates:
            self.certificates[certificate_id]['revoked'] = True
        return f"mock_revocation_hash_{certificate_id}"

blockchain_service = BlockchainService()

def get_blockchain_service() -> BlockchainService:
    return blockchain_service
