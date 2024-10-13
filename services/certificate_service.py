import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
import os
from web3 import Web3
import hashlib
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class CertificateService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.ethereum_node_url = os.getenv("ETHEREUM_NODE_URL")
        self.contract_address = os.getenv("CERTIFICATE_CONTRACT_ADDRESS")
        self.private_key = os.getenv("ETHEREUM_PRIVATE_KEY")
        self.web3 = Web3(Web3.HTTPProvider(self.ethereum_node_url))
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.certificate_data = []

    async def create_certificate(self, user_id: str, course_id: str, completion_date: str) -> Dict[str, Any]:
        try:
            # Generate certificate content using AI
            certificate_content = await self._generate_certificate_content(user_id, course_id, completion_date)
            
            # Create hash of the certificate content
            content_hash = hashlib.sha256(certificate_content.encode()).hexdigest()
            
            # Store the hash on the blockchain
            tx_hash = await self._store_on_blockchain(content_hash)
            
            # Analyze certificate data for anomalies
            await self._analyze_certificate_data(user_id, course_id, completion_date)
            
            return {
                "user_id": user_id,
                "course_id": course_id,
                "completion_date": completion_date,
                "content": certificate_content,
                "blockchain_tx": tx_hash
            }
        except Exception as e:
            logger.error(f"Error creating certificate: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create certificate")

    async def verify_certificate(self, certificate_hash: str) -> bool:
        try:
            # Verify the certificate hash on the blockchain
            is_valid = await self._verify_on_blockchain(certificate_hash)
            return is_valid
        except Exception as e:
            logger.error(f"Error verifying certificate: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to verify certificate")

    async def _generate_certificate_content(self, user_id: str, course_id: str, completion_date: str) -> str:
        prompt = f"""Generate a certificate of completion for:
        User ID: {user_id}
        Course ID: {course_id}
        Completion Date: {completion_date}
        
        The certificate should include a formal congratulatory message, the course name, and a brief description of skills acquired.
        """
        
        content = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI assistant specialized in generating formal certificates."},
            {"role": "user", "content": prompt}
        ], "medium")
        
        return content.strip()

    async def _store_on_blockchain(self, content_hash: str) -> str:
        # Simplified blockchain interaction (replace with actual smart contract interaction)
        nonce = self.web3.eth.get_transaction_count(self.web3.eth.account.from_key(self.private_key).address)
        tx = {
            'nonce': nonce,
            'to': self.contract_address,
            'value': self.web3.to_wei(0, 'ether'),
            'gas': 2000000,
            'gasPrice': self.web3.eth.gas_price,
            'data': self.web3.to_hex(text=content_hash)
        }
        signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return self.web3.to_hex(tx_hash)

    async def _verify_on_blockchain(self, certificate_hash: str) -> bool:
        # Simplified blockchain verification (replace with actual smart contract interaction)
        # For demonstration, we'll assume the verification is successful if the hash exists on the blockchain
        tx = self.web3.eth.get_transaction(certificate_hash)
        return tx is not None

    async def _analyze_certificate_data(self, user_id: str, course_id: str, completion_date: str):
        # Convert data to numerical format for anomaly detection
        user_embedding = await self.text_embedding_service.get_embedding(user_id)
        course_embedding = await self.text_embedding_service.get_embedding(course_id)
        date_embedding = await self.text_embedding_service.get_embedding(completion_date)
        
        certificate_vector = np.concatenate([user_embedding, course_embedding, date_embedding])
        self.certificate_data.append(certificate_vector)
        
        if len(self.certificate_data) > 10:  # Minimum number of samples for meaningful anomaly detection
            X = np.array(self.certificate_data)
            self.isolation_forest.fit(X)
            anomaly_scores = self.isolation_forest.decision_function(X)
            if anomaly_scores[-1] < -0.5:  # Adjust threshold as needed
                logger.warning(f"Potential anomaly detected in certificate issuance: User {user_id}, Course {course_id}")

certificate_service = CertificateService(get_llm_orchestrator(), get_text_embedding_service())

def get_certificate_service() -> CertificateService:
    return certificate_service
