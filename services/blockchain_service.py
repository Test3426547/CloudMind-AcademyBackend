import asyncio
from typing import Dict, List, Any
from fastapi import HTTPException
import logging
import random
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class AdvancedMLModel:
    @staticmethod
    def predict(data):
        return np.random.random()

    @staticmethod
    def forecast(data, steps):
        return [np.random.random() for _ in range(steps)]

class AdvancedNLPModel:
    @staticmethod
    def analyze(text):
        sentiments = ["very negative", "negative", "neutral", "positive", "very positive"]
        return {
            "sentiment": random.choice(sentiments),
            "keywords": random.sample(["education", "blockchain", "certificate", "technology", "innovation", "security", "verification"], k=3),
            "entities": [{"type": "ORG", "text": "CloudMind Academy"}, {"type": "PERSON", "text": "John Doe"}],
            "summary": "This is a simulated summary of the certificate content."
        }

class BlockchainService:
    def __init__(self):
        self.certificates = {}
        self.anomaly_model = AdvancedMLModel()
        self.trend_model = AdvancedMLModel()
        self.nlp_model = AdvancedNLPModel()

    async def store_certificate(self, certificate_id: str, certificate_hash: str, certificate_content: str) -> str:
        self.certificates[certificate_id] = {
            'hash': certificate_hash,
            'revoked': False,
            'content': certificate_content,
            'created_at': datetime.now()
        }
        await self._analyze_certificate_content(certificate_id)
        return await self._simulate_blockchain_transaction(f"store_{certificate_id}")

    async def verify_certificate(self, certificate_id: str, certificate_hash: str) -> Dict[str, Any]:
        if certificate_id in self.certificates:
            is_valid = (self.certificates[certificate_id]['hash'] == certificate_hash and
                        not self.certificates[certificate_id]['revoked'])
            anomaly_score = await self._detect_anomaly(certificate_id)
            return {
                "is_valid": is_valid,
                "anomaly_score": anomaly_score,
                "warning": self._get_anomaly_warning(anomaly_score),
                "verification_time": await self._simulate_blockchain_transaction(f"verify_{certificate_id}")
            }
        return {"is_valid": False, "anomaly_score": 1.0, "warning": "Certificate not found"}

    async def revoke_certificate(self, certificate_id: str) -> str:
        if certificate_id in self.certificates:
            self.certificates[certificate_id]['revoked'] = True
        return await self._simulate_blockchain_transaction(f"revoke_{certificate_id}")

    async def _detect_anomaly(self, certificate_id: str) -> float:
        certificate_data = self.certificates.get(certificate_id, {})
        features = [
            len(certificate_data.get('content', '')),
            (datetime.now() - certificate_data.get('created_at', datetime.now())).days,
            hash(certificate_data.get('hash', '')) % 1000  # Simulating a hash-based feature
        ]
        return self.anomaly_model.predict(features)

    def _get_anomaly_warning(self, anomaly_score: float) -> str:
        if anomaly_score > 0.8:
            return "High risk of fraudulent certificate"
        elif anomaly_score > 0.6:
            return "Moderate risk, additional verification recommended"
        elif anomaly_score > 0.4:
            return "Low risk, but exercise caution"
        return "No significant anomalies detected"

    async def _analyze_certificate_content(self, certificate_id: str):
        content = self.certificates[certificate_id]['content']
        analysis = self.nlp_model.analyze(content)
        self.certificates[certificate_id]['content_analysis'] = analysis

    async def get_issuance_trend(self, days: int = 30) -> Dict[str, Any]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        daily_counts = []
        current_date = start_date
        while current_date <= end_date:
            count = sum(1 for cert in self.certificates.values() if start_date <= cert['created_at'] < current_date)
            daily_counts.append({"date": current_date.strftime("%Y-%m-%d"), "count": count})
            current_date += timedelta(days=1)

        forecast_steps = 7  # Forecast for the next week
        trend_forecast = self.trend_model.forecast(daily_counts, forecast_steps)
        
        return {
            "historical_data": daily_counts,
            "trend_forecast": [
                {"date": (end_date + timedelta(days=i+1)).strftime("%Y-%m-%d"), "predicted_count": count}
                for i, count in enumerate(trend_forecast)
            ]
        }

    async def get_certificate_insights(self, certificate_id: str) -> Dict[str, Any]:
        if certificate_id not in self.certificates:
            raise HTTPException(status_code=404, detail="Certificate not found")
        
        certificate = self.certificates[certificate_id]
        content_analysis = certificate.get('content_analysis', {})
        
        return {
            "certificate_id": certificate_id,
            "issuance_date": certificate['created_at'].strftime("%Y-%m-%d"),
            "revocation_status": "Revoked" if certificate['revoked'] else "Active",
            "content_sentiment": content_analysis.get('sentiment', 'Unknown'),
            "content_keywords": content_analysis.get('keywords', []),
            "named_entities": content_analysis.get('entities', []),
            "content_summary": content_analysis.get('summary', 'No summary available'),
            "anomaly_score": await self._detect_anomaly(certificate_id)
        }

    async def _simulate_blockchain_transaction(self, transaction_id: str) -> str:
        # Simulate blockchain transaction with random delay
        delay = random.uniform(0.5, 2.0)
        await asyncio.sleep(delay)
        return f"tx_{transaction_id}_{hash(transaction_id) % 1000000:06d}"

blockchain_service = BlockchainService()

def get_blockchain_service() -> BlockchainService:
    return blockchain_service
