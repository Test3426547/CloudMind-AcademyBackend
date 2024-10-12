import asyncio
from typing import Dict, List, Any
from fastapi import HTTPException
import logging
import random
from datetime import datetime, timedelta
import hashlib
import json

logger = logging.getLogger(__name__)

class SimulatedMLModel:
    def predict_anomaly(self, data):
        # Simulated anomaly detection
        return random.uniform(-1, 1)

    def forecast(self, data, steps):
        # Simulated forecasting
        return [sum(data) / len(data) + random.uniform(-10, 10) for _ in range(steps)]

class SimulatedNLPModel:
    def analyze(self, text):
        # Simulated NLP analysis
        sentiments = ["positive", "negative", "neutral"]
        return {
            "sentiment": random.choice(sentiments),
            "sentiment_score": random.uniform(0, 1),
            "summary": f"Summary of: {text[:50]}...",
            "entities": [{"type": "PERSON", "text": "John Doe"}, {"type": "ORG", "text": "CloudMind Academy"}]
        }

class CertificateService:
    def __init__(self):
        self.certificates = {}
        self.ml_model = SimulatedMLModel()
        self.nlp_model = SimulatedNLPModel()

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
            int(hashlib.md5(certificate_data.get('hash', '').encode()).hexdigest(), 16) % 1000
        ]
        return self.ml_model.predict_anomaly(features)

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
            daily_counts.append(count)
            current_date += timedelta(days=1)

        forecast_steps = 7  # Forecast for the next week
        trend_forecast = self.ml_model.forecast(daily_counts, forecast_steps)
        
        return {
            "historical_data": [
                {"date": (start_date + timedelta(days=i)).strftime("%Y-%m-%d"), "count": count}
                for i, count in enumerate(daily_counts)
            ],
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
            "sentiment_score": content_analysis.get('sentiment_score', 0.0),
            "content_summary": content_analysis.get('summary', 'No summary available'),
            "named_entities": content_analysis.get('entities', []),
            "anomaly_score": await self._detect_anomaly(certificate_id)
        }

    async def _simulate_blockchain_transaction(self, transaction_id: str) -> str:
        # Simulate blockchain transaction with random delay
        delay = random.uniform(0.5, 2.0)
        await asyncio.sleep(delay)
        return f"tx_{transaction_id}_{hashlib.md5(transaction_id.encode()).hexdigest()[:6]}"

certificate_service = CertificateService()

def get_certificate_service() -> CertificateService:
    return certificate_service
