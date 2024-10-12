import logging
from typing import Dict, Any, List
from fastapi import HTTPException
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class SentimentAnalysisService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
        self.kmeans = KMeans(n_clusters=3, random_state=42)

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        try:
            # Use LLMOrchestrator for advanced sentiment analysis
            prompt = f"Perform a comprehensive sentiment analysis of the following text, including emotion detection, sarcasm detection, and context understanding. Provide a sentiment score between -1 (very negative) and 1 (very positive), an emotion label, and a detailed explanation:\n\n{text}"
            
            analysis = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an advanced AI specializing in sentiment analysis, emotion detection, and linguistic nuance understanding."},
                {"role": "user", "content": prompt}
            ], "high")

            # Parse the sentiment score, emotion label, and explanation from the analysis
            lines = analysis.strip().split("\n")
            sentiment_score = float(lines[0])
            emotion_label = lines[1]
            explanation = "\n".join(lines[2:])

            # Generate text embedding
            embedding = await self.text_embedding_service.get_embedding(text)

            # Perform topic modeling
            topics = self._perform_topic_modeling(text)

            # Perform text clustering
            cluster = self._perform_text_clustering(embedding)

            return {
                "sentiment_score": sentiment_score,
                "emotion_label": emotion_label,
                "explanation": explanation,
                "embedding": embedding,
                "topics": topics,
                "cluster": cluster
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred during sentiment analysis")

    def _perform_topic_modeling(self, text: str) -> List[str]:
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            lda_output = self.lda_model.fit_transform(tfidf_matrix)
            
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-5 - 1:-1]]
                topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
            
            return topics
        except Exception as e:
            logger.error(f"Error in topic modeling: {str(e)}")
            return []

    def _perform_text_clustering(self, embedding: List[float]) -> int:
        try:
            embedding_array = np.array(embedding).reshape(1, -1)
            cluster = self.kmeans.fit_predict(embedding_array)[0]
            return int(cluster)
        except Exception as e:
            logger.error(f"Error in text clustering: {str(e)}")
            return -1

    async def batch_analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        results = []
        for text in texts:
            result = await self.analyze_sentiment(text)
            results.append(result)
        return results

sentiment_analysis_service = SentimentAnalysisService(get_llm_orchestrator(), get_text_embedding_service())

def get_sentiment_analysis_service() -> SentimentAnalysisService:
    return sentiment_analysis_service
