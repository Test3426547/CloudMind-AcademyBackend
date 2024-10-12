import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

class PlagiarismDetectionService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.tfidf_vectorizer = TfidfVectorizer()
        self.cache = {}

    async def check_plagiarism(self, text: str, original_sources: List[str]) -> Dict[str, Any]:
        try:
            # Generate embeddings for the input text and original sources
            input_embedding = await self.text_embedding_service.get_embedding(text)
            source_embeddings = [await self.text_embedding_service.get_embedding(source) for source in original_sources]

            # Calculate cosine similarity between input and sources
            similarities = [cosine_similarity([input_embedding], [source_embedding])[0][0] for source_embedding in source_embeddings]

            # Use TF-IDF for more detailed text comparison
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text] + original_sources)
            tfidf_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

            # Combine embedding and TF-IDF similarities
            combined_similarities = [(s1 + s2) / 2 for s1, s2 in zip(similarities, tfidf_similarities)]

            # Determine plagiarism threshold
            threshold = await self._determine_threshold(text)

            # Identify plagiarized passages
            plagiarized_passages = await self._identify_plagiarized_passages(text, original_sources, combined_similarities, threshold)

            # Calculate overall plagiarism score
            plagiarism_score = max(combined_similarities) if combined_similarities else 0

            return {
                "plagiarism_detected": plagiarism_score > threshold,
                "plagiarism_score": plagiarism_score,
                "threshold": threshold,
                "plagiarized_passages": plagiarized_passages
            }
        except Exception as e:
            logger.error(f"Error in plagiarism detection: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred during plagiarism detection")

    async def _determine_threshold(self, text: str) -> float:
        # Use LLM to determine an appropriate threshold based on the text complexity
        prompt = f"Analyze the following text and suggest an appropriate plagiarism detection threshold (between 0 and 1) based on its complexity:\n\n{text[:500]}...\n\nSuggested threshold:"
        response = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI assistant specialized in plagiarism detection."},
            {"role": "user", "content": prompt}
        ], "medium")
        try:
            threshold = float(response.strip())
            return max(0.3, min(0.8, threshold))  # Ensure threshold is between 0.3 and 0.8
        except ValueError:
            return 0.5  # Default threshold if parsing fails

    async def _identify_plagiarized_passages(self, text: str, original_sources: List[str], similarities: List[float], threshold: float) -> List[Dict[str, Any]]:
        plagiarized_passages = []
        sentences = text.split('.')
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 0:
                sentence_embedding = await self.text_embedding_service.get_embedding(sentence)
                for j, source in enumerate(original_sources):
                    if similarities[j] > threshold:
                        source_sentences = source.split('.')
                        for source_sentence in source_sentences:
                            if len(source_sentence.strip()) > 0:
                                source_sentence_embedding = await self.text_embedding_service.get_embedding(source_sentence)
                                similarity = cosine_similarity([sentence_embedding], [source_sentence_embedding])[0][0]
                                if similarity > threshold:
                                    plagiarized_passages.append({
                                        "text": sentence.strip(),
                                        "source": j,
                                        "similarity": similarity
                                    })
        return plagiarized_passages

    async def batch_check_plagiarism(self, texts: List[str], original_sources: List[str]) -> List[Dict[str, Any]]:
        results = []
        for text in texts:
            result = await self.check_plagiarism(text, original_sources)
            results.append(result)
        return results

plagiarism_detection_service = PlagiarismDetectionService(get_llm_orchestrator(), get_text_embedding_service())

def get_plagiarism_detection_service() -> PlagiarismDetectionService:
    return plagiarism_detection_service
