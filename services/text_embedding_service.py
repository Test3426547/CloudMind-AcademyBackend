import os
from typing import List, Dict, Any
import numpy as np
import json
import logging
import hashlib
from collections import Counter

logger = logging.getLogger(__name__)

class TextEmbeddingService:
    def __init__(self):
        self.cache = {}
        
    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    async def get_embedding(self, text: str) -> List[float]:
        cache_key = self._hash_text(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Simple word frequency-based embedding
        words = text.lower().split()
        word_freq = Counter(words)
        vocab = list(word_freq.keys())
        embedding = [word_freq[word] / len(words) for word in vocab]
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        self.cache[cache_key] = embedding
        return embedding
    
    async def get_tfidf_embedding(self, text: str) -> List[float]:
        cache_key = f"tfidf_{self._hash_text(text)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Simple TF-IDF implementation
        words = text.lower().split()
        word_freq = Counter(words)
        num_words = len(words)
        tfidf = []
        for word, count in word_freq.items():
            tf = count / num_words
            idf = np.log(1 + 1 / (1 + count))  # Simplified IDF
            tfidf.append(tf * idf)
        
        # Normalize the embedding
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf = [x / norm for x in tfidf]
        
        self.cache[cache_key] = tfidf
        return tfidf
    
    async def get_combined_embedding(self, text: str) -> List[float]:
        embedding = await self.get_embedding(text)
        tfidf_embedding = await self.get_tfidf_embedding(text)
        return embedding + tfidf_embedding
    
    async def summarize_text(self, text: str, max_length: int = 150) -> str:
        cache_key = f"summary_{self._hash_text(text)}_{max_length}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Simple extractive summarization
        sentences = text.split('.')
        word_freq = Counter(text.lower().split())
        sentence_scores = []
        for sentence in sentences:
            words = sentence.lower().split()
            score = sum(word_freq[word] for word in words) / len(words) if words else 0
            sentence_scores.append((sentence, score))
        
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        summary = '. '.join(sent for sent, _ in sentence_scores[:3])
        summary = summary[:max_length] + '...' if len(summary) > max_length else summary
        
        self.cache[cache_key] = summary
        return summary
    
    async def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        cache_key = f"keywords_{self._hash_text(text)}_{top_k}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Simple keyword extraction based on word frequency
        words = text.lower().split()
        word_freq = Counter(words)
        keywords = [word for word, _ in word_freq.most_common(top_k)]
        
        self.cache[cache_key] = keywords
        return keywords
    
    async def semantic_search(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = await self.get_combined_embedding(query)
        doc_embeddings = [await self.get_combined_embedding(doc) for doc in documents]
        
        similarities = [np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)) 
                        for doc_embedding in doc_embeddings]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            {"document": documents[i], "similarity": similarities[i]}
            for i in top_indices
        ]
        
        return results

text_embedding_service = TextEmbeddingService()

def get_text_embedding_service() -> TextEmbeddingService:
    return text_embedding_service
