import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
import trafilatura
import requests
from bs4 import BeautifulSoup
from cachetools import TTLCache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class WebScrapingService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.cache = TTLCache(maxsize=100, ttl=3600)  # Cache for 1 hour
        self.tfidf_vectorizer = TfidfVectorizer()

    async def scrape_website(self, url: str) -> Dict[str, Any]:
        try:
            if url in self.cache:
                logger.info(f"Returning cached content for URL: {url}")
                return self.cache[url]

            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                raise HTTPException(status_code=404, detail="Failed to fetch the URL")

            content = trafilatura.extract(downloaded)
            if content is None:
                raise HTTPException(status_code=500, detail="Failed to extract content from the webpage")

            summary = await self.summarize_content(content)
            keywords = await self.extract_keywords(content)
            sentiment = await self.analyze_sentiment(content)
            embedding = await self.text_embedding_service.get_embedding(content)

            result = {
                "url": url,
                "content": content,
                "summary": summary,
                "keywords": keywords,
                "sentiment": sentiment,
                "embedding": embedding
            }

            self.cache[url] = result
            return result
        except Exception as e:
            logger.error(f"Error scraping website: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An error occurred while scraping the website: {str(e)}")

    async def summarize_content(self, content: str) -> str:
        prompt = f"Summarize the following content in 3-5 sentences:\n\n{content[:1000]}..."
        summary = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI assistant that summarizes web content."},
            {"role": "user", "content": prompt}
        ], "medium")
        return summary.strip()

    async def extract_keywords(self, content: str, num_keywords: int = 5) -> List[str]:
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([content])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        sorted_indexes = np.argsort(tfidf_scores)[::-1]
        return [feature_names[i] for i in sorted_indexes[:num_keywords]]

    async def analyze_sentiment(self, content: str) -> Dict[str, Any]:
        prompt = f"Analyze the sentiment of the following content. Provide a sentiment score between -1 (very negative) and 1 (very positive), and a brief explanation:\n\n{content[:1000]}..."
        sentiment_analysis = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI assistant that performs sentiment analysis on web content."},
            {"role": "user", "content": prompt}
        ], "medium")
        
        lines = sentiment_analysis.strip().split('\n')
        sentiment_score = float(lines[0])
        explanation = '\n'.join(lines[1:])

        return {
            "score": sentiment_score,
            "explanation": explanation
        }

    async def search_similar_content(self, query: str, scraped_contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        query_embedding = await self.text_embedding_service.get_embedding(query)
        similarities = []

        for content in scraped_contents:
            similarity = cosine_similarity([query_embedding], [content['embedding']])[0][0]
            similarities.append((content, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [{"url": content["url"], "similarity": similarity} for content, similarity in similarities[:5]]

web_scraping_service = WebScrapingService(get_llm_orchestrator(), get_text_embedding_service())

def get_web_scraping_service() -> WebScrapingService:
    return web_scraping_service
