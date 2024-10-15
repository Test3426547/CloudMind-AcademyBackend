import asyncio
from typing import List, Dict, Any, Optional
from fastapi import HTTPException
import logging
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from cachetools import TTLCache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AdvancedSearchService:
    def __init__(self, text_embedding_service: TextEmbeddingService, llm_orchestrator: LLMOrchestrator):
        # Initialize services and components
        self.text_embedding_service = text_embedding_service
        self.llm_orchestrator = llm_orchestrator
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # Cache for search results (1 hour TTL)
        self.tfidf_vectorizer = TfidfVectorizer()  # For text similarity calculations
        self.content_database = []  # Simulated content database
        self.popular_queries = TTLCache(maxsize=100, ttl=86400)  # Cache for popular queries (1 day TTL)
        self.feedback_database = []  # Simulated feedback database

    async def search(self, query: str, filters: Optional[Dict[str, Any]], limit: int, offset: int, query_embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        try:
            # Generate a unique cache key for this search request
            cache_key = f"{query}_{str(filters)}_{limit}_{offset}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Perform the search
            results = self._perform_search(query, filters, query_embedding)

            # Rank the results
            ranked_results = self._rank_results(results, query)

            # Apply pagination
            paginated_results = ranked_results[offset:offset+limit]

            # Cache the results
            self.cache[cache_key] = paginated_results
            return paginated_results
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred during the search process")

    def _perform_search(self, query: str, filters: Optional[Dict[str, Any]], query_embedding: Optional[List[float]]) -> List[Dict[str, Any]]:
        # Start with all results
        all_results = self.content_database.copy()

        # Apply filters if provided
        if filters:
            all_results = [
                result for result in all_results
                if (not filters.get('content_type') or result['content_type'] == filters['content_type']) and
                (not filters.get('difficulty_level') or result['difficulty_level'] == filters['difficulty_level']) and
                (not filters.get('date_range') or (filters['date_range'][0] <= result['date'] <= filters['date_range'][1])) and
                (not filters.get('tags') or all(tag in result['tags'] for tag in filters['tags'])) and
                (not filters.get('min_rating') or result['rating'] >= filters['min_rating'])
            ]

        # Perform semantic search if query_embedding is provided
        if query_embedding:
            all_results = sorted(
                all_results,
                key=lambda x: cosine_similarity([query_embedding], [x['embedding']])[0][0],
                reverse=True
            )
        else:
            # Fallback to TF-IDF based search
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([result['content'] for result in all_results])
            query_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
            all_results = [result for _, result in sorted(zip(similarities, all_results), key=lambda pair: pair[0], reverse=True)]

        return all_results

    def _rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        for result in results:
            # Calculate relevance score based on multiple factors
            text_similarity = self._calculate_text_similarity(query, result['content'])
            recency_score = self._calculate_recency_score(result['date'])
            popularity_score = self._calculate_popularity_score(result['views'], result['likes'])

            # Combine scores with weights
            result['relevance_score'] = (
                0.5 * text_similarity +
                0.3 * recency_score +
                0.2 * popularity_score
            )

        # Sort results by relevance score
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)

    def _calculate_text_similarity(self, query: str, content: str) -> float:
        # Calculate cosine similarity between query and content
        query_vec = self.tfidf_vectorizer.transform([query])
        content_vec = self.tfidf_vectorizer.transform([content])
        return cosine_similarity(query_vec, content_vec)[0][0]

    def _calculate_recency_score(self, date: datetime) -> float:
        # Calculate recency score (1.0 for new content, decreasing over time)
        days_old = (datetime.now() - date).days
        return max(0, 1 - (days_old / 365))  # Score decreases linearly over a year

    def _calculate_popularity_score(self, views: int, likes: int) -> float:
        # Calculate popularity score based on views and likes
        return min(1, (views + likes * 2) / 10000)  # Assume 10000 as a high number of views+likes

    async def get_total_count(self, query: str, filters: Optional[Dict[str, Any]]) -> int:
        # Get total count of results (ignoring pagination)
        results = self._perform_search(query, filters, None)
        return len(results)

    async def get_facets(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        # Generate facets (counts) for various attributes in the results
        facets = {
            "content_type": {},
            "difficulty_level": {},
            "tags": {}
        }

        for result in results:
            facets["content_type"][result["content_type"]] = facets["content_type"].get(result["content_type"], 0) + 1
            facets["difficulty_level"][result["difficulty_level"]] = facets["difficulty_level"].get(result["difficulty_level"], 0) + 1
            for tag in result["tags"]:
                facets["tags"][tag] = facets["tags"].get(tag, 0) + 1

        return facets

    async def expand_query(self, query: str) -> str:
        try:
            # Use LLM to expand the query for better search results
            expanded_query = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an AI assistant that expands search queries to improve search results."},
                {"role": "user", "content": f"Expand the following search query: {query}"}
            ], "low")
            return expanded_query.strip()
        except Exception as e:
            logger.error(f"Error in query expansion: {str(e)}")
            return query  # Return original query if expansion fails

    async def get_all_tags(self) -> List[str]:
        # Retrieve all unique tags from the content database
        all_tags = set()
        for content in self.content_database:
            all_tags.update(content['tags'])
        return list(all_tags)

    async def get_popular_queries(self, limit: int) -> List[str]:
        # Return cached popular queries if available
        if 'popular_queries' in self.popular_queries:
            return self.popular_queries['popular_queries'][:limit]

        # Calculate popular queries based on feedback
        query_counts = {}
        for feedback in self.feedback_database:
            query_counts[feedback['query']] = query_counts.get(feedback['query'], 0) + 1

        popular_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        result = [query for query, _ in popular_queries[:limit]]

        # Cache the result
        self.popular_queries['popular_queries'] = result

        return result

    async def submit_search_feedback(self, user_id: str, query: str, result_id: str, is_relevant: bool):
        # Store user feedback on search results
        feedback = {
            'user_id': user_id,
            'query': query,
            'result_id': result_id,
            'is_relevant': is_relevant,
            'timestamp': datetime.now()
        }
        self.feedback_database.append(feedback)

        # Clear popular queries cache to reflect new feedback
        self.popular_queries.clear()

# Create a single instance of AdvancedSearchService
advanced_search_service = AdvancedSearchService(get_text_embedding_service(), get_llm_orchestrator())

def get_advanced_search_service() -> AdvancedSearchService:
    # Function to retrieve the singleton instance of AdvancedSearchService
    return advanced_search_service