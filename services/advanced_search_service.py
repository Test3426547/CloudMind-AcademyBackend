import asyncio
from typing import List, Dict, Any, Optional
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
from functools import lru_cache
from fuzzywuzzy import fuzz
import json
import time
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class AdvancedSearchService:
    def __init__(self, text_embedding_service: TextEmbeddingService):
        self.text_embedding_service = text_embedding_service
        self.search_cache = {}
        self.cache_ttl = 3600  # 1 hour

    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 10, offset: int = 0, retries: int = 3) -> List[Dict[str, Any]]:
        try:
            self._validate_search_params(query, filters, limit, offset)
            
            cache_key = f"{query}:{json.dumps(filters)}:{limit}:{offset}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result

            for attempt in range(retries):
                try:
                    query_embedding = await self.text_embedding_service.get_cached_embedding(query)
                    results = await self.text_embedding_service.search_similar_content(query, limit=limit * 2)
                    break
                except Exception as e:
                    if attempt == retries - 1:
                        logger.error(f"Failed to perform search after {retries} attempts: {str(e)}")
                        raise HTTPException(status_code=500, detail="Search operation failed")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

            filtered_results = self._apply_filters(results, filters)
            ranked_results = self._rank_results(filtered_results, query)
            paginated_results = ranked_results[offset:offset + limit]

            self._add_to_cache(cache_key, paginated_results)
            return paginated_results
        except ValueError as ve:
            logger.warning(f"Invalid search parameters: {str(ve)}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Unexpected error in search: {str(e)}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred during the search")

    def _validate_search_params(self, query: str, filters: Optional[Dict[str, Any]], limit: int, offset: int):
        if not query or len(query.strip()) == 0:
            raise ValueError("Query cannot be empty")
        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("Offset cannot be negative")
        if filters:
            if not isinstance(filters, dict):
                raise ValueError("Filters must be a dictionary")
            for key, value in filters.items():
                if not isinstance(key, str):
                    raise ValueError("Filter keys must be strings")

    def _get_from_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        if key in self.search_cache:
            result, timestamp = self.search_cache[key]
            if time.time() - timestamp < self.cache_ttl:
                logger.info(f"Cache hit for key: {key}")
                return result
            else:
                del self.search_cache[key]
        return None

    def _add_to_cache(self, key: str, value: List[Dict[str, Any]]):
        self.search_cache[key] = (value, time.time())
        logger.info(f"Added to cache: {key}")

    def _apply_filters(self, results: List[Dict[str, Any]], filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not filters:
            return results

        filtered_results = []
        for result in results:
            if all(self._match_filter(result, key, value) for key, value in filters.items()):
                filtered_results.append(result)

        return filtered_results

    def _match_filter(self, result: Dict[str, Any], key: str, value: Any) -> bool:
        if key not in result:
            return False

        if isinstance(value, list):
            return any(self._fuzzy_match(str(result[key]), str(v)) for v in value)
        else:
            return self._fuzzy_match(str(result[key]), str(value))

    def _fuzzy_match(self, str1: str, str2: str, threshold: int = 80) -> bool:
        return fuzz.ratio(str1.lower(), str2.lower()) >= threshold

    def _rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        for result in results:
            result['relevance_score'] = self._calculate_relevance_score(result, query)

        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)

    def _calculate_relevance_score(self, result: Dict[str, Any], query: str) -> float:
        content_relevance = fuzz.ratio(query.lower(), result['content'].lower()) / 100
        title_relevance = fuzz.ratio(query.lower(), result.get('title', '').lower()) / 100
        popularity_score = min(result.get('popularity', 0) / 100, 1.0)
        recency_score = min(result.get('recency', 0) / 100, 1.0)

        # Adjusted weights for a more sophisticated ranking
        return (0.4 * content_relevance) + (0.3 * title_relevance) + (0.2 * popularity_score) + (0.1 * recency_score)

    async def get_facets(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        try:
            facets = {}
            for result in results:
                for key, value in result.items():
                    if key not in facets:
                        facets[key] = {}
                    if value not in facets[key]:
                        facets[key][value] = 0
                    facets[key][value] += 1
            return facets
        except Exception as e:
            logger.error(f"Error generating facets: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate facets")

advanced_search_service = AdvancedSearchService(get_text_embedding_service())

def get_advanced_search_service() -> AdvancedSearchService:
    return advanced_search_service
