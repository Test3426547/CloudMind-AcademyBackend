from typing import List, Dict, Any, Optional
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
from functools import lru_cache
from fuzzywuzzy import fuzz
import json

class AdvancedSearchService:
    def __init__(self, text_embedding_service: TextEmbeddingService):
        self.text_embedding_service = text_embedding_service
        self.search_cache = {}

    @lru_cache(maxsize=100)
    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        cache_key = f"{query}:{json.dumps(filters)}:{limit}:{offset}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        query_embedding = await self.text_embedding_service.get_cached_embedding(query)
        results = await self.text_embedding_service.search_similar_content(query, limit=limit * 2)  # Fetch more results for filtering

        filtered_results = self._apply_filters(results, filters)
        ranked_results = self._rank_results(filtered_results, query)
        paginated_results = ranked_results[offset:offset + limit]

        self.search_cache[cache_key] = paginated_results
        return paginated_results

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
        popularity_score = result.get('popularity', 0) / 100  # Assuming a popularity field exists
        recency_score = result.get('recency', 0) / 100  # Assuming a recency field exists

        return (0.5 * content_relevance) + (0.3 * popularity_score) + (0.2 * recency_score)

    async def get_facets(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        facets = {}
        for result in results:
            for key, value in result.items():
                if key not in facets:
                    facets[key] = {}
                if value not in facets[key]:
                    facets[key][value] = 0
                facets[key][value] += 1
        return facets

advanced_search_service = AdvancedSearchService(get_text_embedding_service())

def get_advanced_search_service() -> AdvancedSearchService:
    return advanced_search_service
