from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
from services.advanced_search_service import AdvancedSearchService, get_advanced_search_service
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class ContentType(str, Enum):
    COURSE = "course"
    QUIZ = "quiz"
    ARTICLE = "article"
    VIDEO = "video"

class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class SearchFilter(BaseModel):
    content_type: Optional[ContentType] = None
    difficulty_level: Optional[DifficultyLevel] = None
    date_range: Optional[List[datetime]] = Field(None, description="Start and end date for filtering")
    tags: Optional[List[str]] = None
    min_rating: Optional[float] = Field(None, ge=0, le=5)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    filters: Optional[SearchFilter] = None
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)
    use_semantic_search: bool = Field(True, description="Whether to use semantic search or not")

class SearchResult(BaseModel):
    id: str
    title: str
    content_type: ContentType
    similarity_score: float
    snippet: str
    relevance_score: float
    difficulty_level: DifficultyLevel
    rating: float
    tags: List[str]

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_count: int
    facets: Dict[str, Dict[str, int]]
    query_expansion: Optional[str] = None

@router.post("/search", response_model=SearchResponse)
@cache(expire=300)  # Cache results for 5 minutes
async def advanced_search(
    search_request: SearchRequest,
    user: User = Depends(oauth2_scheme),
    text_embedding_service: TextEmbeddingService = Depends(get_text_embedding_service),
    advanced_search_service: AdvancedSearchService = Depends(get_advanced_search_service)
):
    try:
        # Input validation
        if not search_request.query.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")

        # Query expansion using AI/ML
        expanded_query = await advanced_search_service.expand_query(search_request.query)
        
        # Perform semantic search if requested
        if search_request.use_semantic_search:
            query_embedding = await text_embedding_service.get_embedding(expanded_query)
        else:
            query_embedding = None

        results = await advanced_search_service.search(
            expanded_query,
            search_request.filters.dict() if search_request.filters else None,
            search_request.limit,
            search_request.offset,
            query_embedding
        )

        total_count = await advanced_search_service.get_total_count(expanded_query, search_request.filters.dict() if search_request.filters else None)
        facets = await advanced_search_service.get_facets(results)

        search_results = [
            SearchResult(
                id=result['id'],
                title=result['title'],
                content_type=result['content_type'],
                similarity_score=result['similarity'],
                snippet=result['snippet'],
                relevance_score=result['relevance_score'],
                difficulty_level=result['difficulty_level'],
                rating=result['rating'],
                tags=result['tags']
            )
            for result in results
        ]

        return SearchResponse(
            results=search_results, 
            total_count=total_count, 
            facets=facets,
            query_expansion=expanded_query if expanded_query != search_request.query else None
        )
    except Exception as e:
        logger.error(f"Error in advanced search: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during the search process")

@router.get("/search/filters")
async def get_available_filters(user: User = Depends(oauth2_scheme)):
    return {
        "content_types": [ct.value for ct in ContentType],
        "difficulty_levels": [dl.value for dl in DifficultyLevel],
        "tags": await advanced_search_service.get_all_tags()
    }

@router.get("/search/popular_queries")
@cache(expire=3600)  # Cache results for 1 hour
async def get_popular_queries(
    limit: int = Query(10, ge=1, le=100),
    user: User = Depends(oauth2_scheme),
    advanced_search_service: AdvancedSearchService = Depends(get_advanced_search_service)
):
    try:
        popular_queries = await advanced_search_service.get_popular_queries(limit)
        return {"popular_queries": popular_queries}
    except Exception as e:
        logger.error(f"Error fetching popular queries: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching popular queries")

@router.post("/search/feedback")
async def submit_search_feedback(
    query: str,
    result_id: str,
    is_relevant: bool,
    user: User = Depends(oauth2_scheme),
    advanced_search_service: AdvancedSearchService = Depends(get_advanced_search_service)
):
    try:
        await advanced_search_service.submit_search_feedback(user.id, query, result_id, is_relevant)
        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Error submitting search feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while submitting search feedback")
