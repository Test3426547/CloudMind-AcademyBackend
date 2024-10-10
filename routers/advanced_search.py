from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
from services.advanced_search_service import AdvancedSearchService, get_advanced_search_service
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class SearchFilter(BaseModel):
    content_type: Optional[str] = None
    difficulty_level: Optional[str] = None
    date_range: Optional[List[str]] = None
    tags: Optional[List[str]] = None

class SearchRequest(BaseModel):
    query: str
    filters: Optional[SearchFilter] = None
    limit: int = 10
    offset: int = 0

class SearchResult(BaseModel):
    id: str
    title: str
    content_type: str
    similarity_score: float
    snippet: str
    relevance_score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_count: int
    facets: Dict[str, Dict[str, int]]

@router.post("/search", response_model=SearchResponse)
async def advanced_search(
    search_request: SearchRequest,
    user: User = Depends(oauth2_scheme),
    text_embedding_service: TextEmbeddingService = Depends(get_text_embedding_service),
    advanced_search_service: AdvancedSearchService = Depends(get_advanced_search_service)
):
    try:
        results = await advanced_search_service.search(
            search_request.query,
            search_request.filters.dict() if search_request.filters else None,
            search_request.limit,
            search_request.offset
        )

        total_count = len(results)
        facets = await advanced_search_service.get_facets(results)

        search_results = [
            SearchResult(
                id=result['id'],
                title=result['title'],
                content_type=result['content_type'],
                similarity_score=result['similarity'],
                snippet=result['snippet'],
                relevance_score=result['relevance_score']
            )
            for result in results
        ]

        return SearchResponse(results=search_results, total_count=total_count, facets=facets)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/filters")
async def get_available_filters(user: User = Depends(oauth2_scheme)):
    return {
        "content_types": ["course", "quiz", "article", "video"],
        "difficulty_levels": ["beginner", "intermediate", "advanced"],
        "tags": ["programming", "data science", "machine learning", "web development"]
    }
