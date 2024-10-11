from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.web_scraping_service import WebScrapingService, get_web_scraping_service
from typing import List, Optional
from pydantic import BaseModel, constr, Field
import logging
from fastapi_limiter.depends import RateLimiter

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

logger = logging.getLogger(__name__)

class ScrapingResult(BaseModel):
    content: str
    url: str
    screenshot: str
    embedding: List[float]
    screenshot_valid: bool
    vectorized_content: List[float]
    chunked_content: List[str]

class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str
    screenshot: str
    embedding: List[float]
    screenshot_valid: bool
    vectorized_content: List[float]
    chunked_content: List[str]

class MultipleScrapingRequest(BaseModel):
    provider: constr(regex='^(aws|azure|gcp)$')
    topics: List[constr(min_length=1, max_length=100)] = Field(..., max_items=10)

@router.get("/scrape/{provider}", response_model=ScrapingResult)
async def scrape_documentation(
    provider: constr(regex='^(aws|azure|gcp)$'),
    topic: Optional[constr(min_length=1, max_length=100)] = None,
    user: User = Depends(oauth2_scheme),
    scraping_service: WebScrapingService = Depends(get_web_scraping_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        result = await scraping_service.scrape_documentation(provider, topic)
        return ScrapingResult(**result)
    except ValueError as e:
        logger.warning(f"Invalid input for scrape_documentation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in scrape_documentation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while scraping: {str(e)}")

@router.get("/search/{provider}", response_model=List[SearchResult])
async def search_documentation(
    provider: constr(regex='^(aws|azure|gcp)$'),
    query: constr(min_length=1, max_length=100) = Query(..., description="Search query"),
    user: User = Depends(oauth2_scheme),
    scraping_service: WebScrapingService = Depends(get_web_scraping_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))
):
    try:
        results = await scraping_service.search_documentation(provider, query)
        return [SearchResult(**result) for result in results]
    except ValueError as e:
        logger.warning(f"Invalid input for search_documentation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in search_documentation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while searching: {str(e)}")

@router.post("/scrape-multiple", response_model=List[ScrapingResult])
async def scrape_multiple_pages(
    request: MultipleScrapingRequest,
    user: User = Depends(oauth2_scheme),
    scraping_service: WebScrapingService = Depends(get_web_scraping_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=2, seconds=60))
):
    try:
        results = await scraping_service.batch_scrape_documentation(request.provider, request.topics)
        return [ScrapingResult(**result) for result in results if "error" not in result]
    except ValueError as e:
        logger.warning(f"Invalid input for scrape_multiple_pages: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in scrape_multiple_pages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while scraping multiple pages: {str(e)}")
