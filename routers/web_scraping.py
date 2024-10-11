from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.web_scraping_service import WebScrapingService, get_web_scraping_service
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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
    provider: str
    topics: List[str]

@router.get("/scrape/{provider}", response_model=ScrapingResult)
async def scrape_documentation(
    provider: str,
    topic: Optional[str] = None,
    user: User = Depends(oauth2_scheme),
    scraping_service: WebScrapingService = Depends(get_web_scraping_service)
):
    try:
        result = await scraping_service.scrape_documentation(provider, topic)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return ScrapingResult(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/search/{provider}", response_model=List[SearchResult])
async def search_documentation(
    provider: str,
    query: str,
    user: User = Depends(oauth2_scheme),
    scraping_service: WebScrapingService = Depends(get_web_scraping_service)
):
    try:
        results = await scraping_service.search_documentation(provider, query)
        if results and "error" in results[0]:
            raise HTTPException(status_code=400, detail=results[0]["error"])
        return [SearchResult(**result) for result in results]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/scrape-multiple", response_model=List[ScrapingResult])
async def scrape_multiple_pages(
    request: MultipleScrapingRequest,
    user: User = Depends(oauth2_scheme),
    scraping_service: WebScrapingService = Depends(get_web_scraping_service)
):
    try:
        results = await scraping_service.scrape_multiple_pages(request.provider, request.topics)
        return [ScrapingResult(**result) for result in results if "error" not in result]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
