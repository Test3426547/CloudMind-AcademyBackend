from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.web_scraping_service import WebScrapingService, get_web_scraping_service
from typing import List, Dict, Any
from pydantic import BaseModel, HttpUrl
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class ScrapingRequest(BaseModel):
    url: HttpUrl

class SimilarContentRequest(BaseModel):
    query: str
    urls: List[HttpUrl]

@router.post("/scrape")
async def scrape_website(
    request: ScrapingRequest,
    user: User = Depends(oauth2_scheme),
    scraping_service: WebScrapingService = Depends(get_web_scraping_service),
):
    try:
        result = await scraping_service.scrape_website(str(request.url))
        logger.info(f"Website scraped successfully: {request.url}")
        return result
    except HTTPException as e:
        logger.warning(f"HTTP error in scrape_website: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in scrape_website: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while scraping the website")

@router.post("/search-similar")
async def search_similar_content(
    request: SimilarContentRequest,
    user: User = Depends(oauth2_scheme),
    scraping_service: WebScrapingService = Depends(get_web_scraping_service),
):
    try:
        scraped_contents = []
        for url in request.urls:
            content = await scraping_service.scrape_website(str(url))
            scraped_contents.append(content)

        similar_contents = await scraping_service.search_similar_content(request.query, scraped_contents)
        logger.info(f"Similar content search completed for query: {request.query}")
        return similar_contents
    except HTTPException as e:
        logger.warning(f"HTTP error in search_similar_content: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in search_similar_content: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while searching for similar content")
