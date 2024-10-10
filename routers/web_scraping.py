from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from services.web_scraping_service import get_web_scraping_service, WebScrapingService
from typing import Dict, List
from pydantic import BaseModel, HttpUrl

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class ScrapeRequest(BaseModel):
    url: HttpUrl

class ScrapeResponse(BaseModel):
    content: str
    is_anomaly: bool

class ScrapingHistoryResponse(BaseModel):
    history: List[Dict[str, any]]

@router.post("/scrape", response_model=ScrapeResponse)
async def scrape_website(request: ScrapeRequest, token: str = Depends(oauth2_scheme), service: WebScrapingService = Depends(get_web_scraping_service)):
    content, is_anomaly = await service.scrape_website(str(request.url))
    return ScrapeResponse(content=content, is_anomaly=is_anomaly)

@router.get("/history", response_model=ScrapingHistoryResponse)
async def get_scraping_history(url: HttpUrl, token: str = Depends(oauth2_scheme), service: WebScrapingService = Depends(get_web_scraping_service)):
    history = service.get_scraping_history(str(url))
    return ScrapingHistoryResponse(history=history)
