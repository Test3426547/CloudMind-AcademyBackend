import trafilatura
import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup

class WebScrapingService:
    @staticmethod
    async def scrape_url(url: str) -> str:
        downloaded = trafilatura.fetch_url(url)
        return trafilatura.extract(downloaded)

    @staticmethod
    async def scrape_microsoft_docs(base_url: str) -> List[Dict[str, Any]]:
        content = []
        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', class_='card'):
            doc_url = f"https://learn.microsoft.com{link['href']}"
            doc_content = await WebScrapingService.scrape_url(doc_url)
            content.append({
                "url": doc_url,
                "content": doc_content
            })
        return content

    @staticmethod
    async def scrape_aws_docs(base_url: str) -> List[Dict[str, Any]]:
        content = []
        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', class_='lb-txt-bold'):
            doc_url = f"https://docs.aws.amazon.com{link['href']}"
            doc_content = await WebScrapingService.scrape_url(doc_url)
            content.append({
                "url": doc_url,
                "content": doc_content
            })
        return content

    @staticmethod
    async def scrape_gcp_docs(base_url: str) -> List[Dict[str, Any]]:
        content = []
        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', class_='cloud-body-text'):
            doc_url = f"https://cloud.google.com{link['href']}"
            doc_content = await WebScrapingService.scrape_url(doc_url)
            content.append({
                "url": doc_url,
                "content": doc_content
            })
        return content

web_scraping_service = WebScrapingService()

def get_web_scraping_service() -> WebScrapingService:
    return web_scraping_service
