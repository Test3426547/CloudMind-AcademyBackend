import trafilatura
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import asyncio
import aiohttp

class WebScrapingService:
    def __init__(self):
        self.providers = {
            "aws": "https://docs.aws.amazon.com/",
            "azure": "https://docs.microsoft.com/en-us/azure/",
            "gcp": "https://cloud.google.com/docs/"
        }

    async def scrape_documentation(self, provider: str, topic: Optional[str] = None) -> Dict[str, str]:
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")

        base_url = self.providers[provider]
        url = f"{base_url}{topic}" if topic else base_url

        try:
            downloaded = trafilatura.fetch_url(url)
            content = trafilatura.extract(downloaded, include_links=True, include_tables=True)

            if not content:
                return {"error": "No content found"}

            return {"content": content, "url": url}
        except Exception as e:
            return {"error": str(e)}

    async def search_documentation(self, provider: str, query: str) -> List[Dict[str, str]]:
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")

        base_url = self.providers[provider]
        search_url = f"{base_url}search?q={query}"

        try:
            response = requests.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            for result in soup.find_all('div', class_='search-result'):
                title = result.find('h3').text.strip()
                link = result.find('a')['href']
                snippet = result.find('p').text.strip()
                results.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet
                })

            return results[:5]  # Return top 5 results
        except Exception as e:
            return [{"error": str(e)}]

    async def scrape_multiple_pages(self, provider: str, topics: List[str]) -> List[Dict[str, str]]:
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")

        base_url = self.providers[provider]
        results = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            for topic in topics:
                url = f"{base_url}{topic}"
                tasks.append(self.fetch_and_extract(session, url))
            
            results = await asyncio.gather(*tasks)

        return results

    async def fetch_and_extract(self, session: aiohttp.ClientSession, url: str) -> Dict[str, str]:
        try:
            async with session.get(url) as response:
                html = await response.text()
                content = trafilatura.extract(html, include_links=True, include_tables=True)
                if not content:
                    return {"error": "No content found", "url": url}
                return {"content": content, "url": url}
        except Exception as e:
            return {"error": str(e), "url": url}

web_scraping_service = WebScrapingService()

def get_web_scraping_service() -> WebScrapingService:
    return web_scraping_service
