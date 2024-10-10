import trafilatura
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime

class WebScrapingService:
    def __init__(self):
        self.scraped_data = {}

    async def scrape_website(self, url: str) -> Tuple[str, bool]:
        try:
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded)
            
            if text:
                word_count = len(text.split())
                timestamp = datetime.now().isoformat()
                
                if url not in self.scraped_data:
                    self.scraped_data[url] = []
                
                self.scraped_data[url].append((timestamp, word_count))
                
                is_anomaly = self.detect_anomaly(url, word_count)
                
                return text, is_anomaly
            else:
                return "No content extracted", False
        except Exception as e:
            return f"Error scraping website: {str(e)}", False

    def detect_anomaly(self, url: str, word_count: int) -> bool:
        if len(self.scraped_data[url]) < 5:
            return False
        
        historical_counts = [count for _, count in self.scraped_data[url][:-1]]
        mean = np.mean(historical_counts)
        std = np.std(historical_counts)
        
        z_score = (word_count - mean) / std if std > 0 else 0
        
        return abs(z_score) > 2  # Consider it an anomaly if z-score is beyond 2 standard deviations

    def get_scraping_history(self, url: str) -> List[Dict[str, any]]:
        if url not in self.scraped_data:
            return []
        
        return [{"timestamp": ts, "word_count": wc} for ts, wc in self.scraped_data[url]]

web_scraping_service = WebScrapingService()

def get_web_scraping_service() -> WebScrapingService:
    return web_scraping_service
