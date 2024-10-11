import trafilatura
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import io
import base64
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from supabase import create_client, Client
import os
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
import cv2
import numpy as np

class WebScrapingService:
    def __init__(self):
        self.providers = {
            "aws": "https://docs.aws.amazon.com/",
            "azure": "https://docs.microsoft.com/en-us/azure/",
            "gcp": "https://cloud.google.com/docs/"
        }
        self.text_embedding_service = get_text_embedding_service()
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=chrome_options)
        
        # Initialize Reflection-Llama model
        self.tokenizer = AutoTokenizer.from_pretrained("mattshumer/Reflection-Llama-3.1-70B")
        self.model = AutoModelForCausalLM.from_pretrained("mattshumer/Reflection-Llama-3.1-70B")
        
        # Initialize Supabase client
        self.supabase: Client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

        # Initialize sentence transformer model
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    async def scrape_documentation(self, provider: str, topic: Optional[str] = None) -> Dict[str, Any]:
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")

        base_url = self.providers[provider]
        url = f"{base_url}{topic}" if topic else base_url

        try:
            downloaded = trafilatura.fetch_url(url)
            content = trafilatura.extract(downloaded, include_links=True, include_tables=True)

            if not content:
                return {"error": "No content found"}

            screenshot = self.capture_screenshot(url)
            embedding = await self.text_embedding_service.get_embedding(content)

            # Validate screenshot
            is_valid = self.validate_screenshot(screenshot)

            # Vectorize content
            vectorized_content = self.vectorize_content(content)

            # Chunk content
            chunked_content = self.chunk_content(content)

            # Log data transformation in Supabase
            self.log_data_transformation("scrape_documentation", {
                "provider": provider,
                "topic": topic,
                "url": url,
                "content_length": len(content),
                "screenshot_valid": is_valid,
                "num_chunks": len(chunked_content)
            })

            return {
                "content": content,
                "url": url,
                "screenshot": screenshot,
                "embedding": embedding,
                "screenshot_valid": is_valid,
                "vectorized_content": vectorized_content,
                "chunked_content": chunked_content
            }
        except Exception as e:
            return {"error": str(e)}

    def validate_screenshot(self, screenshot: str) -> bool:
        # Decode base64 screenshot
        screenshot_bytes = base64.b64decode(screenshot)
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Check if the image is not empty
        if img is None or img.size == 0:
            return False

        # Check image dimensions
        height, width, _ = img.shape
        if height < 100 or width < 100:
            return False

        # Check for blank or solid color images
        if len(np.unique(img)) < 10:
            return False

        return True

    def vectorize_content(self, content: str) -> List[float]:
        return self.sentence_model.encode(content).tolist()

    def chunk_content(self, content: str, chunk_size: int = 1000) -> List[str]:
        return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

    # ... (keep other methods unchanged)

    def log_data_transformation(self, operation: str, data: Dict[str, Any]):
        log_entry = {
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "data": json.dumps(data)
        }
        self.supabase.table("data_transformations").insert(log_entry).execute()

    def __del__(self):
        if hasattr(self, 'driver'):
            self.driver.quit()

web_scraping_service = WebScrapingService()

def get_web_scraping_service() -> WebScrapingService:
    return web_scraping_service
