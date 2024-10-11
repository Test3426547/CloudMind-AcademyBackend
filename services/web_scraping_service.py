import trafilatura
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
from playwright.async_api import async_playwright
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
import time
from ratelimit import limits, sleep_and_retry
import logging
from cachetools import TTLCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScrapingService:
    def __init__(self):
        self.providers = {
            "aws": "https://docs.aws.amazon.com/",
            "azure": "https://docs.microsoft.com/en-us/azure/",
            "gcp": "https://cloud.google.com/docs/"
        }
        self.text_embedding_service = get_text_embedding_service()
        
        # Initialize Reflection-Llama model
        self.tokenizer = AutoTokenizer.from_pretrained("mattshumer/Reflection-Llama-3.1-70B")
        self.model = AutoModelForCausalLM.from_pretrained("mattshumer/Reflection-Llama-3.1-70B")
        
        # Initialize Supabase client
        self.supabase: Client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

        # Initialize sentence transformer model
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Initialize cache
        self.cache = TTLCache(maxsize=100, ttl=3600)  # Cache for 1 hour

    @sleep_and_retry
    @limits(calls=5, period=60)  # Rate limit: 5 calls per minute
    async def scrape_documentation(self, provider: str, topic: Optional[str] = None) -> Dict[str, Any]:
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")

        base_url = self.providers[provider]
        url = f"{base_url}{topic}" if topic else base_url

        cache_key = f"{provider}_{topic}"
        if cache_key in self.cache:
            logger.info(f"Returning cached result for {cache_key}")
            return self.cache[cache_key]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to fetch URL: {url}. Status code: {response.status}")
                    html_content = await response.text()

            content = trafilatura.extract(html_content, include_links=True, include_tables=True)

            if not content:
                raise Exception("No content found")

            screenshot = await self.capture_screenshot(url)
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

            result = {
                "content": content,
                "url": url,
                "screenshot": screenshot,
                "embedding": embedding,
                "screenshot_valid": is_valid,
                "vectorized_content": vectorized_content,
                "chunked_content": chunked_content
            }

            self.cache[cache_key] = result
            return result
        except Exception as e:
            self.log_error("scrape_documentation", str(e), {"provider": provider, "topic": topic, "url": url})
            raise

    async def capture_screenshot(self, url: str) -> str:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle")
                screenshot = await page.screenshot(type='png')
                await browser.close()
                return base64.b64encode(screenshot).decode('utf-8')
        except Exception as e:
            self.log_error("capture_screenshot", str(e), {"url": url})
            raise

    def validate_screenshot(self, screenshot: str) -> bool:
        try:
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
        except Exception as e:
            self.log_error("validate_screenshot", str(e))
            return False

    def vectorize_content(self, content: str) -> List[float]:
        try:
            return self.sentence_model.encode(content).tolist()
        except Exception as e:
            self.log_error("vectorize_content", str(e))
            raise

    def chunk_content(self, content: str, chunk_size: int = 1000) -> List[str]:
        return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

    def log_data_transformation(self, operation: str, data: Dict[str, Any]):
        try:
            log_entry = {
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "data": json.dumps(data)
            }
            self.supabase.table("data_transformations").insert(log_entry).execute()
        except Exception as e:
            logger.error(f"Error logging data transformation: {str(e)}")

    def log_error(self, operation: str, error_message: str, context: Dict[str, Any] = None):
        try:
            log_entry = {
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "error_message": error_message,
                "context": json.dumps(context) if context else None
            }
            self.supabase.table("error_logs").insert(log_entry).execute()
        except Exception as e:
            logger.error(f"Error logging error: {str(e)}")

    async def batch_scrape_documentation(self, provider: str, topics: List[str]) -> List[Dict[str, Any]]:
        results = []
        for topic in topics:
            try:
                result = await self.scrape_documentation(provider, topic)
                results.append(result)
            except Exception as e:
                logger.error(f"Error scraping {provider} documentation for topic {topic}: {str(e)}")
                results.append({"error": str(e), "provider": provider, "topic": topic})
        return results

web_scraping_service = WebScrapingService()

def get_web_scraping_service() -> WebScrapingService:
    return web_scraping_service
