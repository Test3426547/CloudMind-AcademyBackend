import os
import asyncio
from openai import OpenAI
from typing import Dict, Any
import logging
from cachetools import TTLCache
import time
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.timestamps = []

    async def wait(self):
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < self.period]
        if len(self.timestamps) >= self.calls:
            sleep_time = self.period - (now - self.timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        self.timestamps.append(time.time())

class V0DevService:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://cloudmindacademy.com"  # Replace with your actual domain
            }
        )
        self.cache = TTLCache(maxsize=100, ttl=3600)  # Cache for 1 hour
        self.rate_limiter = RateLimiter(calls=5, period=60)  # 5 calls per minute

    def validate_prompt(self, prompt: str) -> bool:
        if not prompt or not isinstance(prompt, str):
            return False
        if len(prompt.strip()) < 10 or len(prompt) > 1000:
            return False
        return True

    async def generate_ui_component(self, prompt: str) -> Dict[str, Any]:
        if not self.validate_prompt(prompt):
            raise ValueError("Invalid prompt. Must be a non-empty string with 10-1000 characters.")

        cache_key = prompt
        if cache_key in self.cache:
            logger.info(f"Returning cached UI component for prompt: {prompt[:50]}...")
            return self.cache[cache_key]

        await self.rate_limiter.wait()

        try:
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-haiku",
                messages=[
                    {"role": "system", "content": "You are a UI component generator. Generate React component code based on the user's prompt."},
                    {"role": "user", "content": prompt}
                ]
            )
            result = {
                "status": "success",
                "component": response.choices[0].message.content
            }
            self.cache[cache_key] = result
            logger.info(f"Successfully generated UI component for prompt: {prompt[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Error generating UI component: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating UI component: {str(e)}")

    async def health_check(self) -> Dict[str, str]:
        try:
            # Perform a simple API call to check if the service is working
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-haiku",
                messages=[
                    {"role": "system", "content": "Respond with 'OK' if you receive this message."},
                    {"role": "user", "content": "Health check"}
                ]
            )
            if response.choices[0].message.content.strip().lower() == "ok":
                return {"status": "healthy"}
            else:
                return {"status": "unhealthy", "message": "Unexpected response from API"}
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "message": str(e)}

v0dev_service = V0DevService()

def get_v0dev_service() -> V0DevService:
    return v0dev_service
