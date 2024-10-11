import os
from typing import List, Dict, Any, AsyncGenerator
from openai import OpenAI, AsyncOpenAI
import asyncio
from functools import lru_cache
import logging
import time
import json
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenRouterException(Exception):
    """Custom exception for OpenRouter service"""
    pass

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

class Cache:
    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl

    def get(self, key: str) -> Any:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        self.cache[key] = (value, datetime.now())

class OpenRouterService:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://cloudmindacademy.com"  # Replace with your actual domain
            }
        )
        self.models_rate_limiter = RateLimiter(calls=50, period=60)
        self.completion_rate_limiter = RateLimiter(calls=10, period=60)
        self.cache = Cache(ttl=3600)  # 1 hour TTL

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.client.base_url}{endpoint}"
        headers = {**self.client.default_headers, **kwargs.get('headers', {})}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self.client.get_async_client() as async_client:
                    response = await async_client.request(method, url, headers=headers, **kwargs)
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: Error making request to OpenRouter API: {str(e)}")
                if attempt == max_retries - 1:
                    raise OpenRouterException(f"Max retries exceeded: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    @lru_cache(maxsize=100)
    async def get_available_models(self) -> List[Dict[str, Any]]:
        await self.models_rate_limiter.wait()
        cache_key = "available_models"
        cached_models = self.cache.get(cache_key)
        if cached_models:
            return cached_models

        try:
            response = await self._make_request("GET", "/models")
            models = [{"id": model["id"], "name": model["name"]} for model in response["data"]]
            self.cache.set(cache_key, models)
            return models
        except Exception as e:
            logger.error(f"Error fetching available models: {str(e)}")
            raise OpenRouterException(f"Failed to fetch available models: {str(e)}")

    async def generate_completion(self, model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        await self.completion_rate_limiter.wait()
        try:
            response = await self._make_request(
                "POST",
                "/chat/completions",
                json={"model": model, "messages": messages}
            )
            return {
                "model": response["model"],
                "content": response["choices"][0]["message"]["content"],
                "usage": response["usage"]
            }
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise OpenRouterException(f"Failed to generate completion: {str(e)}")

    async def generate_completion_gpt4(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return await self.generate_completion("openai/gpt-4", messages)

    async def generate_completion_claude3(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return await self.generate_completion("anthropic/claude-3-opus-20240229", messages)

    async def generate_completion_mistral(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return await self.generate_completion("mistralai/mistral-7b-instruct", messages)

    async def generate_completion_llama(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return await self.generate_completion("meta-llama/llama-2-70b-chat", messages)

    async def batch_generate_completions(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tasks = []
        for request in requests:
            model = request.get("model", "openai/gpt-4")
            messages = request.get("messages", [])
            tasks.append(self.generate_completion(model, messages))
        return await asyncio.gather(*tasks)

    async def get_model_details(self, model_id: str) -> Dict[str, Any]:
        cache_key = f"model_details_{model_id}"
        cached_details = self.cache.get(cache_key)
        if cached_details:
            return cached_details

        try:
            response = await self._make_request("GET", f"/models/{model_id}")
            self.cache.set(cache_key, response)
            return response
        except Exception as e:
            logger.error(f"Error fetching model details for {model_id}: {str(e)}")
            raise OpenRouterException(f"Failed to fetch model details for {model_id}: {str(e)}")

    async def get_usage_stats(self) -> Dict[str, Any]:
        try:
            response = await self._make_request("GET", "/usage")
            return response
        except Exception as e:
            logger.error(f"Error fetching usage stats: {str(e)}")
            raise OpenRouterException(f"Failed to fetch usage stats: {str(e)}")

    async def stream_completion(self, model: str, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        await self.completion_rate_limiter.wait()
        try:
            async with self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
            ) as stream:
                async for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error streaming completion: {str(e)}")
            raise OpenRouterException(f"Failed to stream completion: {str(e)}")

    async def cancel_request(self, request_id: str) -> Dict[str, Any]:
        try:
            response = await self._make_request("POST", f"/cancel/{request_id}")
            return response
        except Exception as e:
            logger.error(f"Error cancelling request {request_id}: {str(e)}")
            raise OpenRouterException(f"Failed to cancel request {request_id}: {str(e)}")

openrouter_service = OpenRouterService()

def get_openrouter_service() -> OpenRouterService:
    return openrouter_service
