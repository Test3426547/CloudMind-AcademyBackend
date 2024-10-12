import os
import logging
from openai import OpenAI
from typing import Dict, Any, Optional, List
import time
from functools import lru_cache
import asyncio

class LLMOrchestrator:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://cloudmindacademy.com"  # Replace with your actual domain
            }
        )
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model_performance = {}
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    @lru_cache(maxsize=100)
    def choose_model(self, task_complexity: str) -> str:
        if task_complexity == "high":
            return "anthropic/claude-3-opus"
        elif task_complexity == "medium":
            return "anthropic/claude-3-sonnet"
        else:
            return "anthropic/claude-3-haiku"

    async def process_request(self, messages: List[Dict[str, str]], task_complexity: str) -> Optional[str]:
        model = self.choose_model(task_complexity)
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Processing request with model: {model}")
                start_time = time.time()
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True
                )
                full_response = ""
                async for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        yield chunk.choices[0].delta.content
                end_time = time.time()
                self._update_model_performance(model, end_time - start_time)
                return full_response
            except Exception as e:
                self.logger.error(f"Error processing request (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error(f"Max retries reached. Falling back to simpler model.")
                    return await self._fallback_request(messages)
        return None

    async def _fallback_request(self, messages: List[Dict[str, str]]) -> Optional[str]:
        try:
            fallback_model = "anthropic/claude-3-haiku"
            self.logger.info(f"Using fallback model: {fallback_model}")
            response = await self.client.chat.completions.create(
                model=fallback_model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error in fallback request: {str(e)}")
            return None

    def _update_model_performance(self, model: str, response_time: float):
        if model not in self.model_performance:
            self.model_performance[model] = {"total_time": 0, "requests": 0}
        self.model_performance[model]["total_time"] += response_time
        self.model_performance[model]["requests"] += 1

    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        return {
            model: {
                "avg_response_time": data["total_time"] / data["requests"],
                "total_requests": data["requests"]
            }
            for model, data in self.model_performance.items()
        }

llm_orchestrator = LLMOrchestrator()

def get_llm_orchestrator() -> LLMOrchestrator:
    return llm_orchestrator
