import os
import logging
from openai import OpenAI
from typing import Dict, Any, Optional, List
import time
from functools import lru_cache
import asyncio
import numpy as np
from sklearn.preprocessing import StandardScaler

class LLMOrchestrator:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://cloudmindacademy.com"
            }
        )
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model_performance = {}
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self.scaler = StandardScaler()
        self.model_usage = {
            "anthropic/claude-3-opus": 0,
            "anthropic/claude-3-sonnet": 0,
            "anthropic/claude-3-haiku": 0
        }

    @lru_cache(maxsize=100)
    def choose_model(self, task_complexity: str, input_length: int) -> str:
        if task_complexity == "high" or input_length > 1000:
            return "anthropic/claude-3-opus"
        elif task_complexity == "medium" or input_length > 500:
            return "anthropic/claude-3-sonnet"
        else:
            return "anthropic/claude-3-haiku"

    async def process_request(self, messages: List[Dict[str, str]], task_complexity: str) -> Optional[str]:
        input_length = sum(len(msg['content']) for msg in messages)
        model = self.choose_model(task_complexity, input_length)
        
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
                self.model_usage[model] += 1
                return
            except Exception as e:
                self.logger.error(f"Error processing request (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error(f"Max retries reached. Falling back to simpler model.")
                    async for chunk in self._fallback_request(messages):
                        yield chunk
        return

    async def _fallback_request(self, messages: List[Dict[str, str]]):
        try:
            fallback_model = "anthropic/claude-3-haiku"
            self.logger.info(f"Using fallback model: {fallback_model}")
            response = await self.client.chat.completions.create(
                model=fallback_model,
                messages=messages,
                stream=True
            )
            self.model_usage[fallback_model] += 1
            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self.logger.error(f"Error in fallback request: {str(e)}")
            yield None

    def _update_model_performance(self, model: str, response_time: float):
        if model not in self.model_performance:
            self.model_performance[model] = {"total_time": 0, "requests": 0, "response_times": []}
        self.model_performance[model]["total_time"] += response_time
        self.model_performance[model]["requests"] += 1
        self.model_performance[model]["response_times"].append(response_time)

    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        performance = {}
        for model, data in self.model_performance.items():
            avg_response_time = data["total_time"] / data["requests"] if data["requests"] > 0 else 0
            response_times = np.array(data["response_times"])
            performance[model] = {
                "avg_response_time": avg_response_time,
                "total_requests": data["requests"],
                "std_dev_response_time": np.std(response_times) if len(response_times) > 1 else 0,
                "min_response_time": np.min(response_times) if len(response_times) > 0 else 0,
                "max_response_time": np.max(response_times) if len(response_times) > 0 else 0,
            }
        return performance

    def get_model_usage(self) -> Dict[str, int]:
        return self.model_usage

    async def adaptive_request(self, messages: List[Dict[str, str]], task_complexity: str):
        input_length = sum(len(msg['content']) for msg in messages)
        initial_model = self.choose_model(task_complexity, input_length)
        
        try:
            async for chunk in self.process_request(messages, task_complexity):
                if chunk:
                    yield chunk
            
            # If the initial model fails, try the next more powerful model
            models = ["anthropic/claude-3-haiku", "anthropic/claude-3-sonnet", "anthropic/claude-3-opus"]
            current_model_index = models.index(initial_model)
            
            for model in models[current_model_index + 1:]:
                self.logger.info(f"Attempting with more powerful model: {model}")
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True
                )
                self.model_usage[model] += 1
                async for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            
            self.logger.error("All models failed to process the request")
        except Exception as e:
            self.logger.error(f"Error in adaptive_request: {str(e)}")
            yield None

llm_orchestrator = LLMOrchestrator()

def get_llm_orchestrator() -> LLMOrchestrator:
    return llm_orchestrator
