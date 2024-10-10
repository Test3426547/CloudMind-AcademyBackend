import os
from typing import List, Dict, Any
from openai import OpenAI

class OpenRouterService:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://cloudmindacademy.com"  # Replace with your actual domain
            }
        )

    async def get_available_models(self) -> List[Dict[str, Any]]:
        response = self.client.models.list()
        return [{"id": model.id, "name": model.name} for model in response.data]

    async def generate_completion(self, model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages
        )
        return {
            "model": response.model,
            "content": response.choices[0].message.content,
            "usage": response.usage.dict() if response.usage else None
        }

    async def generate_completion_gpt4(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return await self.generate_completion("openai/gpt-4", messages)

    async def generate_completion_claude3(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return await self.generate_completion("anthropic/claude-3-opus-20240229", messages)

    async def generate_completion_mistral(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return await self.generate_completion("mistralai/mistral-7b-instruct", messages)

    async def generate_completion_llama(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return await self.generate_completion("meta-llama/llama-2-70b-chat", messages)

openrouter_service = OpenRouterService()

def get_openrouter_service() -> OpenRouterService:
    return openrouter_service
