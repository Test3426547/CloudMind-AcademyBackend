import os
import requests
from typing import Dict, Any, Optional

class LLMOrchestrator:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def send_request(self, model: str, messages: list, max_tokens: int = 1000) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        response = requests.post(url, json=payload, headers=self.headers)
        return response.json()

    def handle_response(self, response: Dict[str, Any]) -> Optional[str]:
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        return None

    def choose_model(self, task_complexity: str) -> str:
        if task_complexity == "high":
            return "openai/gpt-4o"
        elif task_complexity == "medium":
            return "anthropic/claude-2"
        else:
            return "google/palm-2-chat-bison"

    def process_request(self, messages: list, task_complexity: str) -> Optional[str]:
        model = self.choose_model(task_complexity)
        response = self.send_request(model, messages)
        return self.handle_response(response)

# Usage example:
# orchestrator = LLMOrchestrator()
# messages = [{"role": "user", "content": "What is the capital of France?"}]
# result = orchestrator.process_request(messages, "low")
# print(result)
