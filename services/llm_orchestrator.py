import os
from openai import OpenAI
from typing import Dict, Any, Optional, List

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

    def choose_model(self, task_complexity: str) -> str:
        if task_complexity == "high":
            return "openai/gpt-4o"
        elif task_complexity == "medium":
            return "anthropic/claude-3-sonnet"
        else:
            return "openai/gpt-4o-mini"

    def process_request(self, messages: List[Dict[str, str]], task_complexity: str) -> Optional[str]:
        model = self.choose_model(task_complexity)
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return None

llm_orchestrator = LLMOrchestrator()

def get_llm_orchestrator() -> LLMOrchestrator:
    return llm_orchestrator
