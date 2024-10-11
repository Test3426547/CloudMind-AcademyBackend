import os
from openai import OpenAI
from typing import Dict, Any

class V0DevService:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://cloudmindacademy.com"  # Replace with your actual domain
            }
        )

    async def generate_ui_component(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-haiku",
                messages=[
                    {"role": "system", "content": "You are a UI component generator. Generate React component code based on the user's prompt."},
                    {"role": "user", "content": prompt}
                ]
            )
            return {
                "status": "success",
                "component": response.choices[0].message.content
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

v0dev_service = V0DevService()

def get_v0dev_service() -> V0DevService:
    return v0dev_service
