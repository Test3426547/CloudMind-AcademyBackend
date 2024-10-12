import os
import asyncio
from openai import OpenAI
from typing import Dict, Any, List
import logging
from cachetools import TTLCache
import time
from fastapi import HTTPException
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
                "HTTP-Referer": "https://cloudmindacademy.com"
            }
        )
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # Increased cache size
        self.rate_limiter = RateLimiter(calls=5, period=60)
        self.vectorizer = TfidfVectorizer()
        self.feedback_data = []

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
            # Analyze prompt using NLP
            prompt_analysis = await self.analyze_prompt(prompt)

            # Generate component with AI-powered suggestions
            response = await self.client.chat.completions.create(
                model="anthropic/claude-3-haiku",
                messages=[
                    {"role": "system", "content": "You are an advanced UI component generator. Generate React component code based on the user's prompt and the provided prompt analysis."},
                    {"role": "user", "content": f"Prompt: {prompt}\nPrompt Analysis: {prompt_analysis}"}
                ]
            )
            
            result = {
                "status": "success",
                "component": response.choices[0].message.content,
                "prompt_analysis": prompt_analysis
            }
            self.cache[cache_key] = result
            logger.info(f"Successfully generated UI component for prompt: {prompt[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Error generating UI component: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating UI component: {str(e)}")

    async def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        try:
            # Perform TF-IDF vectorization
            tfidf_matrix = self.vectorizer.fit_transform([prompt])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top keywords
            top_keywords = [feature_names[i] for i in tfidf_matrix.toarray()[0].argsort()[-5:][::-1]]
            
            # Perform sentiment analysis using AI
            sentiment_response = await self.client.chat.completions.create(
                model="anthropic/claude-3-haiku",
                messages=[
                    {"role": "system", "content": "Perform sentiment analysis on the following prompt. Respond with a sentiment score between -1 (very negative) and 1 (very positive)."},
                    {"role": "user", "content": prompt}
                ]
            )
            sentiment_score = float(sentiment_response.choices[0].message.content)
            
            return {
                "top_keywords": top_keywords,
                "sentiment_score": sentiment_score
            }
        except Exception as e:
            logger.error(f"Error analyzing prompt: {str(e)}")
            return {"error": str(e)}

    async def health_check(self) -> Dict[str, str]:
        try:
            response = await self.client.chat.completions.create(
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

    async def provide_feedback(self, prompt: str, component: str, rating: int) -> Dict[str, str]:
        try:
            self.feedback_data.append({"prompt": prompt, "component": component, "rating": rating})
            if len(self.feedback_data) >= 100:
                await self.train_model()
            return {"status": "success", "message": "Feedback recorded successfully"}
        except Exception as e:
            logger.error(f"Error recording feedback: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def train_model(self) -> None:
        try:
            # In a real-world scenario, this would involve training or fine-tuning the AI model
            # For this example, we'll just log the training process
            logger.info(f"Training model with {len(self.feedback_data)} feedback entries")
            self.feedback_data = []  # Clear the feedback data after training
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")

v0dev_service = V0DevService()

def get_v0dev_service() -> V0DevService:
    return v0dev_service
