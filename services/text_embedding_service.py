import os
from openai import OpenAI
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY
from typing import List, Dict, Any
from functools import lru_cache

client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(str(SUPABASE_URL), str(SUPABASE_KEY))

class TextEmbeddingService:
    def __init__(self):
        self.model = "text-embedding-ada-002"

    async def get_embedding(self, text: str) -> List[float]:
        response = client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    @lru_cache(maxsize=1000)
    async def get_cached_embedding(self, text: str) -> List[float]:
        return await self.get_embedding(text)

    async def store_embedding(self, url: str, content: str, embedding: List[float]):
        data, count = supabase.table("scraped_content").insert({
            "url": url,
            "content": content,
            "embedding": embedding
        }).execute()
        return data

    async def search_similar_content(self, query: str, limit: int = 5):
        query_embedding = await self.get_cached_embedding(query)
        data, count = supabase.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,
                'match_threshold': 0.5,
                'match_count': limit
            }
        ).execute()
        return data

text_embedding_service = TextEmbeddingService()

def get_text_embedding_service() -> TextEmbeddingService:
    return text_embedding_service
