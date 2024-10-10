import os
from openai import OpenAI
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class TextEmbeddingService:
    def __init__(self):
        self.model = "text-embedding-ada-002"

    async def get_embedding(self, text: str):
        response = client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    async def store_embedding(self, url: str, content: str, embedding):
        data, count = supabase.table("scraped_content").insert({
            "url": url,
            "content": content,
            "embedding": embedding
        }).execute()
        return data

    async def search_similar_content(self, query: str, limit: int = 5):
        query_embedding = await self.get_embedding(query)
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
