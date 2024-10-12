from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)

class DocumentInput(BaseModel):
    documents: List[str] = Field(..., min_items=1, max_items=100)

class SemanticSearchInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    documents: List[str] = Field(..., min_items=1, max_items=100)

@router.post("/text-embedding")
async def get_text_embedding(
    text_input: TextInput,
    user: User = Depends(oauth2_scheme),
    embedding_service: TextEmbeddingService = Depends(get_text_embedding_service),
):
    try:
        embedding = await embedding_service.get_combined_embedding(text_input.text)
        logger.info(f"Generated text embedding for user {user.id}")
        return {"embedding": embedding}
    except Exception as e:
        logger.error(f"Error generating text embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the text embedding")

@router.post("/summarize")
async def summarize_text(
    text_input: TextInput,
    user: User = Depends(oauth2_scheme),
    embedding_service: TextEmbeddingService = Depends(get_text_embedding_service),
):
    try:
        summary = await embedding_service.summarize_text(text_input.text)
        logger.info(f"Generated text summary for user {user.id}")
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Error generating text summary: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the text summary")

@router.post("/extract-keywords")
async def extract_keywords(
    text_input: TextInput,
    user: User = Depends(oauth2_scheme),
    embedding_service: TextEmbeddingService = Depends(get_text_embedding_service),
):
    try:
        keywords = await embedding_service.extract_keywords(text_input.text)
        logger.info(f"Extracted keywords for user {user.id}")
        return {"keywords": keywords}
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while extracting keywords")

@router.post("/semantic-search")
async def semantic_search(
    search_input: SemanticSearchInput,
    user: User = Depends(oauth2_scheme),
    embedding_service: TextEmbeddingService = Depends(get_text_embedding_service),
):
    try:
        results = await embedding_service.semantic_search(search_input.query, search_input.documents)
        logger.info(f"Performed semantic search for user {user.id}")
        return {"results": results}
    except Exception as e:
        logger.error(f"Error performing semantic search: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while performing semantic search")
