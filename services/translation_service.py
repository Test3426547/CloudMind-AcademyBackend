import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
import re
from collections import Counter

logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.language_codes = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'ru': 'Russian',
            'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic',
            'hi': 'Hindi', 'bn': 'Bengali', 'ur': 'Urdu', 'fa': 'Persian'
        }

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            prompt = f"""Translate the following text from {self.language_codes.get(source_lang, source_lang)} to {self.language_codes.get(target_lang, target_lang)}:

{text}

Translation:"""

            translation = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an expert translator with knowledge of multiple languages."},
                {"role": "user", "content": prompt}
            ], "medium")

            return translation.strip()
        except Exception as e:
            logger.error(f"Error in translation: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred during translation")

    async def detect_language(self, text: str) -> str:
        try:
            prompt = f"""Analyze the following text and determine its language. Respond with only the two-letter language code:

{text[:200]}  # Using first 200 characters for efficiency

Language code:"""

            detected_lang = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an expert in language detection. Respond only with the two-letter language code."},
                {"role": "user", "content": prompt}
            ], "low")

            return detected_lang.strip().lower()
        except Exception as e:
            logger.error(f"Error in language detection: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred during language detection")

    async def summarize_text(self, text: str, target_length: int = 100) -> str:
        try:
            prompt = f"""Summarize the following text in approximately {target_length} words:

{text}

Summary:"""

            summary = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an expert in text summarization."},
                {"role": "user", "content": prompt}
            ], "medium")

            return summary.strip()
        except Exception as e:
            logger.error(f"Error in text summarization: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred during text summarization")

    async def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        try:
            word_count = len(text.split())
            sentence_count = len(re.findall(r'\w+[.!?]', text))
            unique_words = len(set(text.lower().split()))
            avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
            
            complexity_score = (unique_words / word_count) * (avg_word_length / 5) * (word_count / sentence_count) if sentence_count > 0 else 0
            
            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "unique_words": unique_words,
                "avg_word_length": avg_word_length,
                "complexity_score": complexity_score
            }
        except Exception as e:
            logger.error(f"Error in text complexity analysis: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred during text complexity analysis")

    async def identify_key_phrases(self, text: str, num_phrases: int = 5) -> List[str]:
        try:
            words = text.lower().split()
            word_freq = Counter(words)
            
            # Remove common stop words
            stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            word_freq = {word: freq for word, freq in word_freq.items() if word not in stop_words}
            
            # Sort words by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Select top N words as key phrases
            key_phrases = [word for word, _ in sorted_words[:num_phrases]]
            
            return key_phrases
        except Exception as e:
            logger.error(f"Error in key phrase identification: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred during key phrase identification")

    async def sentiment_analysis(self, text: str) -> Dict[str, Any]:
        try:
            prompt = f"""Perform sentiment analysis on the following text. Provide a sentiment score between -1 (very negative) and 1 (very positive), and a brief explanation:

{text}

Sentiment analysis:"""

            analysis = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an expert in sentiment analysis."},
                {"role": "user", "content": prompt}
            ], "medium")

            lines = analysis.strip().split('\n')
            sentiment_score = float(lines[0])
            explanation = '\n'.join(lines[1:])

            return {
                "sentiment_score": sentiment_score,
                "explanation": explanation
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred during sentiment analysis")

translation_service = TranslationService(get_llm_orchestrator(), get_text_embedding_service())

def get_translation_service() -> TranslationService:
    return translation_service
