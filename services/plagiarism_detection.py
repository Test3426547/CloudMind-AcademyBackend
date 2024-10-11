import os
from typing import Dict, List, Tuple
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from fastapi import Depends, HTTPException
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlagiarismDetectionService:
    def __init__(self, llm_orchestrator: LLMOrchestrator):
        self.llm_orchestrator = llm_orchestrator
        self.cache = {}

    def validate_input(self, text: str) -> bool:
        if not text or not isinstance(text, str):
            return False
        if len(text.strip()) < 10:  # Arbitrary minimum length
            return False
        return True

    @lru_cache(maxsize=100)
    def compare_texts(self, original_text: str, submitted_text: str) -> Tuple[float, str]:
        """
        Compare the submitted text with the original text using LLMOrchestrator to detect plagiarism.
        
        :param original_text: The original text to compare against
        :param submitted_text: The text submitted by the user
        :return: A tuple containing the similarity score and explanation
        """
        if not self.validate_input(original_text) or not self.validate_input(submitted_text):
            raise ValueError("Invalid input: Both texts must be non-empty strings with at least 10 characters.")

        prompt = f"""
        You are a plagiarism detection system. Compare the following two texts and determine if the submitted text is plagiarized from the original text. Provide a similarity score between 0 and 1, where 1 means identical and 0 means completely different. Also provide a brief explanation of your decision.

        Original text:
        {original_text}

        Submitted text:
        {submitted_text}

        Response format:
        {{
            "similarity_score": float,
            "explanation": string
        }}
        """

        try:
            response = self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are a plagiarism detection system."},
                {"role": "user", "content": prompt}
            ], "medium")

            if response is None:
                raise Exception("Error: Unable to process the request.")

            result_dict = eval(response)  # Convert string to dictionary
            return result_dict["similarity_score"], result_dict["explanation"]
        except Exception as e:
            logger.error(f"Error in compare_texts: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in plagiarism detection: {str(e)}")

    def detect_plagiarism(self, submitted_text: str, original_texts: List[str]) -> Dict[str, any]:
        """
        Detect plagiarism by comparing the submitted text with a list of original texts.
        
        :param submitted_text: The text submitted by the user
        :param original_texts: A list of original texts to compare against
        :return: A dictionary containing the plagiarism detection results
        """
        if not self.validate_input(submitted_text):
            raise ValueError("Invalid input: Submitted text must be a non-empty string with at least 10 characters.")
        if not original_texts or not all(self.validate_input(text) for text in original_texts):
            raise ValueError("Invalid input: Original texts must be a non-empty list of valid strings.")

        results = []
        for i, original_text in enumerate(original_texts):
            cache_key = (original_text, submitted_text)
            if cache_key in self.cache:
                similarity_score, explanation = self.cache[cache_key]
            else:
                similarity_score, explanation = self.compare_texts(original_text, submitted_text)
                self.cache[cache_key] = (similarity_score, explanation)

            results.append({
                "original_text_id": i,
                "similarity_score": similarity_score,
                "explanation": explanation
            })

        # Sort results by similarity score in descending order
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Determine overall plagiarism status
        max_similarity = results[0]["similarity_score"] if results else 0
        is_plagiarized = max_similarity > 0.7  # Threshold for plagiarism

        return {
            "is_plagiarized": is_plagiarized,
            "overall_similarity": max_similarity,
            "detailed_results": results
        }

    async def batch_plagiarism_check(self, submitted_texts: List[str], original_texts: List[str]) -> List[Dict[str, any]]:
        """
        Perform plagiarism checks on multiple submitted texts against multiple original texts.
        
        :param submitted_texts: A list of texts submitted by users
        :param original_texts: A list of original texts to compare against
        :return: A list of dictionaries containing plagiarism detection results for each submitted text
        """
        if not submitted_texts or not all(self.validate_input(text) for text in submitted_texts):
            raise ValueError("Invalid input: Submitted texts must be a non-empty list of valid strings.")
        if not original_texts or not all(self.validate_input(text) for text in original_texts):
            raise ValueError("Invalid input: Original texts must be a non-empty list of valid strings.")

        results = []
        for submitted_text in submitted_texts:
            result = self.detect_plagiarism(submitted_text, original_texts)
            results.append(result)

        return results

def get_plagiarism_detection_service(llm_orchestrator: LLMOrchestrator = Depends(get_llm_orchestrator)) -> PlagiarismDetectionService:
    return PlagiarismDetectionService(llm_orchestrator)
