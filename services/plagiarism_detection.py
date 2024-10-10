import os
from typing import Dict, List, Tuple
from services.llm_orchestrator import LLMOrchestrator

llm_orchestrator = LLMOrchestrator()

def compare_texts(original_text: str, submitted_text: str) -> Tuple[float, str]:
    """
    Compare the submitted text with the original text using LLMOrchestrator to detect plagiarism.
    
    :param original_text: The original text to compare against
    :param submitted_text: The text submitted by the user
    :return: A tuple containing the similarity score and explanation
    """
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

    response = llm_orchestrator.process_request([
        {"role": "system", "content": "You are a plagiarism detection system."},
        {"role": "user", "content": prompt}
    ], "medium")

    if response:
        result_dict = eval(response)  # Convert string to dictionary
        return result_dict["similarity_score"], result_dict["explanation"]
    else:
        return 0.0, "Error: Unable to process the request."

def detect_plagiarism(submitted_text: str, original_texts: List[str]) -> Dict[str, any]:
    """
    Detect plagiarism by comparing the submitted text with a list of original texts.
    
    :param submitted_text: The text submitted by the user
    :param original_texts: A list of original texts to compare against
    :return: A dictionary containing the plagiarism detection results
    """
    results = []
    for i, original_text in enumerate(original_texts):
        similarity_score, explanation = compare_texts(original_text, submitted_text)
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
