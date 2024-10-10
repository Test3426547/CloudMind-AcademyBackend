from transformers import pipeline
import numpy as np
from typing import List, Dict, Any
from services.llm_orchestrator import LLMOrchestrator

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
orchestrator = LLMOrchestrator()

def analyze_emotion(text: str) -> Dict[str, Any]:
    results = emotion_classifier(text)
    emotions = [{"emotion": result["label"], "score": float(result["score"])} for result in results[0]]
    emotions.sort(key=lambda x: x["score"], reverse=True)
    
    basic_analysis = {
        "primary_emotion": emotions[0]["emotion"],
        "primary_score": emotions[0]["score"],
        "secondary_emotions": emotions[1:3],
        "all_emotions": emotions,
        "emotion_distribution": {e["emotion"]: e["score"] for e in emotions}
    }
    
    # Use LLMOrchestrator for advanced analysis
    messages = [
        {"role": "system", "content": "You are an expert in emotion analysis. Provide a detailed interpretation of the emotional state based on the given data."},
        {"role": "user", "content": f"Analyze the following emotion data: {basic_analysis}"}
    ]
    advanced_analysis = orchestrator.process_request(messages, "medium")
    
    return {
        **basic_analysis,
        "advanced_analysis": advanced_analysis,
        "emotion_intensity": get_emotion_intensity(basic_analysis),
        "emotion_diversity": get_emotion_diversity(basic_analysis)
    }

def get_emotion_intensity(emotion_result: Dict[str, Any]) -> str:
    score = emotion_result["primary_score"]
    if score > 0.8:
        return "very high"
    elif score > 0.6:
        return "high"
    elif score > 0.4:
        return "moderate"
    elif score > 0.2:
        return "low"
    else:
        return "very low"

def get_emotion_diversity(emotion_result: Dict[str, Any]) -> str:
    scores = [e["score"] for e in emotion_result["all_emotions"]]
    entropy = -np.sum(np.array(scores) * np.log2(np.array(scores) + 1e-10))
    max_entropy = np.log2(len(scores))
    diversity = entropy / max_entropy
    
    if diversity > 0.8:
        return "very diverse"
    elif diversity > 0.6:
        return "diverse"
    elif diversity > 0.4:
        return "moderately diverse"
    elif diversity > 0.2:
        return "slightly diverse"
    else:
        return "not diverse"
