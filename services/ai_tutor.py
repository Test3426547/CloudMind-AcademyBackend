from fastapi import Depends
from services.emotion_analysis import analyze_emotion
from services.llm_orchestrator import LLMOrchestrator
from typing import Dict, List

class AITutorService:
    def __init__(self):
        self.llm_orchestrator = LLMOrchestrator()

    async def chat_with_tutor(self, message: str) -> Dict[str, str]:
        user_emotion = analyze_emotion(message)
        prompt = f"User emotion: {user_emotion}\nUser message: {message}\nRespond as an empathetic AI tutor, addressing the user's emotional state:"
        response = self.llm_orchestrator.process_request([{"role": "user", "content": prompt}], "medium")
        
        ai_emotion = analyze_emotion(response)
        
        return {
            "user_emotion": user_emotion,
            "ai_response": response,
            "ai_emotion": ai_emotion
        }

    async def explain_concept(self, concept: str) -> Dict[str, str]:
        user_emotion = analyze_emotion(f"Explain {concept}")
        prompt = f"User emotion: {user_emotion}\nExplain the following concept in simple terms, considering the user's emotional state: {concept}"
        explanation = self.llm_orchestrator.process_request([{"role": "user", "content": prompt}], "medium")
        
        return {
            "user_emotion": user_emotion,
            "explanation": explanation
        }

    async def get_collaboration_assistance(self, message: str) -> str:
        prompt = f"""
        You are an AI assistant in a collaborative learning environment. Your role is to provide helpful insights, 
        answer questions, and facilitate discussion among students. Please respond to the following message 
        in a way that encourages further collaboration and learning:

        User message: {message}

        Provide a response that:
        1. Addresses the user's question or comment
        2. Encourages further discussion or exploration of the topic
        3. Suggests potential areas for collaboration or group study
        4. If appropriate, recommends relevant learning resources or activities
        5. Promotes inclusive and respectful communication among participants
        """
        response = self.llm_orchestrator.process_request([{"role": "system", "content": prompt}, {"role": "user", "content": message}], "high")
        return response

    async def summarize_collaboration(self, messages: List[str]) -> str:
        prompt = f"""
        You are an AI assistant tasked with summarizing a collaborative learning session. 
        Please analyze the following conversation and provide a concise summary that includes:

        1. Main topics discussed
        2. Key insights or conclusions reached
        3. Any questions that remained unanswered or require further exploration
        4. Suggestions for future collaboration sessions

        Conversation:
        {'\n'.join(messages)}

        Please provide a summary in a clear and organized format.
        """
        summary = self.llm_orchestrator.process_request([{"role": "system", "content": prompt}], "high")
        return summary

ai_tutor_service = AITutorService()

def get_ai_tutor_service() -> AITutorService:
    return ai_tutor_service
