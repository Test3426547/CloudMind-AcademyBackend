from fastapi import Depends
from services.openai_client import send_openai_request
from services.emotion_analysis import analyze_emotion
from typing import Dict

class AITutorService:
    async def chat_with_tutor(self, message: str) -> Dict[str, str]:
        user_emotion = analyze_emotion(message)
        prompt = f"User emotion: {user_emotion}\nUser message: {message}\nRespond as an empathetic AI tutor, addressing the user's emotional state:"
        response = send_openai_request(prompt)
        
        ai_emotion = analyze_emotion(response)
        
        return {
            "user_emotion": user_emotion,
            "ai_response": response,
            "ai_emotion": ai_emotion
        }

    async def explain_concept(self, concept: str) -> Dict[str, str]:
        user_emotion = analyze_emotion(f"Explain {concept}")
        prompt = f"User emotion: {user_emotion}\nExplain the following concept in simple terms, considering the user's emotional state: {concept}"
        explanation = send_openai_request(prompt)
        
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
        """
        response = send_openai_request(prompt)
        return response

ai_tutor_service = AITutorService()

def get_ai_tutor_service() -> AITutorService:
    return ai_tutor_service
