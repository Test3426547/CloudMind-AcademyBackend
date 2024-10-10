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
        response = self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an empathetic AI tutor."},
            {"role": "user", "content": prompt}
        ], "medium")
        
        ai_emotion = analyze_emotion(response)
        
        return {
            "user_emotion": user_emotion,
            "ai_response": response,
            "ai_emotion": ai_emotion
        }

    async def explain_concept(self, concept: str) -> Dict[str, str]:
        user_emotion = analyze_emotion(f"Explain {concept}")
        prompt = f"User emotion: {user_emotion}\nExplain the following concept in simple terms, considering the user's emotional state: {concept}"
        explanation = self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI tutor explaining concepts in simple terms."},
            {"role": "user", "content": prompt}
        ], "medium")
        
        return {
            "user_emotion": user_emotion,
            "explanation": explanation
        }

    async def get_collaboration_assistance(self, message: str, context: List[Dict[str, str]] = []) -> str:
        system_prompt = """
        You are an AI assistant in a collaborative learning environment. Your role is to provide helpful insights, 
        answer questions, and facilitate discussion among students. Please respond to the following message 
        in a way that encourages further collaboration and learning. Consider the context of previous messages, if provided.

        Guidelines for your response:
        1. Address the user's question or comment directly and accurately.
        2. Encourage further discussion by asking thought-provoking follow-up questions.
        3. Suggest potential areas for collaboration or group study related to the topic.
        4. Recommend relevant learning resources or activities when appropriate.
        5. Promote inclusive and respectful communication among participants.
        6. If there are misconceptions, gently correct them and provide accurate information.
        7. Highlight connections between different concepts or ideas mentioned in the discussion.
        8. Summarize key points if the discussion has been lengthy or complex.
        9. Encourage participants to share their own experiences or knowledge related to the topic.
        10. If the topic allows, suggest a small group activity or project that could enhance learning.

        Remember to maintain a supportive and engaging tone throughout your response.
        """

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context)
        messages.append({"role": "user", "content": message})

        response = self.llm_orchestrator.process_request(messages, "high")
        return response

    async def summarize_collaboration(self, messages: List[str]) -> str:
        prompt = """
        You are an AI assistant tasked with summarizing a collaborative learning session. 
        Please analyze the following conversation and provide a concise summary that includes:

        1. Main topics discussed
        2. Key insights or conclusions reached
        3. Any questions that remained unanswered or require further exploration
        4. Suggestions for future collaboration sessions
        5. Notable contributions from participants
        6. Areas where the group showed strong collaboration
        7. Potential action items or next steps for the group

        Conversation:
        {conversation}

        Please provide a summary in a clear, organized format that will be valuable for the participants to review and act upon.
        """
        formatted_prompt = prompt.format(conversation="\n".join(messages))
        summary = self.llm_orchestrator.process_request([
            {"role": "system", "content": formatted_prompt}
        ], "high")
        return summary

ai_tutor_service = AITutorService()

def get_ai_tutor_service() -> AITutorService:
    return ai_tutor_service
