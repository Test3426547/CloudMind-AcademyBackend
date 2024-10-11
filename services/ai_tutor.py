from fastapi import Depends, HTTPException
from services.emotion_analysis import analyze_emotion
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from typing import Dict, List
import logging
from functools import lru_cache
from ratelimit import limits, sleep_and_retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AITutorService:
    def __init__(self, llm_orchestrator: LLMOrchestrator):
        self.llm_orchestrator = llm_orchestrator

    def validate_input(self, message: str) -> bool:
        if not message or not isinstance(message, str):
            return False
        if len(message.strip()) < 5:  # Arbitrary minimum length
            return False
        return True

    @lru_cache(maxsize=100)
    @sleep_and_retry
    @limits(calls=10, period=60)  # Rate limit: 10 calls per minute
    async def chat_with_tutor(self, message: str) -> Dict[str, str]:
        if not self.validate_input(message):
            raise ValueError("Invalid input: Message must be a non-empty string with at least 5 characters.")

        try:
            user_emotion = analyze_emotion(message)
            prompt = f"User emotion: {user_emotion}\nUser message: {message}\nRespond as an empathetic AI tutor, addressing the user's emotional state:"
            response = self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an empathetic AI tutor."},
                {"role": "user", "content": prompt}
            ], "medium")
            
            if response is None:
                raise Exception("Failed to generate AI tutor response")
            
            ai_emotion = analyze_emotion(response)
            
            return {
                "user_emotion": user_emotion,
                "ai_response": response,
                "ai_emotion": ai_emotion
            }
        except Exception as e:
            logger.error(f"Error in chat_with_tutor: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in AI tutor chat: {str(e)}")

    @lru_cache(maxsize=100)
    @sleep_and_retry
    @limits(calls=10, period=60)  # Rate limit: 10 calls per minute
    async def explain_concept(self, concept: str) -> Dict[str, str]:
        if not self.validate_input(concept):
            raise ValueError("Invalid input: Concept must be a non-empty string with at least 5 characters.")

        try:
            user_emotion = analyze_emotion(f"Explain {concept}")
            prompt = f"User emotion: {user_emotion}\nExplain the following concept in simple terms, considering the user's emotional state: {concept}"
            explanation = self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an AI tutor explaining concepts in simple terms."},
                {"role": "user", "content": prompt}
            ], "medium")
            
            if explanation is None:
                raise Exception("Failed to generate concept explanation")
            
            return {
                "user_emotion": user_emotion,
                "explanation": explanation
            }
        except Exception as e:
            logger.error(f"Error in explain_concept: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in concept explanation: {str(e)}")

    @sleep_and_retry
    @limits(calls=5, period=60)  # Rate limit: 5 calls per minute
    async def get_collaboration_assistance(self, message: str, context: List[Dict[str, str]] = []) -> str:
        if not self.validate_input(message):
            raise ValueError("Invalid input: Message must be a non-empty string with at least 5 characters.")

        try:
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
            
            if response is None:
                raise Exception("Failed to generate collaboration assistance")
            
            return response
        except Exception as e:
            logger.error(f"Error in get_collaboration_assistance: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in collaboration assistance: {str(e)}")

    @sleep_and_retry
    @limits(calls=2, period=60)  # Rate limit: 2 calls per minute
    async def summarize_collaboration(self, messages: List[str]) -> str:
        if not messages or not all(self.validate_input(message) for message in messages):
            raise ValueError("Invalid input: Messages must be a non-empty list of valid strings.")

        try:
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
            
            if summary is None:
                raise Exception("Failed to generate collaboration summary")
            
            return summary
        except Exception as e:
            logger.error(f"Error in summarize_collaboration: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in collaboration summary: {str(e)}")

def get_ai_tutor_service(llm_orchestrator: LLMOrchestrator = Depends(get_llm_orchestrator)) -> AITutorService:
    return AITutorService(llm_orchestrator)
