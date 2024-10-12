import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service

logger = logging.getLogger(__name__)

class CourseFeedbackService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.feedback_data = {}

    async def submit_feedback(self, user_id: str, course_id: str, rating: int, comment: str) -> Dict[str, Any]:
        try:
            sentiment = await self._analyze_sentiment(comment)
            category = await self._categorize_feedback(comment)
            embedding = await self.text_embedding_service.get_embedding(comment)

            feedback_entry = {
                "user_id": user_id,
                "course_id": course_id,
                "rating": rating,
                "comment": comment,
                "sentiment": sentiment,
                "category": category,
                "embedding": embedding
            }

            if course_id not in self.feedback_data:
                self.feedback_data[course_id] = []
            self.feedback_data[course_id].append(feedback_entry)

            improvement_suggestions = await self._generate_improvement_suggestions(course_id, category, sentiment)

            return {
                "message": "Feedback submitted successfully",
                "sentiment": sentiment,
                "category": category,
                "improvement_suggestions": improvement_suggestions
            }
        except Exception as e:
            logger.error(f"Error submitting feedback: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while submitting feedback")

    async def get_course_feedback(self, course_id: str) -> List[Dict[str, Any]]:
        try:
            if course_id not in self.feedback_data:
                return []
            return self.feedback_data[course_id]
        except Exception as e:
            logger.error(f"Error retrieving course feedback: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while retrieving course feedback")

    async def get_feedback_summary(self, course_id: str) -> Dict[str, Any]:
        try:
            if course_id not in self.feedback_data:
                return {"message": "No feedback available for this course"}

            feedback_list = self.feedback_data[course_id]
            total_ratings = sum(feedback["rating"] for feedback in feedback_list)
            avg_rating = total_ratings / len(feedback_list)

            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            for feedback in feedback_list:
                sentiment_counts[feedback["sentiment"]] += 1

            category_counts = {}
            for feedback in feedback_list:
                category = feedback["category"]
                category_counts[category] = category_counts.get(category, 0) + 1

            trending_topics = await self._identify_trending_topics(course_id)

            return {
                "average_rating": avg_rating,
                "total_feedback_count": len(feedback_list),
                "sentiment_distribution": sentiment_counts,
                "category_distribution": category_counts,
                "trending_topics": trending_topics
            }
        except Exception as e:
            logger.error(f"Error generating feedback summary: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while generating feedback summary")

    async def _analyze_sentiment(self, comment: str) -> str:
        prompt = f"Analyze the sentiment of the following course feedback. Classify it as 'positive', 'neutral', or 'negative':\n\n{comment}"
        sentiment = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are a sentiment analysis expert. Provide a single-word response: 'positive', 'neutral', or 'negative'."},
            {"role": "user", "content": prompt}
        ], "low")
        return sentiment.strip().lower()

    async def _categorize_feedback(self, comment: str) -> str:
        prompt = f"Categorize the following course feedback into one of these categories: 'Content', 'Instructor', 'Platform', 'Assignments', or 'Other':\n\n{comment}"
        category = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are a text classification expert. Provide a single-word response from the given categories."},
            {"role": "user", "content": prompt}
        ], "low")
        return category.strip()

    async def _generate_improvement_suggestions(self, course_id: str, category: str, sentiment: str) -> List[str]:
        if sentiment == "positive":
            return []

        prompt = f"Generate 3 specific improvement suggestions for a course based on the following information:\nFeedback Category: {category}\nSentiment: {sentiment}\n\nProvide concise, actionable suggestions."
        suggestions_text = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an educational improvement expert. Provide 3 numbered, concise suggestions."},
            {"role": "user", "content": prompt}
        ], "medium")

        return [suggestion.strip() for suggestion in suggestions_text.split("\n") if suggestion.strip()]

    async def _identify_trending_topics(self, course_id: str) -> List[str]:
        feedback_list = self.feedback_data[course_id]
        comments = [feedback["comment"] for feedback in feedback_list]

        prompt = f"Identify the top 3 trending topics from the following course feedback comments:\n\n{comments}\n\nProvide a short phrase for each trending topic."
        trending_topics = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are a topic analysis expert. Identify and summarize the top 3 trending topics from the given feedback comments."},
            {"role": "user", "content": prompt}
        ], "medium")

        return [topic.strip() for topic in trending_topics.split("\n") if topic.strip()][:3]

course_feedback_service = CourseFeedbackService(get_llm_orchestrator(), get_text_embedding_service())

def get_course_feedback_service() -> CourseFeedbackService:
    return course_feedback_service
