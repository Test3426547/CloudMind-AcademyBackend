from typing import List, Dict, Any
from datetime import datetime
import uuid
from statistics import mean

class CourseFeedbackService:
    def __init__(self):
        self.feedbacks = {}

    async def create_feedback(self, user_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        feedback_id = str(uuid.uuid4())
        new_feedback = {
            "id": feedback_id,
            "user_id": user_id,
            "course_id": feedback["course_id"],
            "rating": feedback["rating"],
            "comment": feedback["comment"],
            "created_at": datetime.now().isoformat()
        }
        self.feedbacks[feedback_id] = new_feedback
        return new_feedback

    async def get_course_feedback(self, course_id: str) -> List[Dict[str, Any]]:
        return [feedback for feedback in self.feedbacks.values() if feedback["course_id"] == course_id]

    async def get_course_average_rating(self, course_id: str) -> float:
        course_feedbacks = await self.get_course_feedback(course_id)
        if not course_feedbacks:
            return 0.0
        ratings = [feedback["rating"] for feedback in course_feedbacks]
        return round(mean(ratings), 2)

    async def get_user_feedback(self, user_id: str, course_id: str) -> Dict[str, Any]:
        user_feedback = [feedback for feedback in self.feedbacks.values() if feedback["user_id"] == user_id and feedback["course_id"] == course_id]
        return user_feedback[0] if user_feedback else None

    async def update_feedback(self, feedback_id: str, updated_data: Dict[str, Any]) -> Dict[str, Any]:
        if feedback_id not in self.feedbacks:
            raise ValueError("Feedback not found")
        self.feedbacks[feedback_id].update(updated_data)
        self.feedbacks[feedback_id]["updated_at"] = datetime.now().isoformat()
        return self.feedbacks[feedback_id]

course_feedback_service = CourseFeedbackService()

def get_course_feedback_service() -> CourseFeedbackService:
    return course_feedback_service
