from typing import List, Dict, Optional
from datetime import datetime
from fastapi import HTTPException
from models.user import User
from auth_config import get_supabase_client
import logging

logger = logging.getLogger(__name__)

class CourseFeedbackService:
    def __init__(self):
        self.supabase = get_supabase_client()

    async def create_feedback(self, user_id: str, course_id: str, rating: int, comment: str) -> Dict:
        try:
            if not self._validate_rating(rating):
                raise ValueError("Invalid rating. Must be between 1 and 5.")

            new_feedback = {
                "user_id": user_id,
                "course_id": course_id,
                "rating": rating,
                "comment": comment,
                "created_at": datetime.utcnow().isoformat()
            }
            result = self.supabase.table('course_feedback').insert(new_feedback).execute()
            return result.data[0]
        except Exception as e:
            logger.error(f"Error creating feedback: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create feedback")

    async def get_course_feedback(self, course_id: str, limit: int, offset: int) -> List[Dict]:
        try:
            result = self.supabase.table('course_feedback').select('*').eq('course_id', course_id).order('created_at', desc=True).range(offset, offset + limit - 1).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error retrieving course feedback: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to retrieve course feedback")

    async def get_user_feedback(self, user_id: str, limit: int, offset: int) -> List[Dict]:
        try:
            result = self.supabase.table('course_feedback').select('*').eq('user_id', user_id).order('created_at', desc=True).range(offset, offset + limit - 1).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error retrieving user feedback: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to retrieve user feedback")

    async def update_feedback(self, user_id: str, feedback_id: str, rating: int, comment: str) -> Dict:
        try:
            if not self._validate_rating(rating):
                raise ValueError("Invalid rating. Must be between 1 and 5.")

            existing_feedback = self.supabase.table('course_feedback').select('*').eq('id', feedback_id).eq('user_id', user_id).execute()
            if not existing_feedback.data:
                raise HTTPException(status_code=404, detail="Feedback not found or you're not authorized to update it")

            updated_feedback = {
                "rating": rating,
                "comment": comment,
                "updated_at": datetime.utcnow().isoformat()
            }
            result = self.supabase.table('course_feedback').update(updated_feedback).eq('id', feedback_id).execute()
            return result.data[0]
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating feedback: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to update feedback")

    async def delete_feedback(self, user_id: str, feedback_id: str) -> None:
        try:
            existing_feedback = self.supabase.table('course_feedback').select('*').eq('id', feedback_id).eq('user_id', user_id).execute()
            if not existing_feedback.data:
                raise HTTPException(status_code=404, detail="Feedback not found or you're not authorized to delete it")

            self.supabase.table('course_feedback').delete().eq('id', feedback_id).execute()
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting feedback: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to delete feedback")

    async def get_course_feedback_stats(self, course_id: str) -> Dict:
        try:
            result = self.supabase.table('course_feedback').select('rating').eq('course_id', course_id).execute()
            ratings = [feedback['rating'] for feedback in result.data]
            
            if not ratings:
                return {"average_rating": 0, "total_reviews": 0, "rating_distribution": {}}

            average_rating = sum(ratings) / len(ratings)
            rating_distribution = {i: ratings.count(i) for i in range(1, 6)}

            return {
                "average_rating": round(average_rating, 2),
                "total_reviews": len(ratings),
                "rating_distribution": rating_distribution
            }
        except Exception as e:
            logger.error(f"Error retrieving course feedback stats: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to retrieve course feedback stats")

    def _validate_rating(self, rating: int) -> bool:
        return 1 <= rating <= 5

course_feedback_service = CourseFeedbackService()

def get_course_feedback_service() -> CourseFeedbackService:
    return course_feedback_service
