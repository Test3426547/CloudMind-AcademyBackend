from pydantic import BaseModel
from typing import List

class CoursePrerequisite(BaseModel):
    course_id: str
    prerequisite_course_ids: List[str]

class UserCourseProgress(BaseModel):
    user_id: str
    course_id: str
    completed: bool
    progress_percentage: float
