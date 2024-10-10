from typing import List, Dict, Any
from models.course_prerequisite import CoursePrerequisite, UserCourseProgress

class CoursePrerequisiteService:
    def __init__(self):
        self.prerequisites: Dict[str, CoursePrerequisite] = {}
        self.user_progress: Dict[str, Dict[str, UserCourseProgress]] = {}

    async def add_prerequisite(self, prerequisite: CoursePrerequisite) -> None:
        self.prerequisites[prerequisite.course_id] = prerequisite

    async def get_prerequisites(self, course_id: str) -> List[str]:
        return self.prerequisites.get(course_id, CoursePrerequisite(course_id=course_id, prerequisite_course_ids=[])).prerequisite_course_ids

    async def update_user_progress(self, user_progress: UserCourseProgress) -> None:
        if user_progress.user_id not in self.user_progress:
            self.user_progress[user_progress.user_id] = {}
        self.user_progress[user_progress.user_id][user_progress.course_id] = user_progress

    async def get_user_progress(self, user_id: str, course_id: str) -> UserCourseProgress:
        return self.user_progress.get(user_id, {}).get(course_id, UserCourseProgress(user_id=user_id, course_id=course_id, completed=False, progress_percentage=0.0))

    async def check_prerequisites_met(self, user_id: str, course_id: str) -> bool:
        prerequisites = await self.get_prerequisites(course_id)
        for prereq_id in prerequisites:
            progress = await self.get_user_progress(user_id, prereq_id)
            if not progress.completed:
                return False
        return True

course_prerequisite_service = CoursePrerequisiteService()

def get_course_prerequisite_service() -> CoursePrerequisiteService:
    return course_prerequisite_service
