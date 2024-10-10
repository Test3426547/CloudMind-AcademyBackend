import json
from typing import Dict, List, Any
from datetime import datetime

class OfflineLearningService:
    def __init__(self):
        self.local_cache = {
            "courses": {},
            "user_progress": {},
            "quiz_responses": {}
        }
        self.sync_queue = []

    def cache_course_content(self, course_id: str, content: Dict[str, Any]):
        self.local_cache["courses"][course_id] = content

    def get_cached_course_content(self, course_id: str) -> Dict[str, Any]:
        return self.local_cache["courses"].get(course_id, {})

    def update_user_progress(self, user_id: str, course_id: str, progress: float):
        if user_id not in self.local_cache["user_progress"]:
            self.local_cache["user_progress"][user_id] = {}
        self.local_cache["user_progress"][user_id][course_id] = progress
        self.sync_queue.append({
            "type": "user_progress",
            "user_id": user_id,
            "course_id": course_id,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        })

    def get_user_progress(self, user_id: str, course_id: str) -> float:
        return self.local_cache["user_progress"].get(user_id, {}).get(course_id, 0.0)

    def save_quiz_response(self, user_id: str, quiz_id: str, responses: List[Dict[str, Any]]):
        if user_id not in self.local_cache["quiz_responses"]:
            self.local_cache["quiz_responses"][user_id] = {}
        self.local_cache["quiz_responses"][user_id][quiz_id] = responses
        self.sync_queue.append({
            "type": "quiz_response",
            "user_id": user_id,
            "quiz_id": quiz_id,
            "responses": responses,
            "timestamp": datetime.now().isoformat()
        })

    def get_quiz_responses(self, user_id: str, quiz_id: str) -> List[Dict[str, Any]]:
        return self.local_cache["quiz_responses"].get(user_id, {}).get(quiz_id, [])

    def get_sync_queue(self) -> List[Dict[str, Any]]:
        return self.sync_queue

    def clear_sync_queue(self):
        self.sync_queue = []

    def merge_server_data(self, server_data: Dict[str, Any]):
        # Merge server data with local cache, resolving conflicts
        for data_type, data in server_data.items():
            if data_type == "courses":
                self.local_cache["courses"].update(data)
            elif data_type == "user_progress":
                for user_id, progress in data.items():
                    if user_id not in self.local_cache["user_progress"]:
                        self.local_cache["user_progress"][user_id] = {}
                    self.local_cache["user_progress"][user_id].update(progress)
            elif data_type == "quiz_responses":
                for user_id, responses in data.items():
                    if user_id not in self.local_cache["quiz_responses"]:
                        self.local_cache["quiz_responses"][user_id] = {}
                    self.local_cache["quiz_responses"][user_id].update(responses)

offline_learning_service = OfflineLearningService()

def get_offline_learning_service() -> OfflineLearningService:
    return offline_learning_service
