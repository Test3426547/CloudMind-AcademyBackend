import json
from typing import Dict, List, Any
from datetime import datetime
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class OfflineLearningService:
    def __init__(self):
        self.local_cache = {
            "courses": {},
            "user_progress": {},
            "quiz_responses": {}
        }
        self.sync_queue = []
        self.llm_orchestrator = get_llm_orchestrator()
        self.scaler = StandardScaler()
        self.progress_model = LinearRegression()

    def cache_course_content(self, course_id: str, content: Dict[str, Any]):
        self.local_cache["courses"][course_id] = content

    def get_cached_course_content(self, course_id: str) -> Dict[str, Any]:
        return self.local_cache["courses"].get(course_id, {})

    async def update_user_progress(self, user_id: str, course_id: str, progress: float):
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
        await self._update_progress_model(user_id, course_id, progress)

    def get_user_progress(self, user_id: str, course_id: str) -> float:
        return self.local_cache["user_progress"].get(user_id, {}).get(course_id, 0.0)

    async def save_quiz_response(self, user_id: str, quiz_id: str, responses: List[Dict[str, Any]]):
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
        await self._analyze_quiz_responses(user_id, quiz_id, responses)

    def get_quiz_responses(self, user_id: str, quiz_id: str) -> List[Dict[str, Any]]:
        return self.local_cache["quiz_responses"].get(user_id, {}).get(quiz_id, [])

    def get_sync_queue(self) -> List[Dict[str, Any]]:
        return self.sync_queue

    async def clear_sync_queue(self):
        self.sync_queue = []
        await self._schedule_next_sync()

    async def merge_server_data(self, server_data: Dict[str, Any]):
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
        await self._prioritize_content()

    async def _prioritize_content(self):
        for user_id, courses in self.local_cache["user_progress"].items():
            user_courses = list(courses.keys())
            if len(user_courses) > 1:
                prompt = f"Prioritize the following courses for user {user_id} based on their progress and importance: {', '.join(user_courses)}"
                prioritized_courses = await self.llm_orchestrator.process_request([
                    {"role": "system", "content": "You are an AI assistant that prioritizes educational content."},
                    {"role": "user", "content": prompt}
                ], "low")
                prioritized_courses = [course.strip() for course in prioritized_courses.split(',')]
                self.local_cache["user_progress"][user_id] = {course: courses[course] for course in prioritized_courses if course in courses}

    async def _schedule_next_sync(self):
        # Implement intelligent sync scheduling based on user activity patterns
        # For simplicity, we'll just schedule the next sync after a fixed interval
        # In a real-world scenario, you'd use more sophisticated scheduling algorithms
        print("Next sync scheduled in 1 hour")

    async def _update_progress_model(self, user_id: str, course_id: str, progress: float):
        X = np.array([[len(self.local_cache["user_progress"][user_id]), progress]])
        y = np.array([progress])
        
        if len(X) > 1:
            X = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.progress_model.fit(X_train, y_train)

    async def predict_progress(self, user_id: str, course_id: str) -> float:
        if user_id not in self.local_cache["user_progress"] or course_id not in self.local_cache["user_progress"][user_id]:
            return 0.0
        
        X = np.array([[len(self.local_cache["user_progress"][user_id]), self.local_cache["user_progress"][user_id][course_id]]])
        X = self.scaler.transform(X)
        predicted_progress = self.progress_model.predict(X)[0]
        return max(0.0, min(100.0, predicted_progress))

    async def _analyze_quiz_responses(self, user_id: str, quiz_id: str, responses: List[Dict[str, Any]]):
        correct_answers = sum(1 for response in responses if response.get("is_correct", False))
        total_questions = len(responses)
        score = (correct_answers / total_questions) * 100 if total_questions > 0 else 0

        prompt = f"Analyze the quiz performance for user {user_id} on quiz {quiz_id}. They scored {score}% ({correct_answers}/{total_questions}). Provide a brief feedback and suggest areas for improvement."
        feedback = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI tutor providing feedback on quiz performance."},
            {"role": "user", "content": prompt}
        ], "medium")

        print(f"Quiz Analysis for User {user_id}, Quiz {quiz_id}:")
        print(feedback)

offline_learning_service = OfflineLearningService()

def get_offline_learning_service() -> OfflineLearningService:
    return offline_learning_service
