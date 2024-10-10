from typing import List, Dict, Any
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from fastapi import Depends

class GradingService:
    def __init__(self, llm_orchestrator: LLMOrchestrator):
        self.llm_orchestrator = llm_orchestrator

    async def grade_assignment(self, assignment_text: str, student_submission: str) -> Dict[str, Any]:
        prompt = f"""
        You are an AI grading assistant. Your task is to grade the following student submission based on the given assignment. Provide a detailed evaluation, including a numeric score (0-100), feedback, and areas for improvement.

        Assignment:
        {assignment_text}

        Student Submission:
        {student_submission}

        Please provide your evaluation in the following format:
        {{
            "score": int,
            "feedback": str,
            "areas_for_improvement": List[str],
            "strengths": List[str]
        }}
        """

        response = self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI grading assistant."},
            {"role": "user", "content": prompt}
        ], "high")

        return eval(response)

    async def generate_quiz(self, course_content: str, difficulty: str, num_questions: int) -> List[Dict[str, Any]]:
        prompt = f"""
        Generate a quiz based on the following course content. The quiz should have {num_questions} questions and be of {difficulty} difficulty.

        Course Content:
        {course_content}

        Please provide the quiz in the following format:
        [
            {{
                "question": str,
                "options": List[str],
                "correct_answer": int,
                "explanation": str
            }},
            ...
        ]
        """

        response = self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI quiz generator."},
            {"role": "user", "content": prompt}
        ], "high")

        return eval(response)

def get_grading_service(llm_orchestrator: LLMOrchestrator = Depends(get_llm_orchestrator)) -> GradingService:
    return GradingService(llm_orchestrator)
