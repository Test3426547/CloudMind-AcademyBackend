import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class GradingService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.assignment_cache = {}
        self.difficulty_levels = {
            "easy": 0.7,
            "medium": 0.8,
            "hard": 0.9
        }

    async def grade_assignment(self, assignment_id: str, student_submission: str, rubric: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if assignment_id in self.assignment_cache:
                reference_answer = self.assignment_cache[assignment_id]["reference_answer"]
                difficulty = self.assignment_cache[assignment_id]["difficulty"]
            else:
                reference_answer = await self._generate_reference_answer(rubric)
                difficulty = rubric.get("difficulty", "medium")
                self.assignment_cache[assignment_id] = {
                    "reference_answer": reference_answer,
                    "difficulty": difficulty
                }

            content_score = await self._grade_content(student_submission, reference_answer, rubric)
            writing_quality_score = await self._grade_writing_quality(student_submission)
            plagiarism_score = await self._check_plagiarism(student_submission, reference_answer)

            total_score = self._calculate_total_score(content_score, writing_quality_score, plagiarism_score, difficulty)
            
            feedback = await self._generate_feedback(student_submission, content_score, writing_quality_score, plagiarism_score, total_score)

            return {
                "total_score": total_score,
                "content_score": content_score,
                "writing_quality_score": writing_quality_score,
                "plagiarism_score": plagiarism_score,
                "feedback": feedback
            }
        except Exception as e:
            logger.error(f"Error in grade_assignment: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred during assignment grading")

    async def _generate_reference_answer(self, rubric: Dict[str, Any]) -> str:
        prompt = f"Generate a high-quality reference answer for the following assignment rubric:\n\n{rubric}"
        reference_answer = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an expert in generating reference answers for academic assignments."},
            {"role": "user", "content": prompt}
        ], "high")
        return reference_answer

    async def _grade_content(self, student_submission: str, reference_answer: str, rubric: Dict[str, Any]) -> float:
        student_embedding = await self.text_embedding_service.get_embedding(student_submission)
        reference_embedding = await self.text_embedding_service.get_embedding(reference_answer)
        
        similarity = cosine_similarity([student_embedding], [reference_embedding])[0][0]
        
        content_score = similarity * 100
        
        # Adjust score based on rubric criteria
        for criterion, weight in rubric.get("criteria", {}).items():
            criterion_score = await self._evaluate_criterion(student_submission, criterion)
            content_score += criterion_score * weight
        
        return min(max(content_score, 0), 100)

    async def _evaluate_criterion(self, submission: str, criterion: str) -> float:
        prompt = f"Evaluate the following submission for the criterion: {criterion}\n\nSubmission: {submission}\n\nProvide a score between 0 and 1."
        response = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an expert in evaluating academic submissions based on specific criteria."},
            {"role": "user", "content": prompt}
        ], "medium")
        return float(response.strip())

    async def _grade_writing_quality(self, student_submission: str) -> float:
        prompt = f"Evaluate the writing quality of the following submission. Consider grammar, style, coherence, and overall clarity. Provide a score between 0 and 100.\n\nSubmission: {student_submission}"
        response = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an expert in evaluating writing quality."},
            {"role": "user", "content": prompt}
        ], "medium")
        return float(response.strip())

    async def _check_plagiarism(self, student_submission: str, reference_answer: str) -> float:
        vectorizer = TfidfVectorizer().fit_transform([student_submission, reference_answer])
        similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
        plagiarism_score = (1 - similarity) * 100  # Higher score means less plagiarism
        return plagiarism_score

    def _calculate_total_score(self, content_score: float, writing_quality_score: float, plagiarism_score: float, difficulty: str) -> float:
        base_score = (content_score * 0.6 + writing_quality_score * 0.3 + plagiarism_score * 0.1)
        difficulty_factor = self.difficulty_levels.get(difficulty, 0.8)
        return min(max(base_score * difficulty_factor, 0), 100)

    async def _generate_feedback(self, student_submission: str, content_score: float, writing_quality_score: float, plagiarism_score: float, total_score: float) -> str:
        prompt = f"""
        Generate constructive feedback for the following student submission:

        Submission: {student_submission}

        Scores:
        - Content: {content_score:.2f}
        - Writing Quality: {writing_quality_score:.2f}
        - Originality: {plagiarism_score:.2f}
        - Total Score: {total_score:.2f}

        Provide specific feedback on content, writing quality, and originality. Offer suggestions for improvement and highlight strengths.
        """
        feedback = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an expert academic advisor providing constructive feedback to students."},
            {"role": "user", "content": prompt}
        ], "high")
        return feedback

    async def generate_quiz(self, topic: str, difficulty: str, num_questions: int) -> List[Dict[str, Any]]:
        try:
            prompt = f"Generate a quiz on the topic of {topic} with {num_questions} questions at {difficulty} difficulty level. For each question, provide the question, 4 multiple-choice options, and the correct answer."
            quiz_content = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an expert in creating educational quizzes."},
                {"role": "user", "content": prompt}
            ], "high")

            quiz = []
            question_data = quiz_content.split("\n\n")
            for question in question_data:
                lines = question.strip().split("\n")
                quiz_item = {
                    "question": lines[0],
                    "options": lines[1:5],
                    "correct_answer": lines[5].replace("Correct answer: ", "")
                }
                quiz.append(quiz_item)

            return quiz
        except Exception as e:
            logger.error(f"Error in generate_quiz: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred during quiz generation")

grading_service = GradingService(get_llm_orchestrator(), get_text_embedding_service())

def get_grading_service() -> GradingService:
    return grading_service
