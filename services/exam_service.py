import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
import random
import math
from collections import defaultdict
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service

logger = logging.getLogger(__name__)

class ExamService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.exams = {}
        self.user_exam_results = defaultdict(dict)

    async def create_exam(self, course_id: str, exam_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            exam_id = f"exam_{len(self.exams) + 1}"
            exam_embedding = await self._generate_exam_embedding(exam_data['title'], exam_data['description'], exam_data['questions'])
            difficulty = await self._estimate_exam_difficulty(exam_data['questions'])
            
            self.exams[exam_id] = {
                **exam_data,
                "course_id": course_id,
                "embedding": exam_embedding,
                "difficulty": difficulty
            }
            return {"exam_id": exam_id, "message": "Exam created successfully"}
        except Exception as e:
            logger.error(f"Error creating exam: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create exam")

    async def get_exam(self, exam_id: str) -> Dict[str, Any]:
        if exam_id not in self.exams:
            raise HTTPException(status_code=404, detail="Exam not found")
        return self.exams[exam_id]

    async def submit_exam(self, user_id: str, exam_id: str, answers: List[str]) -> Dict[str, Any]:
        if exam_id not in self.exams:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        exam = self.exams[exam_id]
        score = await self._grade_exam(exam, answers)
        self.user_exam_results[user_id][exam_id] = score
        
        feedback = await self._generate_exam_feedback(exam, answers, score)
        
        return {
            "score": score,
            "feedback": feedback,
            "message": "Exam submitted successfully"
        }

    async def get_user_exam_results(self, user_id: str) -> Dict[str, float]:
        return self.user_exam_results[user_id]

    async def _generate_exam_embedding(self, title: str, description: str, questions: List[Dict[str, Any]]) -> List[float]:
        # Simulated PyTorch-like embedding generation
        combined_text = f"{title} {description} {' '.join([q['question'] for q in questions])}"
        return await self.text_embedding_service.get_embedding(combined_text)

    async def _estimate_exam_difficulty(self, questions: List[Dict[str, Any]]) -> float:
        # Simulated TensorFlow-like difficulty estimation
        question_features = [
            [
                len(q['question'].split()),  # Length of question
                len(q['options']),  # Number of options
                1 if q['type'] == 'multiple_choice' else 2,  # Question type difficulty
                len([word for word in q['question'].split() if len(word) > 7])  # Number of complex words
            ]
            for q in questions
        ]
        
        # Simulated neural network for difficulty estimation
        weights = [[0.1, 0.2, 0.3, 0.4]]
        bias = [0.1]
        
        difficulties = [
            math.tanh(sum([feature[i] * weights[0][i] for i in range(len(feature))]) + bias[0])
            for feature in question_features
        ]
        
        return sum(difficulties) / len(difficulties) * 5 + 5  # Scale to 1-10

    async def _grade_exam(self, exam: Dict[str, Any], answers: List[str]) -> float:
        # Simulated PyTorch-like grading system
        correct_answers = [q['correct_answer'] for q in exam['questions']]
        
        answer_embeddings = [await self.text_embedding_service.get_embedding(ans) for ans in answers]
        correct_embeddings = [await self.text_embedding_service.get_embedding(ans) for ans in correct_answers]
        
        similarities = [
            self._cosine_similarity(ans_emb, correct_emb)
            for ans_emb, correct_emb in zip(answer_embeddings, correct_embeddings)
        ]
        
        score = sum(sim > 0.8 for sim in similarities) / len(similarities) * 100
        return score

    async def _generate_exam_feedback(self, exam: Dict[str, Any], answers: List[str], score: float) -> str:
        # Simulated HuggingFace Transformers text generation
        feedback_prompt = f"Generate feedback for an exam with a score of {score:.2f}%. Provide encouragement and suggestions for improvement."
        feedback = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI exam feedback generator using advanced language models."},
            {"role": "user", "content": feedback_prompt}
        ], "high")
        return feedback.strip()

    async def recommend_exams(self, user_id: str, num_recommendations: int = 3) -> List[Dict[str, Any]]:
        user_results = self.user_exam_results.get(user_id, {})
        if not user_results:
            return await self._get_popular_exams(num_recommendations)

        user_performance = await self._calculate_user_performance(user_id)
        exam_scores = []

        for exam_id, exam in self.exams.items():
            if exam_id not in user_results:
                similarity = self._cosine_similarity(user_performance['embedding'], exam['embedding'])
                difficulty_score = self._calculate_difficulty_score(user_performance['avg_difficulty'], exam['difficulty'])
                popularity_score = await self._calculate_popularity_score(exam_id)
                
                # Simulated PyTorch neural network for recommendation scoring
                features = [similarity, difficulty_score, popularity_score]
                weights = [0.5, 0.3, 0.2]
                bias = 0.1
                total_score = sum([f * w for f, w in zip(features, weights)]) + bias
                total_score = 1 / (1 + math.exp(-total_score))  # Simulated sigmoid activation
                
                exam_scores.append((exam_id, total_score))

        exam_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [
            {
                "exam_id": exam_id,
                "title": self.exams[exam_id]['title'],
                "score": score
            }
            for exam_id, score in exam_scores[:num_recommendations]
        ]

        return recommendations

    async def _calculate_user_performance(self, user_id: str) -> Dict[str, Any]:
        user_results = self.user_exam_results[user_id]
        if not user_results:
            return {"embedding": [0] * 100, "avg_difficulty": 5}

        completed_exams = [self.exams[exam_id] for exam_id in user_results.keys()]
        exam_embeddings = [exam['embedding'] for exam in completed_exams]
        
        # Simulated NumPy-like operations
        avg_embedding = [sum(emb[i] for emb in exam_embeddings) / len(exam_embeddings) for i in range(len(exam_embeddings[0]))]
        avg_difficulty = sum(exam['difficulty'] for exam in completed_exams) / len(completed_exams)

        return {
            "embedding": avg_embedding,
            "avg_difficulty": avg_difficulty
        }

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        # Simulated NumPy/PyTorch cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0

    def _calculate_difficulty_score(self, user_avg_difficulty: float, exam_difficulty: float) -> float:
        # Simulated TensorFlow operation
        difficulty_difference = abs(exam_difficulty - user_avg_difficulty)
        return 1.0 / (1.0 + difficulty_difference)

    async def _calculate_popularity_score(self, exam_id: str) -> float:
        # Simulated NumPy operation
        num_taken = sum(1 for results in self.user_exam_results.values() if exam_id in results)
        total_users = len(self.user_exam_results)
        return num_taken / total_users if total_users > 0 else 0

    async def _get_popular_exams(self, num_exams: int) -> List[Dict[str, Any]]:
        exam_popularities = [(exam_id, await self._calculate_popularity_score(exam_id)) 
                             for exam_id in self.exams]
        exam_popularities.sort(key=lambda x: x[1], reverse=True)
        popular_exams = exam_popularities[:num_exams]
        return [
            {
                "exam_id": exam_id,
                "title": self.exams[exam_id]['title'],
                "popularity_score": popularity
            }
            for exam_id, popularity in popular_exams
        ]

    async def generate_exam_questions(self, course_id: str, num_questions: int = 10) -> List[Dict[str, Any]]:
        # Simulated HuggingFace Transformers question generation
        course = await self._get_course_content(course_id)
        prompt = f"Generate {num_questions} exam questions based on the following course content:\n\n{course}"
        
        generated_questions = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI exam question generator using advanced language models."},
            {"role": "user", "content": prompt}
        ], "high")
        
        # Parse and structure the generated questions
        questions = self._parse_generated_questions(generated_questions)
        return questions

    async def _get_course_content(self, course_id: str) -> str:
        # Simulated course content retrieval
        # In a real implementation, this would fetch the course content from a database
        return f"This is the content for course {course_id}. It covers various topics related to the subject."

    def _parse_generated_questions(self, generated_questions: str) -> List[Dict[str, Any]]:
        # Simple parsing of generated questions
        # In a real implementation, this would use more sophisticated NLP techniques
        lines = generated_questions.strip().split('\n')
        questions = []
        for i in range(0, len(lines), 6):
            if i + 5 < len(lines):
                question = {
                    "question": lines[i],
                    "options": lines[i+1:i+5],
                    "correct_answer": lines[i+5],
                    "type": "multiple_choice"
                }
                questions.append(question)
        return questions

    async def analyze_exam_performance(self, exam_id: str) -> Dict[str, Any]:
        if exam_id not in self.exams:
            raise HTTPException(status_code=404, detail="Exam not found")

        exam = self.exams[exam_id]
        scores = [result[exam_id] for result in self.user_exam_results.values() if exam_id in result]

        if not scores:
            return {"message": "No exam results available for analysis"}

        # Simulated NumPy operations
        avg_score = sum(scores) / len(scores)
        median_score = sorted(scores)[len(scores) // 2]
        std_dev = math.sqrt(sum((s - avg_score) ** 2 for s in scores) / len(scores))

        # Simulated sklearn-like clustering for difficulty analysis
        difficulty_clusters = self._cluster_questions_by_difficulty(exam, scores)

        return {
            "exam_id": exam_id,
            "average_score": avg_score,
            "median_score": median_score,
            "standard_deviation": std_dev,
            "num_participants": len(scores),
            "difficulty_analysis": difficulty_clusters
        }

    def _cluster_questions_by_difficulty(self, exam: Dict[str, Any], scores: List[float]) -> Dict[str, List[int]]:
        # Simulated sklearn KMeans clustering
        question_difficulties = []
        for i, question in enumerate(exam['questions']):
            correct_count = sum(1 for result in self.user_exam_results.values() 
                                if exam['id'] in result and result[exam['id']]['answers'][i] == question['correct_answer'])
            difficulty = 1 - (correct_count / len(scores))
            question_difficulties.append((i, difficulty))

        # Simple clustering based on difficulty
        easy = [i for i, diff in question_difficulties if diff < 0.3]
        medium = [i for i, diff in question_difficulties if 0.3 <= diff < 0.7]
        hard = [i for i, diff in question_difficulties if diff >= 0.7]

        return {
            "easy": easy,
            "medium": medium,
            "hard": hard
        }

exam_service = ExamService(get_llm_orchestrator(), get_text_embedding_service())

def get_exam_service() -> ExamService:
    return exam_service
