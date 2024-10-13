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
        words = combined_text.lower().split()
        vocab = list(set(words))
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        
        # Create a simple bag-of-words embedding (simulating PyTorch's nn.Embedding)
        embedding = [0.0] * len(vocab)
        for word in words:
            embedding[word_to_idx[word]] += 1.0
        
        # Apply simulated PyTorch's F.normalize
        norm = math.sqrt(sum([x**2 for x in embedding]))
        embedding = [x / norm for x in embedding]
        
        return embedding

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
        
        # Simulated TensorFlow's tf.keras.layers.Dense
        weights = [[0.1, 0.2, 0.3, 0.4]]
        bias = [0.1]
        
        # Simulated matrix multiplication and activation (tf.nn.tanh)
        difficulties = [
            math.tanh(sum([feature[i] * weights[0][i] for i in range(len(feature))]) + bias[0])
            for feature in question_features
        ]
        
        return sum(difficulties) / len(difficulties) * 5 + 5  # Scale to 1-10

    async def _grade_exam(self, exam: Dict[str, Any], answers: List[str]) -> float:
        # Simulated PyTorch-like grading system
        correct_answers = [q['correct_answer'] for q in exam['questions']]
        
        # Simulated embedding generation for answers
        answer_embeddings = [await self._generate_answer_embedding(ans) for ans in answers]
        correct_embeddings = [await self._generate_answer_embedding(ans) for ans in correct_answers]
        
        # Simulated cosine similarity calculation (PyTorch's F.cosine_similarity)
        similarities = [
            self._cosine_similarity(ans_emb, correct_emb)
            for ans_emb, correct_emb in zip(answer_embeddings, correct_embeddings)
        ]
        
        # Calculate score based on similarities
        score = sum(sim > 0.8 for sim in similarities) / len(similarities) * 100
        return score

    async def _generate_answer_embedding(self, answer: str) -> List[float]:
        # Simulated HuggingFace Transformers tokenization and embedding
        words = answer.lower().split()
        embedding = [hash(word) % 100 / 100 for word in words]  # Simple hash-based embedding
        return embedding

    async def _generate_exam_feedback(self, exam: Dict[str, Any], answers: List[str], score: float) -> str:
        # Simulated HuggingFace Transformers text generation
        feedback_prompt = f"Generate feedback for an exam with a score of {score:.2f}%. Provide encouragement and suggestions for improvement."
        feedback = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI exam feedback generator using advanced language models."},
            {"role": "user", "content": feedback_prompt}
        ], "high")  # Using high quality for more sophisticated feedback
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

exam_service = ExamService(get_llm_orchestrator(), get_text_embedding_service())

def get_exam_service() -> ExamService:
    return exam_service
