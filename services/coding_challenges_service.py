import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
import random
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class CodingChallengesService:
    # ... (previous methods remain unchanged)

    async def submit_solution(self, user_id: str, challenge_id: str, solution: str) -> Dict[str, Any]:
        if challenge_id not in self.challenges:
            raise HTTPException(status_code=404, detail="Challenge not found")

        await asyncio.sleep(1)
        
        code_analysis = await self.analyze_code(solution, self.challenges[challenge_id])

        feedback = await self.generate_feedback(user_id, challenge_id, code_analysis)

        learning_resources = await self.suggest_learning_resources(code_analysis)

        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}
        self.user_progress[user_id][challenge_id] = {
            "completed": True,
            "solution": solution,
            "analysis": code_analysis,
            "feedback": feedback
        }

        return {
            "success": True,
            "message": "Solution submitted successfully",
            "analysis": code_analysis,
            "feedback": feedback,
            "learning_resources": learning_resources
        }

    async def generate_feedback(self, user_id: str, challenge_id: str, analysis: Dict[str, Any]) -> str:
        user_history = self.user_progress.get(user_id, {})
        challenge = self.challenges[challenge_id]

        prompt = f"""
        Generate personalized feedback for a user who has completed a coding challenge.
        Challenge difficulty: {challenge['difficulty']}
        User's past performance: {len(user_history)} challenges completed
        Code analysis:
        - Quality score: {analysis['quality_score']}
        - Efficiency score: {analysis['efficiency_score']}
        - Correctness score: {analysis['correctness_score']}
        - Innovation score: {analysis['innovation_score']}
        - Maintainability score: {analysis['maintainability_score']}
        - Scalability score: {analysis['scalability_score']}
        - Insights: {analysis['insights']}

        Provide detailed and constructive feedback, including:
        1. Specific praise for aspects done well
        2. Areas for improvement with actionable suggestions
        3. Explanation of advanced concepts related to the solution
        4. Comparison with best practices and industry standards
        5. Suggestions for optimizing the code's performance and scalability
        6. Tips for improving code maintainability and readability
        7. Encouragement to explore related topics or technologies
        8. Advice on how to apply the learned concepts in real-world scenarios
        """

        feedback = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an expert coding mentor with years of experience in teaching and guiding software developers. Provide personalized, motivating, and insightful feedback for a student who has completed a coding challenge."},
            {"role": "user", "content": prompt}
        ], "high")

        return feedback

    async def suggest_learning_resources(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        prompt = f"""
        Based on the following code analysis, suggest 3-5 learning resources (articles, tutorials, or courses) that would help the user improve their skills in areas where they can grow:

        Analysis insights: {analysis['insights']}

        For each resource, provide:
        1. Title
        2. Brief description (1-2 sentences)
        3. URL (use a placeholder if you don't have a real URL)
        """

        resources_text = await self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an expert in recommending programming learning resources. Suggest high-quality, relevant resources based on the given code analysis."},
            {"role": "user", "content": prompt}
        ], "medium")

        # Parse the resources_text to extract individual resources
        resources = []
        for resource in resources_text.split("\n\n"):
            lines = resource.split("\n")
            if len(lines) >= 3:
                resources.append({
                    "title": lines[0].strip(),
                    "description": lines[1].strip(),
                    "url": lines[2].strip()
                })

        return resources

    # ... (other methods remain unchanged)

coding_challenges_service = CodingChallengesService()

def get_coding_challenges_service() -> CodingChallengesService:
    return coding_challenges_service
