import uuid
from typing import List, Dict, Any
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from fastapi import Depends

class CodingChallengesService:
    def __init__(self, llm_orchestrator: LLMOrchestrator):
        self.challenges = {}
        self.submissions = {}
        self.llm_orchestrator = llm_orchestrator

    async def get_challenges(self) -> List[Dict[str, Any]]:
        return list(self.challenges.values())

    async def get_challenge(self, challenge_id: str) -> Dict[str, Any]:
        return self.challenges.get(challenge_id)

    async def create_challenge(self, challenge_data: Dict[str, Any]) -> Dict[str, Any]:
        challenge_id = str(uuid.uuid4())
        challenge = {
            "id": challenge_id,
            **challenge_data
        }
        self.challenges[challenge_id] = challenge
        return challenge

    async def evaluate_submission(self, challenge_id: str, user_code: str) -> Dict[str, Any]:
        challenge = self.challenges.get(challenge_id)
        if not challenge:
            raise ValueError("Challenge not found")

        prompt = f"""
        Evaluate the following code submission for the coding challenge:

        Challenge: {challenge['title']}
        Description: {challenge['description']}
        Test Cases: {challenge['test_cases']}

        User Code:
        {user_code}

        Please provide an evaluation in the following format:
        {{
            "passed": boolean,
            "score": int,
            "feedback": str,
            "test_results": List[Dict[str, Any]]
        }}
        """

        response = self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI code evaluator."},
            {"role": "user", "content": prompt}
        ], "high")

        if response is None:
            return {"error": "Failed to evaluate code submission"}

        try:
            evaluation = eval(response)
            return evaluation
        except Exception as e:
            return {"error": f"Failed to parse evaluation response: {str(e)}"}

    async def get_leaderboard(self, challenge_id: str) -> List[Dict[str, Any]]:
        if challenge_id not in self.submissions:
            return []

        sorted_submissions = sorted(
            self.submissions[challenge_id],
            key=lambda x: x['score'],
            reverse=True
        )

        return [
            {
                "user_id": submission['user_id'],
                "score": submission['score'],
                "submission_time": submission['submission_time']
            }
            for submission in sorted_submissions[:10]  # Top 10 submissions
        ]

def get_coding_challenges_service(llm_orchestrator: LLMOrchestrator = Depends(get_llm_orchestrator)) -> CodingChallengesService:
    return CodingChallengesService(llm_orchestrator)
