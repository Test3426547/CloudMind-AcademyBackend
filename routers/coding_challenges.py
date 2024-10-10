from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.coding_challenges_service import CodingChallengesService, get_coding_challenges_service
from typing import List, Dict, Any
from pydantic import BaseModel

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class CodingChallenge(BaseModel):
    id: str
    title: str
    description: str
    difficulty: str
    initial_code: str
    test_cases: List[Dict[str, Any]]

class ChallengeSubmission(BaseModel):
    challenge_id: str
    user_code: str

@router.get("/coding-challenges", response_model=List[CodingChallenge])
async def get_coding_challenges(
    user: User = Depends(oauth2_scheme),
    challenges_service: CodingChallengesService = Depends(get_coding_challenges_service)
):
    return await challenges_service.get_challenges()

@router.get("/coding-challenges/{challenge_id}", response_model=CodingChallenge)
async def get_coding_challenge(
    challenge_id: str,
    user: User = Depends(oauth2_scheme),
    challenges_service: CodingChallengesService = Depends(get_coding_challenges_service)
):
    challenge = await challenges_service.get_challenge(challenge_id)
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    return challenge

@router.post("/coding-challenges/submit")
async def submit_coding_challenge(
    submission: ChallengeSubmission,
    user: User = Depends(oauth2_scheme),
    challenges_service: CodingChallengesService = Depends(get_coding_challenges_service)
):
    result = await challenges_service.evaluate_submission(submission.challenge_id, submission.user_code)
    return result

@router.get("/coding-challenges/leaderboard/{challenge_id}")
async def get_challenge_leaderboard(
    challenge_id: str,
    user: User = Depends(oauth2_scheme),
    challenges_service: CodingChallengesService = Depends(get_coding_challenges_service)
):
    leaderboard = await challenges_service.get_leaderboard(challenge_id)
    return leaderboard
