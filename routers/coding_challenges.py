from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.coding_challenges_service import CodingChallengesService, get_coding_challenges_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

class ChallengeCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=10, max_length=1000)
    difficulty: str = Field(..., regex="^(easy|medium|hard)$")

class SolutionSubmit(BaseModel):
    solution: str = Field(..., min_length=1)

@router.post("/challenges")
async def create_challenge(
    challenge: ChallengeCreate,
    user: User = Depends(oauth2_scheme),
    challenges_service: CodingChallengesService = Depends(get_coding_challenges_service),
):
    try:
        result = await challenges_service.create_challenge(challenge.dict())
        logger.info(f"Challenge created by user {user.id}")
        return result
    except Exception as e:
        logger.error(f"Error creating challenge: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while creating the challenge")

@router.get("/challenges/{challenge_id}")
async def get_challenge(
    challenge_id: str,
    user: User = Depends(oauth2_scheme),
    challenges_service: CodingChallengesService = Depends(get_coding_challenges_service),
):
    try:
        result = await challenges_service.get_challenge(challenge_id)
        logger.info(f"Challenge {challenge_id} retrieved by user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"Challenge not found: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error retrieving challenge: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving the challenge")

@router.post("/challenges/{challenge_id}/submit")
async def submit_solution(
    challenge_id: str,
    solution: SolutionSubmit,
    user: User = Depends(oauth2_scheme),
    challenges_service: CodingChallengesService = Depends(get_coding_challenges_service),
):
    try:
        result = await challenges_service.submit_solution(user.id, challenge_id, solution.solution)
        logger.info(f"Solution submitted for challenge {challenge_id} by user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"Error submitting solution: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error submitting solution: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while submitting the solution")

@router.get("/challenges/user-progress")
async def get_user_progress(
    user: User = Depends(oauth2_scheme),
    challenges_service: CodingChallengesService = Depends(get_coding_challenges_service),
):
    try:
        result = await challenges_service.get_user_progress(user.id)
        logger.info(f"User progress retrieved for user {user.id}")
        return result
    except Exception as e:
        logger.error(f"Error retrieving user progress: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving user progress")

@router.post("/challenges/{challenge_id}/generate-test-cases")
async def generate_test_cases(
    challenge_id: str,
    num_test_cases: int = Query(5, ge=1, le=20),
    user: User = Depends(oauth2_scheme),
    challenges_service: CodingChallengesService = Depends(get_coding_challenges_service),
):
    try:
        result = await challenges_service.generate_test_cases(challenge_id, num_test_cases)
        logger.info(f"Test cases generated for challenge {challenge_id} by user {user.id}")
        return result
    except Exception as e:
        logger.error(f"Error generating test cases: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating test cases")

@router.post("/challenges/{challenge_id}/adjust-difficulty")
async def adjust_difficulty(
    challenge_id: str,
    user: User = Depends(oauth2_scheme),
    challenges_service: CodingChallengesService = Depends(get_coding_challenges_service),
):
    try:
        result = await challenges_service.adjust_difficulty_ml(user.id, challenge_id)
        logger.info(f"Difficulty adjusted for challenge {challenge_id} for user {user.id}")
        return result
    except Exception as e:
        logger.error(f"Error adjusting difficulty: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while adjusting the difficulty")

@router.post("/challenges/{challenge_id}/check-similarity")
async def check_code_similarity(
    challenge_id: str,
    solution: SolutionSubmit,
    user: User = Depends(oauth2_scheme),
    challenges_service: CodingChallengesService = Depends(get_coding_challenges_service),
):
    try:
        result = await challenges_service.check_code_similarity(user.id, challenge_id, solution.solution)
        logger.info(f"Code similarity checked for challenge {challenge_id} by user {user.id}")
        return result
    except HTTPException as e:
        logger.warning(f"Error checking code similarity: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error checking code similarity: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while checking code similarity")

@router.post("/challenges/{challenge_id}/generate-hint")
async def generate_hint(
    challenge_id: str,
    solution: SolutionSubmit,
    user: User = Depends(oauth2_scheme),
    challenges_service: CodingChallengesService = Depends(get_coding_challenges_service),
):
    try:
        hint = await challenges_service.generate_hint(challenge_id, solution.solution)
        logger.info(f"Hint generated for challenge {challenge_id} for user {user.id}")
        return {"hint": hint}
    except HTTPException as e:
        logger.warning(f"Error generating hint: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error generating hint: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating the hint")
