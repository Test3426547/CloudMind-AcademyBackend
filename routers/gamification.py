from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.gamification_engine import get_gamification_engine, GamificationEngine
from typing import List, Dict
from datetime import datetime

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/gamification/award-points")
async def award_points(user_id: str, points: int, reason: str, user: User = Depends(oauth2_scheme)) -> Dict[str, int]:
    gamification_engine = get_gamification_engine()
    result = gamification_engine.award_points(user_id, points, reason)
    return result

@router.get("/gamification/leaderboard")
async def get_leaderboard(limit: int = 10, user: User = Depends(oauth2_scheme)) -> Dict[str, List[Dict[str, int]]]:
    gamification_engine = get_gamification_engine()
    leaderboard = gamification_engine.get_leaderboard(limit)
    return {"leaderboard": leaderboard}

@router.post("/gamification/generate-schedule")
async def generate_adaptive_schedule(
    user_id: str,
    available_hours: List[int],
    study_duration: int,
    user: User = Depends(oauth2_scheme)
) -> Dict[str, List[Dict[str, datetime]]]:
    gamification_engine = get_gamification_engine()
    schedule = gamification_engine.generate_adaptive_schedule(user_id, available_hours, study_duration)
    return {"schedule": schedule}

@router.post("/gamification/complete-challenge")
async def complete_challenge(challenge_id: str, user: User = Depends(oauth2_scheme)) -> Dict[str, str]:
    # Here you would typically mark the challenge as completed and award points
    gamification_engine = get_gamification_engine()
    points_awarded = 50  # This could be dynamic based on the challenge difficulty
    result = gamification_engine.award_points(user.id, points_awarded, f"Completed challenge {challenge_id}")
    return {"message": f"Challenge {challenge_id} completed successfully. Awarded {points_awarded} points."}
