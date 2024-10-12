from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.gamification_engine import get_gamification_engine, GamificationEngine
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class LeaderboardEntry(BaseModel):
    user_id: str
    points: int

class LeaderboardResponse(BaseModel):
    leaderboard: List[LeaderboardEntry]

class ChallengeResponse(BaseModel):
    user_id: str
    type: str
    target: int
    description: str
    start_time: datetime
    end_time: datetime
    completed: bool

class ChallengeProgressUpdate(BaseModel):
    challenge_id: str
    progress: int

class UserEngagementPrediction(BaseModel):
    prediction: str
    engagement_score: float
    confidence: float

@router.post("/gamification/award-points")
async def award_points(user_id: str, points: int, reason: str, user: User = Depends(oauth2_scheme)):
    gamification_engine = get_gamification_engine()
    result = gamification_engine.award_points(user_id, points, reason)
    return result

@router.get("/gamification/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(limit: int = 10, period: str = "all_time", user: User = Depends(oauth2_scheme)):
    gamification_engine = get_gamification_engine()
    leaderboard = gamification_engine.get_leaderboard(limit, period)
    return LeaderboardResponse(leaderboard=[LeaderboardEntry(**entry) for entry in leaderboard])

@router.post("/gamification/generate-schedule")
async def generate_adaptive_schedule(
    user_id: str,
    available_hours: List[int],
    study_duration: int,
    user: User = Depends(oauth2_scheme)
):
    gamification_engine = get_gamification_engine()
    schedule = gamification_engine.generate_adaptive_schedule(user_id, available_hours, study_duration)
    return {"schedule": schedule}

@router.get("/gamification/user-stats")
async def get_user_stats(user: User = Depends(oauth2_scheme)):
    gamification_engine = get_gamification_engine()
    stats = gamification_engine.get_user_stats(user.id)
    return stats

@router.post("/gamification/share-achievement")
async def share_achievement(achievement_id: str, platform: str, user: User = Depends(oauth2_scheme)):
    gamification_engine = get_gamification_engine()
    result = gamification_engine.share_achievement(user.id, achievement_id, platform)
    return {"message": f"Achievement shared on {platform}", "share_id": result}

@router.post("/gamification/generate-challenge", response_model=ChallengeResponse)
async def generate_personalized_challenge(user: User = Depends(oauth2_scheme)):
    gamification_engine = get_gamification_engine()
    challenge = gamification_engine.generate_personalized_challenge(user.id)
    return ChallengeResponse(**challenge)

@router.post("/gamification/update-challenge-progress", response_model=ChallengeResponse)
async def update_challenge_progress(progress_update: ChallengeProgressUpdate, user: User = Depends(oauth2_scheme)):
    gamification_engine = get_gamification_engine()
    updated_challenge = gamification_engine.update_challenge_progress(user.id, progress_update.challenge_id, progress_update.progress)
    return ChallengeResponse(**updated_challenge)

@router.get("/gamification/predict-engagement", response_model=UserEngagementPrediction)
async def predict_user_engagement(user: User = Depends(oauth2_scheme)):
    gamification_engine = get_gamification_engine()
    prediction = gamification_engine.predict_user_engagement(user.id)
    return UserEngagementPrediction(**prediction)
