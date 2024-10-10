from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from typing import List

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/gamification/award-points")
async def award_points(user_id: str, points: int, reason: str, user: User = Depends(oauth2_scheme)):
    # Here you would typically update the user's points in your database
    return {"message": f"Awarded {points} points to user {user_id} for {reason}"}

@router.get("/gamification/leaderboard")
async def get_leaderboard(user: User = Depends(oauth2_scheme)):
    # Here you would typically fetch the leaderboard data from your database
    mock_leaderboard = [
        {"user_id": "1", "username": "user1", "points": 1000},
        {"user_id": "2", "username": "user2", "points": 900},
        {"user_id": "3", "username": "user3", "points": 800},
    ]
    return {"leaderboard": mock_leaderboard}

@router.post("/gamification/complete-challenge")
async def complete_challenge(challenge_id: str, user: User = Depends(oauth2_scheme)):
    # Here you would typically mark the challenge as completed and award points
    return {"message": f"Challenge {challenge_id} completed successfully"}
