from fastapi import HTTPException
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random
import uuid
import math

class GamificationEngine:
    def __init__(self):
        self.user_points = {}
        self.user_schedules = {}
        self.user_achievements = {}
        self.shared_achievements = {}
        self.user_challenges = {}
        self.user_performance = {}

    def award_points(self, user_id: str, points: int, reason: str) -> Dict[str, int]:
        if user_id not in self.user_points:
            self.user_points[user_id] = {"total": 0, "history": []}
        self.user_points[user_id]["total"] += points
        self.user_points[user_id]["history"].append({"points": points, "reason": reason, "timestamp": datetime.now()})
        self._update_user_performance(user_id, points)
        return {"user_id": user_id, "total_points": self.user_points[user_id]["total"], "awarded_points": points, "reason": reason}

    def get_leaderboard(self, limit: int = 10, period: str = "all_time") -> List[Dict[str, any]]:
        if period == "all_time":
            sorted_users = sorted(self.user_points.items(), key=lambda x: x[1]["total"], reverse=True)
            return [{"user_id": user_id, "points": points["total"]} for user_id, points in sorted_users[:limit]]
        elif period in ["daily", "weekly", "monthly"]:
            now = datetime.now()
            if period == "daily":
                start_time = now - timedelta(days=1)
            elif period == "weekly":
                start_time = now - timedelta(weeks=1)
            else:
                start_time = now - timedelta(days=30)
            
            period_points = {}
            for user_id, data in self.user_points.items():
                period_points[user_id] = sum(entry["points"] for entry in data["history"] if entry["timestamp"] >= start_time)
            
            sorted_users = sorted(period_points.items(), key=lambda x: x[1], reverse=True)
            return [{"user_id": user_id, "points": points} for user_id, points in sorted_users[:limit]]
        else:
            raise ValueError("Invalid period. Choose 'all_time', 'daily', 'weekly', or 'monthly'.")

    def generate_adaptive_schedule(self, user_id: str, available_hours: List[int], study_duration: int) -> List[Dict[str, datetime]]:
        if user_id not in self.user_schedules:
            self.user_schedules[user_id] = []

        now = datetime.now()
        schedule = []

        for _ in range(7):  # Generate schedule for the next 7 days
            day = now.date()
            available_slots = [datetime.combine(day, datetime.min.time().replace(hour=h)) for h in available_hours]
            
            prioritized_slots = self._prioritize_time_slots(user_id, available_slots)
            
            if prioritized_slots:
                chosen_slot = prioritized_slots[0]
                end_time = chosen_slot + timedelta(minutes=study_duration)
                schedule.append({"start_time": chosen_slot, "end_time": end_time})
            
            now += timedelta(days=1)

        self.user_schedules[user_id] = schedule
        return schedule

    def _prioritize_time_slots(self, user_id: str, available_slots: List[datetime]) -> List[datetime]:
        user_performance = self.user_performance.get(user_id, {"average_points": 0, "consistency": 0})
        
        if user_performance["consistency"] > 0.7:
            return sorted(available_slots)
        elif user_performance["average_points"] > 50:
            return sorted(available_slots, key=lambda x: x.hour)
        else:
            return random.sample(available_slots, len(available_slots))

    def get_user_stats(self, user_id: str) -> Dict[str, any]:
        if user_id not in self.user_points:
            raise HTTPException(status_code=404, detail="User not found")
        
        total_points = self.user_points[user_id]["total"]
        point_history = self.user_points[user_id]["history"]
        achievements = self.user_achievements.get(user_id, [])
        
        return {
            "user_id": user_id,
            "total_points": total_points,
            "point_history": point_history,
            "achievements": achievements,
            "rank": self._get_user_rank(user_id),
            "performance": self.user_performance.get(user_id, {})
        }

    def _get_user_rank(self, user_id: str) -> int:
        sorted_users = sorted(self.user_points.items(), key=lambda x: x[1]["total"], reverse=True)
        return next(i for i, (uid, _) in enumerate(sorted_users, 1) if uid == user_id)

    def share_achievement(self, user_id: str, achievement_id: str, platform: str) -> str:
        if user_id not in self.user_achievements or achievement_id not in self.user_achievements[user_id]:
            raise HTTPException(status_code=404, detail="Achievement not found")
        
        share_id = str(uuid.uuid4())
        self.shared_achievements[share_id] = {
            "user_id": user_id,
            "achievement_id": achievement_id,
            "platform": platform,
            "timestamp": datetime.now()
        }
        return share_id

    def generate_personalized_challenge(self, user_id: str) -> Dict[str, Any]:
        user_performance = self.user_performance.get(user_id, {"average_points": 0, "consistency": 0})
        
        if user_performance["consistency"] > 0.8:
            challenge_type = "time_based"
            target = random.randint(5, 10) * 60  # 5 to 10 hours
            description = f"Complete {target // 60} hours of study this week"
        elif user_performance["average_points"] > 100:
            challenge_type = "point_based"
            target = random.randint(500, 1000)
            description = f"Earn {target} points in the next 3 days"
        else:
            challenge_type = "streak_based"
            target = random.randint(3, 7)
            description = f"Maintain a {target}-day study streak"

        challenge = {
            "user_id": user_id,
            "type": challenge_type,
            "target": target,
            "description": description,
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(days=7),
            "completed": False
        }

        if user_id not in self.user_challenges:
            self.user_challenges[user_id] = []
        self.user_challenges[user_id].append(challenge)

        return challenge

    def update_challenge_progress(self, user_id: str, challenge_id: str, progress: int) -> Dict[str, Any]:
        user_challenges = self.user_challenges.get(user_id, [])
        challenge = next((c for c in user_challenges if c.get("id") == challenge_id), None)

        if not challenge:
            raise HTTPException(status_code=404, detail="Challenge not found")

        challenge["current_progress"] = progress
        if progress >= challenge["target"]:
            challenge["completed"] = True
            self.award_points(user_id, 100, f"Completed challenge: {challenge['description']}")

        return challenge

    def _update_user_performance(self, user_id: str, points: int):
        if user_id not in self.user_performance:
            self.user_performance[user_id] = {"total_points": 0, "activities": 0, "last_activity": None, "consistency": 0}

        performance = self.user_performance[user_id]
        performance["total_points"] += points
        performance["activities"] += 1
        performance["average_points"] = performance["total_points"] / performance["activities"]

        now = datetime.now()
        if performance["last_activity"]:
            time_diff = (now - performance["last_activity"]).days
            performance["consistency"] = 1 / (1 + math.exp(-0.5 * (7 - time_diff)))  # Logistic function for consistency

        performance["last_activity"] = now

    def predict_user_engagement(self, user_id: str) -> Dict[str, Any]:
        performance = self.user_performance.get(user_id, {})
        if not performance:
            return {"prediction": "Not enough data", "confidence": 0}

        avg_points = performance.get("average_points", 0)
        consistency = performance.get("consistency", 0)
        total_activities = performance.get("activities", 0)

        engagement_score = (avg_points * 0.4 + consistency * 0.4 + min(1, total_activities / 100) * 0.2) * 100
        confidence = min(1, total_activities / 50)  # Confidence increases with more activities, max at 50

        if engagement_score > 80:
            prediction = "High engagement"
        elif engagement_score > 50:
            prediction = "Moderate engagement"
        else:
            prediction = "Low engagement"

        return {
            "prediction": prediction,
            "engagement_score": round(engagement_score, 2),
            "confidence": round(confidence, 2)
        }

gamification_engine = GamificationEngine()

def get_gamification_engine() -> GamificationEngine:
    return gamification_engine
