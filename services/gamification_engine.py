from fastapi import HTTPException
from datetime import datetime, timedelta
from typing import List, Dict
import random
import uuid

class GamificationEngine:
    def __init__(self):
        self.user_points = {}
        self.user_schedules = {}
        self.user_achievements = {}
        self.shared_achievements = {}

    def award_points(self, user_id: str, points: int, reason: str) -> Dict[str, int]:
        if user_id not in self.user_points:
            self.user_points[user_id] = {"total": 0, "history": []}
        self.user_points[user_id]["total"] += points
        self.user_points[user_id]["history"].append({"points": points, "reason": reason, "timestamp": datetime.now()})
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
        if user_id in self.user_points:
            if self.user_points[user_id]["total"] > 100:
                return sorted(available_slots)
            else:
                return sorted(available_slots, reverse=True)
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
            "rank": self._get_user_rank(user_id)
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

gamification_engine = GamificationEngine()

def get_gamification_engine() -> GamificationEngine:
    return gamification_engine
