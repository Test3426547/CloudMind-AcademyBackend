from fastapi import HTTPException
from datetime import datetime, timedelta
from typing import List, Dict
import random

class GamificationEngine:
    def __init__(self):
        self.user_points = {}
        self.user_schedules = {}

    def award_points(self, user_id: str, points: int, reason: str) -> Dict[str, int]:
        if user_id not in self.user_points:
            self.user_points[user_id] = 0
        self.user_points[user_id] += points
        return {"user_id": user_id, "total_points": self.user_points[user_id], "awarded_points": points, "reason": reason}

    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, int]]:
        sorted_users = sorted(self.user_points.items(), key=lambda x: x[1], reverse=True)
        return [{"user_id": user_id, "points": points} for user_id, points in sorted_users[:limit]]

    def generate_adaptive_schedule(self, user_id: str, available_hours: List[int], study_duration: int) -> List[Dict[str, datetime]]:
        if user_id not in self.user_schedules:
            self.user_schedules[user_id] = []

        now = datetime.now()
        schedule = []

        for _ in range(7):  # Generate schedule for the next 7 days
            day = now.date()
            available_slots = [datetime.combine(day, datetime.min.time().replace(hour=h)) for h in available_hours]
            
            # Prioritize time slots based on user's past performance and current streak
            prioritized_slots = self._prioritize_time_slots(user_id, available_slots)
            
            if prioritized_slots:
                chosen_slot = prioritized_slots[0]
                end_time = chosen_slot + timedelta(minutes=study_duration)
                schedule.append({"start_time": chosen_slot, "end_time": end_time})
            
            now += timedelta(days=1)

        self.user_schedules[user_id] = schedule
        return schedule

    def _prioritize_time_slots(self, user_id: str, available_slots: List[datetime]) -> List[datetime]:
        # This is a simplified prioritization. In a real-world scenario, you would use machine learning
        # to analyze user behavior, performance, and preferences to prioritize time slots.
        if user_id in self.user_points:
            # Prioritize morning slots for users with higher points (assuming they're more committed)
            if self.user_points[user_id] > 100:
                return sorted(available_slots)
            else:
                return sorted(available_slots, reverse=True)
        else:
            return random.sample(available_slots, len(available_slots))

gamification_engine = GamificationEngine()

def get_gamification_engine() -> GamificationEngine:
    return gamification_engine
