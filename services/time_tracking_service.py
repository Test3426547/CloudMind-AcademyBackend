from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator

class TimeTrackingService:
    def __init__(self, llm_orchestrator: LLMOrchestrator):
        self.time_entries: Dict[str, Dict[str, Any]] = {}
        self.llm_orchestrator = llm_orchestrator

    async def start_timer(self, user_id: str, task_id: str, description: str) -> Dict[str, Any]:
        entry_id = str(uuid.uuid4())
        entry = {
            "id": entry_id,
            "user_id": user_id,
            "task_id": task_id,
            "description": description,
            "start_time": datetime.now(),
            "end_time": None,
            "duration": None
        }
        self.time_entries[entry_id] = entry
        return entry

    async def stop_timer(self, entry_id: str) -> Dict[str, Any]:
        if entry_id not in self.time_entries:
            raise ValueError("Time entry not found")
        
        entry = self.time_entries[entry_id]
        if entry["end_time"] is not None:
            raise ValueError("Timer already stopped")
        
        entry["end_time"] = datetime.now()
        entry["duration"] = (entry["end_time"] - entry["start_time"]).total_seconds()
        return entry

    async def get_user_time_entries(self, user_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        return [
            entry for entry in self.time_entries.values()
            if entry["user_id"] == user_id and start_date <= entry["start_time"] <= end_date
        ]

    async def get_productivity_analytics(self, user_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        entries = await self.get_user_time_entries(user_id, start_date, end_date)
        total_time = sum(entry["duration"] or 0 for entry in entries)
        task_breakdown = {}
        
        for entry in entries:
            task_id = entry["task_id"]
            duration = entry["duration"] or 0
            if task_id in task_breakdown:
                task_breakdown[task_id] += duration
            else:
                task_breakdown[task_id] = duration

        productivity_score = self._calculate_productivity_score(entries)
        productivity_trend = self._calculate_productivity_trend(entries)
        task_efficiency = self._calculate_task_efficiency(task_breakdown)
        
        analytics = {
            "total_time": total_time,
            "task_breakdown": task_breakdown,
            "num_entries": len(entries),
            "avg_entry_duration": total_time / len(entries) if entries else 0,
            "productivity_score": productivity_score,
            "productivity_trend": productivity_trend,
            "task_efficiency": task_efficiency
        }

        insights = await self._generate_productivity_insights(analytics)
        analytics["insights"] = insights

        return analytics

    def _calculate_productivity_score(self, entries: List[Dict[str, Any]]) -> float:
        total_time = sum(entry["duration"] or 0 for entry in entries)
        num_entries = len(entries)
        if num_entries == 0:
            return 0
        avg_duration = total_time / num_entries
        return min(100, (avg_duration / 3600) * 100)  # Assuming 1 hour of focused work is optimal

    def _calculate_productivity_trend(self, entries: List[Dict[str, Any]]) -> str:
        if len(entries) < 2:
            return "Not enough data"
        
        sorted_entries = sorted(entries, key=lambda x: x["start_time"])
        first_half = sorted_entries[:len(sorted_entries)//2]
        second_half = sorted_entries[len(sorted_entries)//2:]
        
        first_half_score = self._calculate_productivity_score(first_half)
        second_half_score = self._calculate_productivity_score(second_half)
        
        if second_half_score > first_half_score:
            return "Improving"
        elif second_half_score < first_half_score:
            return "Declining"
        else:
            return "Stable"

    def _calculate_task_efficiency(self, task_breakdown: Dict[str, float]) -> Dict[str, float]:
        total_time = sum(task_breakdown.values())
        return {task: (duration / total_time) * 100 for task, duration in task_breakdown.items()}

    async def _generate_productivity_insights(self, analytics: Dict[str, Any]) -> str:
        prompt = f"""
        Analyze the following productivity data and provide detailed insights and recommendations:

        Total study time: {analytics['total_time'] / 3600:.2f} hours
        Number of study sessions: {analytics['num_entries']}
        Average session duration: {analytics['avg_entry_duration'] / 60:.2f} minutes
        Productivity score: {analytics['productivity_score']:.2f}/100
        Productivity trend: {analytics['productivity_trend']}

        Task breakdown:
        {analytics['task_breakdown']}

        Task efficiency:
        {analytics['task_efficiency']}

        Please provide:
        1. An overall assessment of the user's productivity
        2. Detailed analysis of the productivity trend and its implications
        3. Recommendations for improving study habits and time management
        4. Insights on the most and least efficient tasks
        5. Suggestions for optimizing task allocation and focus
        6. Tips for maintaining and improving productivity over time
        7. Potential areas for skill development based on task efficiency
        """

        insights = self.llm_orchestrator.process_request([
            {"role": "system", "content": "You are an AI productivity coach specializing in student performance analysis and time management."},
            {"role": "user", "content": prompt}
        ], "high")

        return insights

def get_time_tracking_service(llm_orchestrator: LLMOrchestrator = get_llm_orchestrator()) -> TimeTrackingService:
    return TimeTrackingService(llm_orchestrator)
