from pydantic import BaseModel
from typing import List, Optional

class Exam(BaseModel):
    id: Optional[str] = None
    title: str
    description: str
    duration: int  # in minutes
    passing_score: float
    questions: List[str]  # List of question IDs

class ExamCreate(BaseModel):
    title: str
    description: str
    duration: int
    passing_score: float
    questions: List[str]
