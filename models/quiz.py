from pydantic import BaseModel
from typing import List, Optional

class Question(BaseModel):
    id: Optional[str] = None
    text: str
    options: List[str]
    correct_answer: int

class Quiz(BaseModel):
    id: Optional[str] = None
    title: str
    questions: List[Question]
    course_id: str

class QuizCreate(BaseModel):
    title: str
    questions: List[Question]
    course_id: str
