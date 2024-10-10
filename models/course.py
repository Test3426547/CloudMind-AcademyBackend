from pydantic import BaseModel
from typing import List, Optional

class Course(BaseModel):
    id: Optional[str] = None
    title: str
    description: str
    modules: List[str]
    difficulty: str
    duration: int  # in minutes

class CourseCreate(BaseModel):
    title: str
    description: str
    modules: List[str]
    difficulty: str
    duration: int
