from pydantic import BaseModel
from typing import Optional, List

class VideoContent(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    duration: int  # in seconds
    url: str
    tags: List[str] = []
    provider: str  # This could be 'percipio' or any other provider in the future

class VideoContentCreate(BaseModel):
    title: str
    description: Optional[str] = None
    duration: int
    url: str
    tags: List[str] = []
    provider: str
