from pydantic import BaseModel, EmailStr
from typing import Optional

class User(BaseModel):
    id: Optional[str] = None
    email: EmailStr
    full_name: str
    is_active: bool = True
    is_enterprise: bool = False

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
