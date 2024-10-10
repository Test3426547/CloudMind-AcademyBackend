from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class CertificateBase(BaseModel):
    course_id: str
    issue_date: datetime

class CertificateCreate(CertificateBase):
    pass

class Certificate(CertificateBase):
    id: str
    user_id: str
    hash: str
    revoked: bool = False

    class Config:
        orm_mode = True
