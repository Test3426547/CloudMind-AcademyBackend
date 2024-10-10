import aiohttp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

router = APIRouter()

class PercipioConfig(BaseModel):
    api_key: Optional[str] = os.environ.get("PERCIPIO_API_KEY")
    org_id: Optional[str] = os.environ.get("PERCIPIO_ORG_ID")
    base_url: str = "https://api.percipio.com/content-discovery/v1"

class PercipioCourse(BaseModel):
    id: str
    title: str
    description: Optional[str] = None

class PercipioIntegration:
    def __init__(self, config: PercipioConfig):
        self.config = config

    def credentials_available(self) -> bool:
        return self.config.api_key is not None and self.config.org_id is not None

    def get_mock_courses(self) -> List[PercipioCourse]:
        return [
            PercipioCourse(id="1", title="Introduction to Python", description="Learn the basics of Python programming"),
            PercipioCourse(id="2", title="Web Development Fundamentals", description="Understanding HTML, CSS, and JavaScript"),
            PercipioCourse(id="3", title="Data Science Essentials", description="Explore data analysis and machine learning concepts")
        ]

    async def get_courses(self) -> List[PercipioCourse]:
        if not self.credentials_available():
            return self.get_mock_courses()

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "x-org-id": self.config.org_id
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.config.base_url}/courses", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return [PercipioCourse(id=course['id'], title=course['title'], description=course.get('description')) for course in data['courses']]
                else:
                    raise HTTPException(status_code=response.status, detail="Failed to fetch Percipio courses")

    def update_credentials(self, api_key: str, org_id: str):
        self.config.api_key = api_key
        self.config.org_id = org_id

@router.get("/percipio/courses")
async def get_percipio_courses():
    config = PercipioConfig()
    integration = PercipioIntegration(config)
    courses = await integration.get_courses()
    return {"courses": courses}

@router.post("/percipio/update-credentials")
async def update_percipio_credentials(api_key: str, org_id: str):
    config = PercipioConfig()
    integration = PercipioIntegration(config)
    integration.update_credentials(api_key, org_id)
    return {"message": "Percipio credentials updated successfully"}
