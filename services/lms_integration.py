from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import aiohttp
from fastapi.security import OAuth2PasswordBearer

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class LMSCourse(BaseModel):
    id: str
    title: str
    description: Optional[str] = None

class LMSIntegration:
    def __init__(self, lms_type: str, api_key: str, base_url: str):
        self.lms_type = lms_type
        self.api_key = api_key
        self.base_url = base_url

    @staticmethod
    def create(lms_type: str, api_key: str, base_url: str):
        return LMSIntegration(lms_type, api_key, base_url)

    async def get_courses(self) -> List[LMSCourse]:
        if self.lms_type == "canvas":
            return await self._get_canvas_courses()
        elif self.lms_type == "moodle":
            return await self._get_moodle_courses()
        else:
            raise ValueError(f"Unsupported LMS type: {self.lms_type}")

    async def _get_canvas_courses(self) -> List[LMSCourse]:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/v1/courses", headers={"Authorization": f"Bearer {self.api_key}"}) as response:
                if response.status == 200:
                    data = await response.json()
                    return [LMSCourse(id=str(course['id']), title=course['name'], description=course.get('public_description')) for course in data]
                else:
                    raise HTTPException(status_code=response.status, detail="Failed to fetch Canvas courses")

    async def _get_moodle_courses(self) -> List[LMSCourse]:
        params = {
            'wstoken': self.api_key,
            'wsfunction': 'core_course_get_courses',
            'moodlewsrestformat': 'json'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/webservice/rest/server.php", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [LMSCourse(id=str(course['id']), title=course['fullname'], description=course.get('summary')) for course in data]
                else:
                    raise HTTPException(status_code=response.status, detail="Failed to fetch Moodle courses")

    async def sync_course(self, course_id: str) -> bool:
        if self.lms_type == "canvas":
            return await self._sync_canvas_course(course_id)
        elif self.lms_type == "moodle":
            return await self._sync_moodle_course(course_id)
        else:
            raise ValueError(f"Unsupported LMS type: {self.lms_type}")

    async def _sync_canvas_course(self, course_id: str) -> bool:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/v1/courses/{course_id}", headers={"Authorization": f"Bearer {self.api_key}"}) as response:
                if response.status == 200:
                    course_data = await response.json()
                    # Here you would typically save or update the course data in your own database
                    print(f"Synced Canvas course: {course_data['name']}")
                    return True
                else:
                    raise HTTPException(status_code=response.status, detail=f"Failed to sync Canvas course {course_id}")

    async def _sync_moodle_course(self, course_id: str) -> bool:
        params = {
            'wstoken': self.api_key,
            'wsfunction': 'core_course_get_courses',
            'courseids[]': course_id,
            'moodlewsrestformat': 'json'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/webservice/rest/server.php", params=params) as response:
                if response.status == 200:
                    course_data = await response.json()
                    if course_data and not isinstance(course_data, dict):  # Moodle returns a list
                        # Here you would typically save or update the course data in your own database
                        print(f"Synced Moodle course: {course_data[0]['fullname']}")
                        return True
                    else:
                        raise HTTPException(status_code=404, detail=f"Moodle course {course_id} not found")
                else:
                    raise HTTPException(status_code=response.status, detail=f"Failed to sync Moodle course {course_id}")

@router.post("/lms/integrate")
async def integrate_lms(lms_type: str, api_key: str, base_url: str):
    integration = LMSIntegration.create(lms_type, api_key, base_url)
    return {"message": f"Integration with {lms_type} LMS successful"}

@router.get("/lms/courses")
async def get_lms_courses(lms_type: str, api_key: str, base_url: str):
    integration = LMSIntegration.create(lms_type, api_key, base_url)
    courses = await integration.get_courses()
    return {"courses": courses}

@router.post("/lms/sync_course/{course_id}")
async def sync_lms_course(lms_type: str, api_key: str, base_url: str, course_id: str):
    integration = LMSIntegration.create(lms_type, api_key, base_url)
    success = await integration.sync_course(course_id)
    if success:
        return {"message": f"Course {course_id} synced successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to sync course")
