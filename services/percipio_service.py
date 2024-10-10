import os
import requests
from typing import Dict, List, Any

class PercipioService:
    def __init__(self):
        self.base_url = "https://api.percipio.com/v1"  # Replace with the actual Percipio API base URL
        self.api_key = os.getenv("PERCIPIO_API_KEY")
        self.org_id = os.getenv("PERCIPIO_ORG_ID")
        self.email = "ogovender@emmconsulting.com.au"  # Add the provided email

    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Percipio-Email": self.email  # Include the email in the request headers
        }
        url = f"{self.base_url}/{endpoint}"
        response = requests.request(method, url, headers=headers, params=params, json=data)
        response.raise_for_status()
        return response.json()

    def get_courses(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        endpoint = f"organizations/{self.org_id}/content-discovery/v2/search"
        params = {
            "limit": limit,
            "offset": offset,
            "contentTypes": "course"
        }
        return self._make_request("GET", endpoint, params=params)

    def get_course_details(self, course_id: str) -> Dict[str, Any]:
        endpoint = f"organizations/{self.org_id}/content-discovery/v2/content/{course_id}"
        return self._make_request("GET", endpoint)

    def start_course(self, user_id: str, course_id: str) -> Dict[str, Any]:
        endpoint = f"organizations/{self.org_id}/users/{user_id}/learning-activity"
        data = {
            "contentId": course_id,
            "action": "START"
        }
        return self._make_request("POST", endpoint, data=data)

    def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        endpoint = f"organizations/{self.org_id}/users/{user_id}/learning-activity"
        return self._make_request("GET", endpoint)

percipio_service = PercipioService()

def get_percipio_service() -> PercipioService:
    return percipio_service
