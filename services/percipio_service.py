import os
import asyncio
from typing import List, Dict, Any, Optional
import logging
import time
import json
from urllib.parse import urlencode

class PercipioService:
    def __init__(self):
        self.api_key = os.getenv("PERCIPIO_API_KEY")
        self.base_url = "https://api.percipio.com/v1"  # Replace with actual Percipio API URL
        self.cache = {}  # Simple dictionary cache
        self.cache_ttl = 3600  # Cache TTL in seconds (1 hour)
        self.logger = logging.getLogger(__name__)
        self.last_request_time = 0
        self.request_interval = 6  # 10 requests per minute = 1 request per 6 seconds

    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Simple rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_interval:
            await asyncio.sleep(self.request_interval - time_since_last_request)
        self.last_request_time = time.time()

        if params:
            url += "?" + urlencode(params)

        try:
            reader, writer = await asyncio.open_connection(self.base_url.split("//")[1], 443, ssl=True)
            
            request = f"{method} {endpoint} HTTP/1.1\r\n"
            request += f"Host: {self.base_url.split('//')[1]}\r\n"
            for header, value in headers.items():
                request += f"{header}: {value}\r\n"
            
            if data:
                body = json.dumps(data)
                request += f"Content-Length: {len(body)}\r\n"
            
            request += "\r\n"
            
            if data:
                request += body

            writer.write(request.encode())
            await writer.drain()

            response = await reader.read()
            writer.close()
            await writer.wait_closed()

            # Parse the response
            response_parts = response.split(b'\r\n\r\n', 1)
            headers = response_parts[0].decode()
            body = response_parts[1] if len(response_parts) > 1 else None

            if body:
                return json.loads(body)
            else:
                return {}

        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")
            raise

    async def get_courses(self, limit: int, offset: int) -> List[Dict[str, Any]]:
        cache_key = f"courses_{limit}_{offset}"
        if cache_key in self.cache and time.time() - self.cache[cache_key]['timestamp'] < self.cache_ttl:
            self.logger.info(f"Returning cached courses for {cache_key}")
            return self.cache[cache_key]['data']

        params = {"limit": limit, "offset": offset}
        try:
            response = await self._make_request("GET", "/courses", params=params)
            self.cache[cache_key] = {'data': response["courses"], 'timestamp': time.time()}
            return response["courses"]
        except Exception as e:
            self.logger.error(f"Error fetching courses: {str(e)}")
            raise

    async def get_course_details(self, course_id: str) -> Dict[str, Any]:
        cache_key = f"course_details_{course_id}"
        if cache_key in self.cache and time.time() - self.cache[cache_key]['timestamp'] < self.cache_ttl:
            self.logger.info(f"Returning cached course details for {course_id}")
            return self.cache[cache_key]['data']

        try:
            response = await self._make_request("GET", f"/courses/{course_id}")
            self.cache[cache_key] = {'data': response, 'timestamp': time.time()}
            return response
        except Exception as e:
            self.logger.error(f"Error fetching course details: {str(e)}")
            raise

    async def start_course(self, user_id: str, course_id: str) -> Dict[str, Any]:
        try:
            data = {"user_id": user_id, "course_id": course_id}
            return await self._make_request("POST", f"/courses/{course_id}/start", data=data)
        except Exception as e:
            self.logger.error(f"Error starting course: {str(e)}")
            raise

    async def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        cache_key = f"user_progress_{user_id}"
        if cache_key in self.cache and time.time() - self.cache[cache_key]['timestamp'] < self.cache_ttl:
            self.logger.info(f"Returning cached user progress for {user_id}")
            return self.cache[cache_key]['data']

        try:
            response = await self._make_request("GET", f"/users/{user_id}/progress")
            self.cache[cache_key] = {'data': response, 'timestamp': time.time()}
            return response
        except Exception as e:
            self.logger.error(f"Error fetching user progress: {str(e)}")
            raise

    async def search_content(self, query: str, content_type: Optional[str], limit: int) -> List[Dict[str, Any]]:
        cache_key = f"search_{query}_{content_type}_{limit}"
        if cache_key in self.cache and time.time() - self.cache[cache_key]['timestamp'] < self.cache_ttl:
            self.logger.info(f"Returning cached search results for {cache_key}")
            return self.cache[cache_key]['data']

        params = {"query": query, "limit": limit}
        if content_type:
            params["content_type"] = content_type

        try:
            response = await self._make_request("GET", "/search", params=params)
            self.cache[cache_key] = {'data': response["results"], 'timestamp': time.time()}
            return response["results"]
        except Exception as e:
            self.logger.error(f"Error searching content: {str(e)}")
            raise

    async def get_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        cache_key = f"recommendations_{user_id}"
        if cache_key in self.cache and time.time() - self.cache[cache_key]['timestamp'] < self.cache_ttl:
            self.logger.info(f"Returning cached recommendations for {user_id}")
            return self.cache[cache_key]['data']

        try:
            response = await self._make_request("GET", f"/users/{user_id}/recommendations")
            self.cache[cache_key] = {'data': response["recommendations"], 'timestamp': time.time()}
            return response["recommendations"]
        except Exception as e:
            self.logger.error(f"Error fetching recommendations: {str(e)}")
            raise

percipio_service = PercipioService()

def get_percipio_service() -> PercipioService:
    return percipio_service
