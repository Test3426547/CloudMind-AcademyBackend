from typing import List, Optional
from models.video_content import VideoContent, VideoContentCreate

class VideoContentService:
    def __init__(self):
        # This would be replaced with a database connection in a real implementation
        self.video_contents = []

    async def create_video_content(self, video_content: VideoContentCreate) -> VideoContent:
        # In a real implementation, this would save to a database
        new_id = str(len(self.video_contents) + 1)
        video_content_dict = video_content.dict()
        video_content_dict['id'] = new_id
        new_video_content = VideoContent(**video_content_dict)
        self.video_contents.append(new_video_content)
        return new_video_content

    async def get_video_content(self, video_id: str) -> Optional[VideoContent]:
        # In a real implementation, this would fetch from a database
        for video in self.video_contents:
            if video.id == video_id:
                return video
        return None

    async def list_video_contents(self, skip: int = 0, limit: int = 10) -> List[VideoContent]:
        # In a real implementation, this would use database pagination
        return self.video_contents[skip : skip + limit]

    async def search_video_contents(self, query: str) -> List[VideoContent]:
        # In a real implementation, this would use database full-text search
        return [video for video in self.video_contents if query.lower() in video.title.lower() or query.lower() in video.description.lower()]

    async def update_video_content(self, video_id: str, video_content: VideoContentCreate) -> Optional[VideoContent]:
        # In a real implementation, this would update the database
        for i, video in enumerate(self.video_contents):
            if video.id == video_id:
                updated_video = VideoContent(id=video_id, **video_content.dict())
                self.video_contents[i] = updated_video
                return updated_video
        return None

    async def delete_video_content(self, video_id: str) -> bool:
        # In a real implementation, this would delete from the database
        for i, video in enumerate(self.video_contents):
            if video.id == video_id:
                del self.video_contents[i]
                return True
        return False

video_content_service = VideoContentService()

def get_video_content_service() -> VideoContentService:
    return video_content_service
