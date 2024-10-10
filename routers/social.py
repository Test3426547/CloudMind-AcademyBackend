from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from typing import Dict, Any
import uuid

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock database for social sharing
shared_content = {}

@router.post("/social/share")
async def share_content(content_type: str, content_id: str, platform: str, user: User = Depends(oauth2_scheme)) -> Dict[str, str]:
    share_id = str(uuid.uuid4())
    shared_content[share_id] = {
        "user_id": user.id,
        "content_type": content_type,
        "content_id": content_id,
        "platform": platform
    }
    return {"message": f"Content shared on {platform}", "share_id": share_id}

@router.get("/social/shared/{share_id}")
async def get_shared_content(share_id: str, user: User = Depends(oauth2_scheme)) -> Dict[str, Any]:
    if share_id not in shared_content:
        raise HTTPException(status_code=404, detail="Shared content not found")
    return shared_content[share_id]

@router.post("/social/like")
async def like_content(content_type: str, content_id: str, user: User = Depends(oauth2_scheme)) -> Dict[str, str]:
    # In a real implementation, you would update a database to record the like
    return {"message": f"Content {content_id} liked by user {user.id}"}

@router.post("/social/comment")
async def comment_on_content(content_type: str, content_id: str, comment: str, user: User = Depends(oauth2_scheme)) -> Dict[str, str]:
    # In a real implementation, you would save the comment to a database
    return {"message": f"Comment added to content {content_id} by user {user.id}"}
