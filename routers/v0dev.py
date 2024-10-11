from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.v0dev_service import V0DevService, get_v0dev_service
from typing import Dict
from pydantic import BaseModel

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class UIGenerationRequest(BaseModel):
    prompt: str

@router.post("/v0dev/generate-ui")
async def generate_ui_component(
    request: UIGenerationRequest,
    user: User = Depends(oauth2_scheme),
    v0dev_service: V0DevService = Depends(get_v0dev_service)
):
    try:
        result = await v0dev_service.generate_ui_component(request.prompt)
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
