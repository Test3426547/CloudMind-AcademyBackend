from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Dict
import subprocess
import tempfile
import os

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class CodeExecutionRequest(BaseModel):
    code: str
    language: str

class CodeExecutionResponse(BaseModel):
    output: str
    errors: str

@router.post("/virtual-lab/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    if request.language not in ["python", "javascript"]:
        raise HTTPException(status_code=400, detail="Unsupported programming language")

    with tempfile.NamedTemporaryFile(mode="w+", suffix=f".{request.language}", delete=False) as temp_file:
        temp_file.write(request.code)
        temp_file.flush()

    try:
        if request.language == "python":
            result = subprocess.run(["python3", temp_file.name], capture_output=True, text=True, timeout=5)
        elif request.language == "javascript":
            result = subprocess.run(["node", temp_file.name], capture_output=True, text=True, timeout=5)

        return CodeExecutionResponse(
            output=result.stdout,
            errors=result.stderr
        )
    except subprocess.TimeoutExpired:
        return CodeExecutionResponse(
            output="",
            errors="Execution timed out"
        )
    finally:
        os.unlink(temp_file.name)
