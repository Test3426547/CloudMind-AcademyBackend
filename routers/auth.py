from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from services.supabase_client import supabase_client
from models.user import User, UserCreate

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/register")
async def register(user: UserCreate):
    try:
        response = supabase_client.auth.sign_up({
            "email": user.email,
            "password": user.password,
        })
        return {"message": "User registered successfully", "user_id": response.user.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login")
async def login(user: UserCreate):
    try:
        response = supabase_client.auth.sign_in_with_password({
            "email": user.email,
            "password": user.password,
        })
        return {"access_token": response.session.access_token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@router.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    try:
        supabase_client.auth.sign_out(token)
        return {"message": "Logged out successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/me", response_model=User)
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        user = supabase_client.auth.get_user(token)
        return User(**user.dict())
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")
