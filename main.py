from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from auth_config import get_supabase_client
from supabase import Client
from routers import (
    notification,
    analytics,
    plagiarism,
    ai_tutor,
    percipio,
    video_content,
    live_streaming,
    course_feedback,
    time_tracking,
    course_prerequisite,
    data_visualization,
    openrouter,
    ai_model_training,
    v0dev,
    translation,
    ar_vr,
    emotion_analysis,
    grading  # Add this line
)

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    supabase: Client = get_supabase_client()
    try:
        user = supabase.auth.get_user(token)
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

app.include_router(notification.router)
app.include_router(analytics.router)
app.include_router(plagiarism.router)
app.include_router(ai_tutor.router)
app.include_router(percipio.router)
app.include_router(video_content.router)
app.include_router(live_streaming.router)
app.include_router(course_feedback.router)
app.include_router(time_tracking.router)
app.include_router(course_prerequisite.router)
app.include_router(data_visualization.router)
app.include_router(openrouter.router)
app.include_router(ai_model_training.router)
app.include_router(v0dev.router)
app.include_router(translation.router)
app.include_router(ar_vr.router)
app.include_router(emotion_analysis.router)
app.include_router(grading.router)  # Add this line

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
