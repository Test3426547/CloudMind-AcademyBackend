from dotenv import load_dotenv
load_dotenv()  # This should be at the very top of main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
# Rest of your imports and code...
from supabase import Client
from auth_config import get_supabase_client
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
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
    grading,
    lms_integration,
    supabase,
    text_embedding,
    voice_recognition,
    web_scraping,
    certificate,
    course,
    exam,
    user,
    advanced_search  # Add this line
)
from contextlib import asynccontextmanager
from admin_config import site as admin_site

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code here (if any)
    yield
    # Shutdown code here (if any)

app = FastAPI(lifespan=lifespan)

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
app.include_router(grading.router)
app.include_router(lms_integration.router)
app.include_router(supabase.router)
app.include_router(text_embedding.router)
app.include_router(voice_recognition.router)
app.include_router(web_scraping.router)
app.include_router(certificate.router)
app.include_router(course.router)
app.include_router(exam.router)
app.include_router(user.router)
app.include_router(advanced_search.router)  # Add this line

@app.get("/")
async def root():
    return {"message": "Welcome to CloudMind Academy Backend"}

@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())

# Mount the admin site
app.mount("/admin", admin_site.app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
