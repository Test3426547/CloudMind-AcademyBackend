from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, ai_tutor, quiz, code_sandbox, gamification, learning_path, analytics, video_content, plagiarism, virtual_lab, ar_vr, social, web_scraping, collaboration

app = FastAPI(title="CloudMind Academy", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(auth.router, prefix="/api/v1")
app.include_router(ai_tutor.router, prefix="/api/v1")
app.include_router(quiz.router, prefix="/api/v1")
app.include_router(code_sandbox.router, prefix="/api/v1")
app.include_router(gamification.router, prefix="/api/v1")
app.include_router(learning_path.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")
app.include_router(video_content.router, prefix="/api/v1")
app.include_router(plagiarism.router, prefix="/api/v1")
app.include_router(virtual_lab.router, prefix="/api/v1")
app.include_router(ar_vr.router, prefix="/api/v1")
app.include_router(social.router, prefix="/api/v1")
app.include_router(web_scraping.router, prefix="/api/v1")
app.include_router(collaboration.router, prefix="/api/v1")  # Add the new collaboration router

@app.get("/")
async def root():
    return {"message": "Welcome to CloudMind Academy"}

@app.get("/routes")
def get_routes():
    routes = []
    for route in app.routes:
        routes.append(f"{route.methods} {route.path}")
    return {"routes": routes}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
