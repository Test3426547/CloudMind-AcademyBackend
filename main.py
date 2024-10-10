import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, ai_tutor, quiz, code_sandbox, gamification, learning_path, analytics, video_content, plagiarism, virtual_lab, ar_vr, social, web_scraping, collaboration, certificate
import uvicorn
import requests
import json

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
app.include_router(collaboration.router, prefix="/api/v1")
app.include_router(certificate.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to CloudMind Academy"}

@app.get("/routes")
def get_routes():
    routes = []
    for route in app.routes:
        routes.append(f"{route.methods} {route.path}")
    return {"routes": routes}

def test_certificate_functionality():
    base_url = "http://localhost:8000/api/v1"
    
    # Create a certificate
    create_certificate_url = f"{base_url}/certificates"
    create_certificate_data = {
        "course_id": "course123",
        "user_id": "user456",
        "issue_date": "2024-10-10T12:00:00"
    }
    create_certificate_response = requests.post(create_certificate_url, json=create_certificate_data)
    certificate = create_certificate_response.json()
    print(f"Created certificate: {json.dumps(certificate, indent=2)}")
    
    # Get the created certificate
    get_certificate_url = f"{base_url}/certificates/{certificate['id']}"
    get_certificate_response = requests.get(get_certificate_url)
    print(f"Retrieved certificate: {json.dumps(get_certificate_response.json(), indent=2)}")
    
    # Verify the certificate
    verify_certificate_url = f"{base_url}/certificates/{certificate['id']}/verify"
    verify_certificate_data = {"certificate_hash": certificate['hash']}
    verify_certificate_response = requests.post(verify_certificate_url, json=verify_certificate_data)
    print(f"Verify certificate response: {json.dumps(verify_certificate_response.json(), indent=2)}")
    
    # Revoke the certificate
    revoke_certificate_url = f"{base_url}/certificates/{certificate['id']}/revoke"
    revoke_certificate_response = requests.post(revoke_certificate_url)
    print(f"Revoke certificate response: {json.dumps(revoke_certificate_response.json(), indent=2)}")
    
    # Try to verify the revoked certificate
    verify_revoked_response = requests.post(verify_certificate_url, json=verify_certificate_data)
    print(f"Verify revoked certificate response: {json.dumps(verify_revoked_response.json(), indent=2)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    test_certificate_functionality()
