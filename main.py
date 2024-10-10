import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, ai_tutor, quiz, code_sandbox, gamification, learning_path, analytics, video_content, plagiarism, virtual_lab, ar_vr, social, web_scraping, collaboration
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

@app.get("/")
async def root():
    return {"message": "Welcome to CloudMind Academy"}

@app.get("/routes")
def get_routes():
    routes = []
    for route in app.routes:
        routes.append(f"{route.methods} {route.path}")
    return {"routes": routes}

async def test_collaboration_tool():
    await asyncio.sleep(5)  # Wait for the server to start
    base_url = "http://localhost:8000/api/v1"
    
    # Create a collaboration session
    create_session_url = f"{base_url}/collaboration/create-session"
    create_session_data = {"participants": ["user1", "user2"]}
    create_session_response = requests.post(create_session_url, json=create_session_data)
    session_id = create_session_response.json()["session_id"]
    print(f"Created session: {session_id}")
    
    # Send a message to the session
    send_message_url = f"{base_url}/collaboration/{session_id}/send-message"
    send_message_data = {"message": "Hello, can someone help me understand the concept of machine learning?"}
    send_message_response = requests.post(send_message_url, json=send_message_data)
    print(f"Sent message response: {json.dumps(send_message_response.json(), indent=2)}")
    
    # Get messages from the session
    get_messages_url = f"{base_url}/collaboration/{session_id}/messages"
    get_messages_response = requests.get(get_messages_url)
    print(f"Retrieved messages: {json.dumps(get_messages_response.json(), indent=2)}")
    
    # Invite a new participant
    invite_url = f"{base_url}/collaboration/{session_id}/invite"
    invite_data = {"new_participant": "user3"}
    invite_response = requests.post(invite_url, json=invite_data)
    print(f"Invite response: {json.dumps(invite_response.json(), indent=2)}")
    
    # End the collaboration session
    end_session_url = f"{base_url}/collaboration/{session_id}"
    end_session_response = requests.delete(end_session_url)
    print(f"End session response: {json.dumps(end_session_response.json(), indent=2)}")

async def start_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(test_collaboration_tool())
    loop.run_until_complete(start_server())
