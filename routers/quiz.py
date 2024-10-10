from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from services.openai_client import send_openai_request
from models.quiz import Quiz, QuizCreate
from models.user import User

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/quiz/generate")
async def generate_quiz(topic: str, num_questions: int, user: User = Depends(oauth2_scheme)):
    try:
        prompt = f"Generate a quiz with {num_questions} multiple-choice questions about {topic}. Format the response as a JSON object with 'title' and 'questions' fields. Each question should have 'text', 'options', and 'correct_answer' fields."
        quiz_data = send_openai_request(prompt)
        return Quiz(**quiz_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quiz/create")
async def create_quiz(quiz: QuizCreate, user: User = Depends(oauth2_scheme)):
    # Here you would typically save the quiz to your database
    # For this example, we'll just return the created quiz
    return Quiz(**quiz.dict(), id="generated_id")

@router.get("/quiz/{quiz_id}")
async def get_quiz(quiz_id: str, user: User = Depends(oauth2_scheme)):
    # Here you would typically fetch the quiz from your database
    # For this example, we'll return a mock quiz
    return Quiz(id=quiz_id, title="Sample Quiz", questions=[])
