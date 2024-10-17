from fastapi_amis_admin.admin.site import AdminSite
from fastapi_amis_admin.admin.settings import Settings
from fastapi_amis_admin.admin.admin import AdminApp
from fastapi_amis_admin.models import Field
from fastapi_amis_admin.amis.components import Form
from sqlmodel import SQLModel
from datetime import datetime

# Create Settings instance
settings = Settings(
    database_url_async="sqlite+aiosqlite:///amisadmin.db",
    debug=True,
    site_title="CloudMind Academy Admin"
)

# Create AdminSite instance with settings
site = AdminSite(settings=settings)

# Define your SQLModel classes
class AIQuestion(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    question: str = Field(title="Question")
    answer: str = Field(title="Answer")
    category: str = Field(title="Category")

class UserProgress(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(title="User ID")
    course_id: int = Field(title="Course ID")
    progress: float = Field(title="Progress")
    last_activity: datetime = Field(default=datetime.utcnow)

# Create AdminApps for each model
class AIQuestionAdmin(AdminApp):
    page_schema = "AI Questions"
    model = AIQuestion
    
    form = Form(
        body=[
            {"type": "input-text", "name": "question", "label": "Question"},
            {"type": "input-text", "name": "answer", "label": "Answer"},
            {"type": "input-text", "name": "category", "label": "Category"},
        ]
    )

class UserProgressAdmin(AdminApp):
    page_schema = "User Progress"
    model = UserProgress
    
    form = Form(
        body=[
            {"type": "input-number", "name": "user_id", "label": "User ID"},
            {"type": "input-number", "name": "course_id", "label": "Course ID"},
            {"type": "input-number", "name": "progress", "label": "Progress"},
            {"type": "input-datetime", "name": "last_activity", "label": "Last Activity"},
        ]
    )

# Register the AdminApps with the site
site.register_admin(AIQuestionAdmin)
site.register_admin(UserProgressAdmin)

# In your AITutor class:
class AITutor:
    def __init__(self):
        self.client = InferenceClient(
            model=os.environ["MODEL_NAME"],
            token=os.environ["HF_API_TOKEN"]
        )

    async def answer_question(self, question, context):
        async with site.db_session() as session:
            new_question = AIQuestion(question=question)
            session.add(new_question)
            await session.commit()

            # Generate answer using your HuggingFace model
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            answer = await self.generate_response(prompt)

            new_question.answer = answer
            await session.commit()

        return answer

    async def generate_response(self, prompt):
        # Use your HuggingFace model here
        response = await self.client.text_generation(
            prompt,
            max_new_tokens=int(os.environ["MAX_NEW_TOKENS"]),
            temperature=float(os.environ["TEMPERATURE"]),
        )
        return response

# In your main.py file, add:
# from admin_config import site as admin_site
# app.mount("/admin", admin_site.app)
