from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from models.user import User
from services.ai_tutor import AITutorService, get_ai_tutor_service
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging
from sqlalchemy.orm import Session
from auth_config import get_db
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import asyncio
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)

# Load pre-trained language model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
language_model = AutoModel.from_pretrained("bert-base-uncased")

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)

class ConceptExplanationRequest(BaseModel):
    concept: str = Field(..., min_length=1, max_length=100)

class LearningPathRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=100)
    current_level: str = Field(..., min_length=1, max_length=50)

class PerformanceData(BaseModel):
    grades: Dict[str, float]
    completed_courses: List[str]
    time_spent: Dict[str, int]

class UserDataset(Dataset):
    def __init__(self, user_data):
        self.user_data = user_data
        self.scaler = StandardScaler()
        self.user_data_scaled = self.scaler.fit_transform(user_data)

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, idx):
        return torch.tensor(self.user_data_scaled[idx], dtype=torch.float32)

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attention_weights = self.softmax(self.attention(x))
        return torch.sum(attention_weights * x, dim=1)

class UserEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_layers=3):
        super(UserEmbedding, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim if i == 0 else embedding_dim, embedding_dim) for i in range(num_layers)])
        self.activation = nn.ReLU()
        self.attention = AttentionLayer(embedding_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.attention(x)

user_embedding_model = UserEmbedding(10, 50)  # Assuming 10 input features and 50-dimensional embedding
optimizer = optim.Adam(user_embedding_model.parameters(), lr=0.001)

async def train_user_embedding_model(user_data):
    dataset = UserDataset(user_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(10):  # 10 epochs for example
        for batch in dataloader:
            optimizer.zero_grad()
            output = user_embedding_model(batch)
            loss = nn.MSELoss()(output, torch.zeros_like(output))  # Example loss, adjust as needed
            loss.backward()
            optimizer.step()

@router.post("/ai-tutor/chat")
async def chat_with_tutor(
    chat_message: ChatMessage,
    background_tasks: BackgroundTasks,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service),
    db: Session = Depends(get_db)
):
    try:
        # Get user data and create embedding
        user_data = get_user_data(user.id, db)  # Implement this function to fetch user data
        user_embedding = user_embedding_model(torch.tensor(user_data, dtype=torch.float32).unsqueeze(0))
        
        # Tokenize and encode the chat message
        inputs = tokenizer(chat_message.message, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = language_model(**inputs)
        
        # Combine user embedding with message encoding
        combined_embedding = torch.cat([user_embedding, outputs.last_hidden_state.mean(dim=1)], dim=1)
        
        result = await ai_tutor_service.chat_with_tutor(user.id, chat_message.message, db, combined_embedding)
        
        # Asynchronously update user statistics and retrain the embedding model
        background_tasks.add_task(update_user_stats, user.id, db)
        background_tasks.add_task(train_user_embedding_model, get_all_user_data(db))
        
        logger.info(f"AI Tutor chat completed for user {user.id}")
        return result
    except HTTPException as e:
        logger.error(f"HTTP error in chat_with_tutor: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in chat_with_tutor: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during the chat")

@router.post("/ai-tutor/explain-concept")
async def explain_concept(
    request: ConceptExplanationRequest,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service),
):
    try:
        # Encode the concept using the language model
        inputs = tokenizer(request.concept, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            concept_encoding = language_model(**inputs).last_hidden_state.mean(dim=1)
        
        result = await ai_tutor_service.explain_concept(request.concept, concept_encoding)
        logger.info(f"Concept explanation completed for user {user.id}")
        
        # Use pandas to structure and analyze the explanation data
        df = pd.DataFrame({
            'concept': [request.concept],
            'explanation': [result['explanation']],
            'quiz_questions': [result['quiz_questions']],
            'mind_map': [result['mind_map']]
        })
        
        # Perform some analysis on the explanation
        explanation_length = len(result['explanation'].split())
        num_quiz_questions = len(result['quiz_questions'])
        complexity_score = calculate_complexity_score(result['explanation'])  # Implement this function
        
        df['explanation_length'] = explanation_length
        df['num_quiz_questions'] = num_quiz_questions
        df['complexity_score'] = complexity_score
        
        # Convert DataFrame to dict for JSON serialization
        return df.to_dict(orient='records')[0]
    except HTTPException as e:
        logger.error(f"HTTP error in explain_concept: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in explain_concept: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while explaining the concept")

@router.post("/ai-tutor/generate-learning-path")
async def generate_learning_path(
    request: LearningPathRequest,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service),
    db: Session = Depends(get_db)
):
    try:
        # Get user embedding
        user_data = get_user_data(user.id, db)
        user_embedding = user_embedding_model(torch.tensor(user_data, dtype=torch.float32).unsqueeze(0))
        
        # Encode the subject and current level
        subject_encoding = encode_text(request.subject)
        level_encoding = encode_text(request.current_level)
        
        # Combine all encodings
        combined_encoding = torch.cat([user_embedding, subject_encoding, level_encoding], dim=1)
        
        result = await ai_tutor_service.generate_personalized_learning_path(user.id, request.subject, request.current_level, combined_encoding)
        logger.info(f"Personalized learning path generated for user {user.id}")
        
        # Use numpy to create a timeline for the learning path
        timeline = np.linspace(0, 52, num=len(result['learning_path']), dtype=int)  # Assuming a year-long plan
        
        # Create a DataFrame for the learning path
        path_df = pd.DataFrame({
            'week': timeline,
            'task': result['learning_path'],
            'milestone': result['milestones']
        })
        
        # Perform some analysis on the learning path
        total_tasks = len(path_df)
        milestones_count = path_df['milestone'].notna().sum()
        avg_tasks_per_milestone = total_tasks / milestones_count if milestones_count > 0 else 0
        
        return {
            "user_id": user.id,
            "subject": request.subject,
            "current_level": request.current_level,
            "learning_path": path_df.to_dict(orient='records'),
            "analysis": {
                "total_tasks": total_tasks,
                "milestones_count": milestones_count,
                "avg_tasks_per_milestone": avg_tasks_per_milestone
            }
        }
    except HTTPException as e:
        logger.error(f"HTTP error in generate_learning_path: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in generate_learning_path: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the learning path")

@router.post("/ai-tutor/analyze-performance")
async def analyze_performance(
    performance_data: PerformanceData,
    user: User = Depends(oauth2_scheme),
    ai_tutor_service: AITutorService = Depends(get_ai_tutor_service),
    db: Session = Depends(get_db)
):
    try:
        # Convert performance data to pandas DataFrame for analysis
        grades_df = pd.DataFrame.from_dict(performance_data.grades, orient='index', columns=['grade'])
        time_spent_df = pd.DataFrame.from_dict(performance_data.time_spent, orient='index', columns=['time'])
        
        # Perform some basic analysis
        avg_grade = grades_df['grade'].mean()
        total_time = time_spent_df['time'].sum()
        grade_time_correlation = grades_df['grade'].corr(time_spent_df['time'])
        
        # Get user embedding
        user_data = get_user_data(user.id, db)
        user_embedding = user_embedding_model(torch.tensor(user_data, dtype=torch.float32).unsqueeze(0))
        
        # Combine user embedding with performance statistics
        performance_tensor = torch.tensor([avg_grade, total_time, grade_time_correlation], dtype=torch.float32).unsqueeze(0)
        combined_data = torch.cat([user_embedding, performance_tensor], dim=1)
        
        # Pass the analysis to the AI tutor service
        result = await ai_tutor_service.analyze_student_performance(
            user.id, 
            {
                "avg_grade": avg_grade,
                "total_time": total_time,
                "completed_courses": performance_data.completed_courses,
                "grade_time_correlation": grade_time_correlation
            },
            combined_data
        )
        logger.info(f"Performance analysis completed for user {user.id}")
        
        # Perform additional analysis
        strengths = grades_df[grades_df['grade'] > avg_grade].index.tolist()
        weaknesses = grades_df[grades_df['grade'] < avg_grade].index.tolist()
        time_efficiency = grades_df['grade'] / time_spent_df['time']
        most_efficient_subject = time_efficiency.idxmax()
        least_efficient_subject = time_efficiency.idxmin()
        
        # Combine the AI analysis with our statistical analysis
        result.update({
            "statistical_analysis": {
                "average_grade": avg_grade,
                "total_time_spent": total_time,
                "number_of_completed_courses": len(performance_data.completed_courses),
                "grade_time_correlation": grade_time_correlation,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "most_efficient_subject": most_efficient_subject,
                "least_efficient_subject": least_efficient_subject
            }
        })
        
        return result
    except HTTPException as e:
        logger.error(f"HTTP error in analyze_performance: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in analyze_performance: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while analyzing performance")

async def update_user_stats(user_id: int, db: Session):
    try:
        user = db.query(User).filter(User.id == user_id).first()
        user.last_activity = datetime.utcnow()
        user.chat_count += 1
        db.commit()
    except Exception as e:
        logger.error(f"Error updating user stats: {str(e)}")
        db.rollback()

@router.get("/ai-tutor/user-progress/{user_id}")
async def get_user_progress(
    user_id: int,
    db: Session = Depends(get_db)
):
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Calculate progress metrics
        days_active = (datetime.utcnow() - user.created_at).days
        avg_chats_per_day = user.chat_count / max(days_active, 1)
        
        # Get user's performance data
        performance_data = get_user_performance_data(user_id, db)  # Implement this function
        
        # Calculate learning velocity
        learning_velocity = calculate_learning_velocity(performance_data)  # Implement this function
        
        # Predict future performance
        future_performance = predict_future_performance(user, performance_data)  # Implement this function
        
        return {
            "user_id": user.id,
            "days_active": days_active,
            "total_chats": user.chat_count,
            "avg_chats_per_day": avg_chats_per_day,
            "last_activity": user.last_activity,
            "learning_velocity": learning_velocity,
            "predicted_future_performance": future_performance
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in get_user_progress: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while fetching user progress")

def encode_text(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        return language_model(**inputs).last_hidden_state.mean(dim=1)

def get_user_data(user_id: int, db: Session) -> List[float]:
    # Implement this function to fetch relevant user data from the database
    # Return a list of numerical features representing the user
    pass

def get_all_user_data(db: Session) -> np.ndarray:
    # Implement this function to fetch data for all users
    # Return a 2D numpy array where each row represents a user's data
    pass

def calculate_complexity_score(text: str) -> float:
    # Implement this function to calculate a complexity score for the given text
    # You could use metrics like sentence length, word rarity, etc.
    pass

def calculate_learning_velocity(performance_data: Dict[str, Any]) -> float:
    # Implement this function to calculate the user's learning velocity
    # based on their performance data over time
    pass

def predict_future_performance(user: User, performance_data: Dict[str, Any]) -> Dict[str, Any]:
    # Implement this function to predict the user's future performance
    # You could use a simple trend analysis or a more complex ML model
    pass
