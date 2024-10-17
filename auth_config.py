import os
from dotenv import load_dotenv
from supabase import create_client, Client
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", 0.2))
MODEL_MAX_TOTAL_TOKENS = int(os.getenv("MODEL_MAX_TOTAL_TOKENS", 4096))
MODEL_TOP_K = int(os.getenv("MODEL_TOP_K", 50))
MODEL_TOP_P = float(os.getenv("MODEL_TOP_P", 0.95))
MODEL_DO_SAMPLE = os.getenv("MODEL_DO_SAMPLE", "true").lower() == "true"
MODEL_NUM_RETURN_SEQUENCES = int(os.getenv("MODEL_NUM_RETURN_SEQUENCES", 1))

if not all([SUPABASE_URL, SUPABASE_KEY, DATABASE_URL, HF_ENDPOINT_URL]):
    raise ValueError("All environment variables must be set in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_supabase_client() -> Client:
    return supabase

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
