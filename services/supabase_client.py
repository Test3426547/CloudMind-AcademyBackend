from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
