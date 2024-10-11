from fastapi_users import FastAPIUsers, models
from fastapi_users.authentication import JWTAuthentication
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from fastapi_users.db import SQLAlchemyBaseUserTable
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base

DATABASE_URL = "sqlite+aiosqlite:///./test.db"

class User(SQLAlchemyBaseUserTable[int], models.BaseUser):
    pass

class UserCreate(models.BaseUserCreate):
    pass

class UserUpdate(models.BaseUserUpdate):
    pass

class UserDB(User, models.BaseUserDB):
    pass

Base: DeclarativeMeta = declarative_base()

engine = create_async_engine(DATABASE_URL)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_async_session():
    async with async_session_maker() as session:
        yield session

async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, UserDB)

jwt_authentication = JWTAuthentication(secret="your-secret-key", lifetime_seconds=3600)

fastapi_users = FastAPIUsers(
    get_user_db,
    [jwt_authentication],
    User,
    UserCreate,
    UserUpdate,
    UserDB,
)

async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

current_user = fastapi_users.current_user()
