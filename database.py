import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 환경 변수 로드 및 기본값 설정
POSTGRES_USER = os.getenv("POSTGRES_USER", "default_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "default_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "default_db")

# 데이터베이스 URL 생성
try:
    DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    print(f"Using DATABASE_URL: {DATABASE_URL}")
except Exception as e:
    raise ValueError(f"Error creating DATABASE_URL: {e}")

# SQLAlchemy Async Engine 및 세션 설정
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Declarative Base
Base = declarative_base()

# Dependency Injection: DB 세션 가져오기
async def get_db():
    async with async_session() as session:
        yield session
