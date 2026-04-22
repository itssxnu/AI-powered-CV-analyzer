import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, Enum, DateTime, JSON, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
import enum

# Define Database URL directly or through env vars
# Using the same database as the Java Spring Boot application (hotel_ai)
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "2003")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "aiml_project")

DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class MatchLabel(str, enum.Enum):
    good = "good"
    close = "close"
    bad = "bad"

class CVParsed(Base):
    __tablename__ = "cv_parsed"

    id = Column(Integer, primary_key=True, index=True)
    source_file = Column(String(255), nullable=False)
    candidate_name = Column(String(255), nullable=True)
    candidate_email = Column(String(255), nullable=True)
    career_level = Column(String(50), nullable=True) # Junior, Mid, or Senior
    parsed_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class MatchResult(Base):
    __tablename__ = "match_results"

    id = Column(Integer, primary_key=True, index=True)
    cv_id = Column(Integer, nullable=False)  # FK to cv_parsed.id conceptually
    job_id = Column(Integer, nullable=False) # FK to job_descriptions.id conceptually
    score = Column(Float, nullable=False)
    label = Column(Enum(MatchLabel), nullable=False)
    details_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('cv_id', 'job_id', name='uq_cv_job_match'),
    )

class EmbeddingCache(Base):
    __tablename__ = "embedding_cache"

    id = Column(Integer, primary_key=True, index=True)
    text_hash = Column(String(64), unique=True, index=True, nullable=False)
    phrase = Column(Text, nullable=False)
    embedding_vector = Column(Text, nullable=False) # JSON encoded list of floats
    created_at = Column(DateTime, default=datetime.utcnow)

class JobDescriptionRef(Base):
    """
    Read-only reference to the job_descriptions table managed by the Spring Boot app.
    We define it here just to be able to fetch JDs using SQLAlchemy.
    """
    __tablename__ = "job_descriptions"
    
    id = Column(Integer, primary_key=True)
    title = Column(String(120), nullable=False)
    department = Column(String(60), nullable=False)
    min_years = Column(Integer)
    job_text = Column(Text)
    required_skills_json = Column(Text, nullable=False)
    preferred_skills_json = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

def init_db():
    # Only create tables that don't exist yet
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database tables...")
    init_db()
    print("Database tables initialized successfully.")
