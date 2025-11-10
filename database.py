from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "sqlite:///./attendance.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    face_encoding = Column(Text)  # JSON string of numpy array
    image_path = Column(String)

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    date = Column(String)
    time = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class Visitor(Base):
    __tablename__ = "visitors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    person_to_meet = Column(String, nullable=True)
    status = Column(String, default="Pending")  # Pending, Approved, Rejected
    image_path = Column(String)
    face_encoding = Column(Text, nullable=True)  # JSON string of numpy array
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
