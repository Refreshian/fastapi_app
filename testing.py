from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
# # SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

engine = create_engine(
    "postgresql://postgres:ffsfds&fdv12w@localhost:5432/datadb")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
db = SessionLocal()

from sqlalchemy.orm import Session
import models 


def get_user(db: Session):
    return db.query(models.user).filter(models.user.c.email == 'test@test.ru').first()


print(get_user(db))


import uvicorn
print(uvicorn.__version__)

print(int(((2+1) / 3)*100))