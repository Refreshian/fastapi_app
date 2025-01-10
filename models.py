from datetime import datetime

from sqlalchemy import MetaData, Table, Column, Integer, String, TIMESTAMP, ForeignKey, JSON, Boolean
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

metadata = MetaData()

# conn_string = "host='localhost' dbname='app_ind' user='user' password='postgres'"
engine = create_engine('postgresql+asyncpg://postgres:ffsfds&fdv12w@localhost:5432/datadb')
Base = declarative_base()

from sqlalchemy import (
    Column,
    Integer,
    String,
)

role = Table(
    "role",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("permissions", JSON),
) 

user = Table(
    "user",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String, nullable=False),
    Column("username", String, nullable=False),
    Column("registered_at", TIMESTAMP, default=datetime.utcnow),
    Column("role_id", Integer, ForeignKey(role.c.id)),
    Column("hashed_password", String, nullable=False),
    Column("is_active", Boolean, default=True, nullable=False),
    Column("is_superuser", Boolean, default=False, nullable=False),
    Column("is_verified", Boolean, default=False, nullable=False),
    Column("theme_rules", JSON),
)

if __name__ == "__main__":
    Base.metadata.create_all(engine)
