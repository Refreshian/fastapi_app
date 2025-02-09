from datetime import datetime

from sqlalchemy import MetaData, Table, Column, Integer, String, TIMESTAMP, ForeignKey, JSON, Boolean
from sqlalchemy import *

metadata = MetaData()

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

# embeddings = Table(
#     "embeddings",
#     metadata,
#     Column("id", Integer, primary_key=True),
#     Column("user_id", Integer, ForeignKey(user.c.id), nullable=False),  # Установлен внешний ключ на таблицу user
#     Column("filename", String(255), nullable=False),
#     Column("folder_name", String(255), nullable=False),
#     Column("embedding", LargeBinary, nullable=False),
#     Column("created_at", TIMESTAMP, default=datetime.utcnow),
# )

# embedding = Table(
#     "embedding",
#     metadata,
#     Column("id", Integer, primary_key=True, index=True),
#     Column("user_id", Integer, nullable=False),  # Указан идентификатор пользователя
#     Column("filename", String(255), nullable=False),  # Имя файла
#     Column("vectors", JSON, nullable=False),  # Поле для хранения эмбеддингов в формате JSON
# )


embeddings = Table(
    "embeddings_pg",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("user_id", Integer, nullable=False),  # Указан идентификатор пользователя
    Column("filename", String(255), nullable=False),  # Имя файла
    Column("folder_name", String(255), nullable=False),  # Имя папки
    Column("vectors", JSON, nullable=False),  # Поле для хранения эмбеддингов в формате JSON
)