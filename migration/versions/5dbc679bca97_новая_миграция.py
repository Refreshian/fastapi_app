"""Добавление таблицы embeddings

Revision ID: ef5934af6235
Revises: 
Create Date: 2025-01-29 11:55:10.643303

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ef5934af6235'
down_revision: Union[str, None] = None  # Указываем None, так как это первая миграция
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Здесь вы можете добавить логику повышения миграции
    pass

def downgrade() -> None:
    # Логика для удаления колонки, если такая существует
    op.drop_column('user', 'theme_rules')