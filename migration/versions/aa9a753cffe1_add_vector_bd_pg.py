"""add vector bd pg

Revision ID: aa9a753cffe1
Revises: 1c526ed0c5cb
Create Date: 2025-02-08 14:55:06.818725

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'aa9a753cffe1'
down_revision: Union[str, None] = '1c526ed0c5cb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
