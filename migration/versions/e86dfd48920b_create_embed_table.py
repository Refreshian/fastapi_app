"""create_embed_table

Revision ID: e86dfd48920b
Revises: ef5934af6235
Create Date: 2025-02-05 21:53:22.625385

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e86dfd48920b'
down_revision: Union[str, None] = 'ef5934af6235'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
