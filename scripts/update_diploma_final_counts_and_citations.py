from __future__ import annotations

import sys
from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "Диплом все - с графиками Tableau.docx"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: update_diploma_final_counts_and_citations.py <page_count>")
    page_count = sys.argv[1]

    doc = Document(DOC_PATH)
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text.startswith("Пояснительная записка к выпускной квалификационной работе содержит"):
            paragraph.text = (
                f"Пояснительная записка к выпускной квалификационной работе содержит "
                f"{page_count} с., 34 табл., 21 рис., 42 источника."
            )
        if "слой хранения данных - на SQLAlchemy ORM с возможностью использования PostgreSQL и миграций Alembic [35-37]" in text:
            paragraph.text = text.replace(
                "слой хранения данных - на SQLAlchemy ORM с возможностью использования PostgreSQL и миграций Alembic [35-37]",
                "слой хранения данных - на SQLAlchemy ORM с возможностью использования PostgreSQL и миграций Alembic [35, 36, 37]",
            )
        if "SQLAlchemy ORM с возможностью использования PostgreSQL и миграций Alembic [35-37]" in text:
            paragraph.text = text.replace(
                "SQLAlchemy ORM с возможностью использования PostgreSQL и миграций Alembic [35-37]",
                "SQLAlchemy ORM с возможностью использования PostgreSQL и миграций Alembic [35, 36, 37]",
            )
    doc.save(DOC_PATH)
    print(DOC_PATH)


if __name__ == "__main__":
    main()
