"""Alembic environment.

URL подключения берётся из ``app.models.database.resolve_database_url`` —
значит, его задают переменные окружения ``DATABASE_URL`` либо ``DPLM_DB_*``
(см. описание функции). Значение из ``alembic.ini`` используется только как
запасной вариант для офлайн-режима.
"""
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

import sys

print("[alembic] Импорт ORM и resolve_database_url …", file=sys.stderr, flush=True)
from app.models.database import Base, resolve_database_url

target_metadata = Base.metadata

# Перекрываем sqlalchemy.url из alembic.ini значением из окружения,
# чтобы не дублировать конфигурацию.
runtime_url = resolve_database_url()
print("[alembic] URL собран, дальше — подключение к серверу БД.", file=sys.stderr, flush=True)
config.set_main_option("sqlalchemy.url", runtime_url)


def run_migrations_offline() -> None:
    """Migrations in 'offline' mode (без подключения)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Migrations in 'online' mode (с подключением к БД)."""
    url = config.get_main_option("sqlalchemy.url") or ""
    tail = url.split("@", 1)[-1] if "@" in url else url
    print(
        f"[alembic] Подключение к БД … ({tail[:80]}{'…' if len(tail) > 80 else ''})",
        file=sys.stderr,
        flush=True,
    )
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
