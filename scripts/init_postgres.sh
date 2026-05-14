#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Создание PostgreSQL роли и базы данных для DPLM.
#
# Использование:
#   ./scripts/init_postgres.sh                # значения по умолчанию
#   DPLM_DB_USER=foo DPLM_DB_PASSWORD=bar ./scripts/init_postgres.sh
#
# После успешного выполнения скрипт применяет миграции через Alembic
# (если активировано виртуальное окружение и установлен alembic).
#
# Требования: установленный PostgreSQL с доступным `psql`. Скрипт
# подключается к системной БД `postgres` под суперпользователем
# (по умолчанию `postgres` или текущим пользователем ОС).
# -----------------------------------------------------------------------------
set -euo pipefail

DPLM_DB_HOST="${DPLM_DB_HOST:-localhost}"
DPLM_DB_PORT="${DPLM_DB_PORT:-5432}"
DPLM_DB_USER="${DPLM_DB_USER:-dplm}"
DPLM_DB_PASSWORD="${DPLM_DB_PASSWORD:-dplm}"
DPLM_DB_NAME="${DPLM_DB_NAME:-dplm}"
PG_SUPERUSER="${PG_SUPERUSER:-postgres}"

echo "[i] Проверяю наличие psql..."
if ! command -v psql >/dev/null 2>&1; then
  echo "[!] psql не найден. Установите PostgreSQL и повторите запуск." >&2
  exit 1
fi

PSQL=(psql -h "$DPLM_DB_HOST" -p "$DPLM_DB_PORT" -U "$PG_SUPERUSER" -d postgres -v ON_ERROR_STOP=1)

echo "[i] Создаю роль '$DPLM_DB_USER' (если отсутствует)..."
"${PSQL[@]}" <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '$DPLM_DB_USER') THEN
    CREATE ROLE "$DPLM_DB_USER" WITH LOGIN PASSWORD '$DPLM_DB_PASSWORD';
  END IF;
END
\$\$;
SQL

echo "[i] Создаю базу '$DPLM_DB_NAME' (если отсутствует)..."
EXISTS=$("${PSQL[@]}" -tAc "SELECT 1 FROM pg_database WHERE datname='$DPLM_DB_NAME'" || true)
if [[ "$EXISTS" != "1" ]]; then
  "${PSQL[@]}" -c "CREATE DATABASE \"$DPLM_DB_NAME\" OWNER \"$DPLM_DB_USER\";"
fi

echo "[i] Назначаю права..."
"${PSQL[@]}" -c "GRANT ALL PRIVILEGES ON DATABASE \"$DPLM_DB_NAME\" TO \"$DPLM_DB_USER\";"

if command -v alembic >/dev/null 2>&1; then
  echo "[i] Применяю миграции Alembic..."
  DPLM_DB_HOST="$DPLM_DB_HOST" \
  DPLM_DB_PORT="$DPLM_DB_PORT" \
  DPLM_DB_USER="$DPLM_DB_USER" \
  DPLM_DB_PASSWORD="$DPLM_DB_PASSWORD" \
  DPLM_DB_NAME="$DPLM_DB_NAME" \
  alembic upgrade head
else
  echo "[w] alembic не найден — миграции пропущены."
  echo "    Активируйте venv и выполните: alembic upgrade head"
fi

echo "[OK] PostgreSQL готов: postgresql+psycopg2://$DPLM_DB_USER@$DPLM_DB_HOST:$DPLM_DB_PORT/$DPLM_DB_NAME"
