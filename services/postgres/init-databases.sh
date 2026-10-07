#!/bin/sh
# 서비스별 데이터베이스를 만든다. 컨테이너 최초 기동 시 한 번만 실행된다.
set -eu
for db in dlp labelstudio prefect mlflow; do
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
    -tc "SELECT 1 FROM pg_database WHERE datname = '$db'" | grep -q 1 \
    || psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
      -c "CREATE DATABASE $db OWNER $POSTGRES_USER"
done
