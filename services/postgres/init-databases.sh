#!/bin/sh
# 서비스별 데이터베이스를 만든다. 컨테이너 최초 기동 시 한 번만 실행된다.
# 공식 postgres 이미지가 데이터 디렉터리가 비어 있을 때만 /docker-entrypoint-initdb.d/*.sh를 실행한다
# (services/docker-compose.yml이 이 파일을 마운트). 볼륨을 지우지 않으면(`make clean`) 다시 돌지 않는다.
# DB: dlp(메타데이터, `dlp db upgrade`), labelstudio, prefect, mlflow — 모두 POSTGRES_USER 소유.
# POSTGRES_DB=dlp라 dlp는 이미 있으므로 존재 확인 뒤 없을 때만 만든다 (멱등).
# compose 헬스체크는 마지막 mlflow DB로 접속해 이 스크립트가 끝났는지 본다.
set -eu
for db in dlp labelstudio prefect mlflow; do
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
    -tc "SELECT 1 FROM pg_database WHERE datname = '$db'" | grep -q 1 \
    || psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
      -c "CREATE DATABASE $db OWNER $POSTGRES_USER"
done
