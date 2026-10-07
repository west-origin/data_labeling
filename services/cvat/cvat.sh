#!/bin/sh
# CVAT 공식 docker compose를 고정 버전으로 받아 띄운다.
# 사용법: services/cvat/cvat.sh up|down|status|superuser
set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a

VERSION=${DLP_CVAT_VERSION:-v2.78.0}
PORT=${DLP_CVAT_PORT:-8080}
SRC="$ROOT/services/cvat/.src"
COMPOSE="docker compose -p dlp-cvat -f $SRC/docker-compose.yml"

fetch() {
  if [ -d "$SRC" ] && [ "$(git -C "$SRC" describe --tags 2>/dev/null)" = "$VERSION" ]; then
    return
  fi
  rm -rf "$SRC"
  git clone --quiet --depth 1 --branch "$VERSION" https://github.com/cvat-ai/cvat.git "$SRC"
}

wait_healthy() {
  i=0
  until curl -fs -o /dev/null "http://localhost:$PORT/api/server/about"; do
    i=$((i + 1))
    if [ "$i" -ge 120 ]; then
      echo "CVAT가 제한 시간 안에 응답하지 않습니다" >&2
      exit 1
    fi
    sleep 5
  done
  echo "CVAT 준비됨: http://localhost:$PORT"
}

case "${1:-}" in
  up)
    fetch
    CVAT_VERSION="$VERSION" CVAT_HOST=localhost $COMPOSE up -d
    wait_healthy
    ;;
  down)
    [ -d "$SRC" ] && $COMPOSE down
    ;;
  status)
    $COMPOSE ps
    ;;
  superuser)
    # DLP_CVAT_ADMIN_USER / DLP_CVAT_ADMIN_PASSWORD / DLP_CVAT_ADMIN_EMAIL로 관리자 계정을 만든다.
    docker exec \
      -e DJANGO_SUPERUSER_USERNAME="${DLP_CVAT_ADMIN_USER:-admin}" \
      -e DJANGO_SUPERUSER_PASSWORD="${DLP_CVAT_ADMIN_PASSWORD:?DLP_CVAT_ADMIN_PASSWORD를 설정하세요}" \
      -e DJANGO_SUPERUSER_EMAIL="${DLP_CVAT_ADMIN_EMAIL:-admin@dlp.local}" \
      cvat_server python3 manage.py createsuperuser --noinput
    ;;
  *)
    echo "사용법: $0 up|down|status|superuser" >&2
    exit 2
    ;;
esac
