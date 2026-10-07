#!/bin/sh
# CVAT 공식 docker compose를 고정 버전으로 받아 띄운다.
# 사용법: services/cvat/cvat.sh up|down|status|superuser
set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a

VERSION=${DLP_CVAT_VERSION:-v2.78.0}
PORT=${DLP_CVAT_PORT:-8080}
SRC="$ROOT/services/cvat/.src"
# 공식 compose는 traefik을 호스트 8080에 묶는다. DLP_CVAT_PORT로 호스트 포트를 바꾸도록 ports를 덮어쓴다
# (컨테이너 안 포트는 그대로 8080, 대시보드 8090은 열지 않는다).
OVERRIDE="$ROOT/services/cvat/ports.override.yml"
COMPOSE="docker compose -p dlp-cvat -f $SRC/docker-compose.yml -f $OVERRIDE"

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
    # 받아 둔 compose가 없으면 띄운 적이 없다 (set -e에서 실패로 끝나지 않게 if로 쓴다)
    if [ -d "$SRC" ]; then
      $COMPOSE down
    fi
    ;;
  status)
    if [ -d "$SRC" ]; then
      $COMPOSE ps
    else
      echo "CVAT를 받아 둔 적이 없습니다 (make cvat-up)"
    fi
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
