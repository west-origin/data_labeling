#!/bin/sh
# CVAT 공식 docker compose를 고정 버전으로 받아 띄운다.
# 사용법: services/cvat/cvat.sh up|down|status|superuser
#
# 하위 명령 (Makefile 대상):
#   up        — 공식 저장소를 DLP_CVAT_VERSION 태그로 받아(.src, 저장소에 넣지 않음) 띄우고 API가 응답할 때까지
#               최대 10분(5초 x 120) 기다린다 (make cvat-up)
#   down      — CVAT 컨테이너를 내린다. 볼륨은 남는다 (make cvat-down)
#   status    — 컨테이너 상태
#   superuser — 관리자 계정을 만든다 (make cvat-superuser). `dlp review`의 CVAT 클라이언트가 이 계정을 쓴다
# 필요: docker compose, git, curl. 환경 변수는 저장소 루트의 .env에서 읽는다 (있으면).
# -e: 명령 실패 시 중단, -u: 정의되지 않은 변수 사용 시 중단
set -eu

# 저장소 루트 (이 스크립트는 services/cvat/에 있다)
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
# .env의 모든 변수를 내보낸다 (set -a). 없으면 아래 기본값을 쓴다
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a

VERSION=${DLP_CVAT_VERSION:-v2.78.0}
PORT=${DLP_CVAT_PORT:-8080}
SRC="$ROOT/services/cvat/.src"
# 공식 compose는 traefik을 호스트 8080에 묶는다. DLP_CVAT_PORT로 호스트 포트를 바꾸도록 ports를 덮어쓴다
# (컨테이너 안 포트는 그대로 8080, 대시보드 8090은 열지 않는다).
OVERRIDE="$ROOT/services/cvat/ports.override.yml"
# 프로젝트 이름 dlp-cvat: 개발 서비스(dlp)와 컨테이너·볼륨을 섞지 않는다
COMPOSE="docker compose -p dlp-cvat -f $SRC/docker-compose.yml -f $OVERRIDE"

# 공식 저장소를 고정 태그로 얕게 받는다. 이미 같은 태그면 건너뛴다 (멱등). 다른 태그면 지우고 다시 받는다
fetch() {
  if [ -d "$SRC" ] && [ "$(git -C "$SRC" describe --tags 2>/dev/null)" = "$VERSION" ]; then
    return
  fi
  rm -rf "$SRC"
  git clone --quiet --depth 1 --branch "$VERSION" https://github.com/cvat-ai/cvat.git "$SRC"
}

# CVAT API(/api/server/about)가 200을 줄 때까지 5초 간격으로 최대 120번 확인한다
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
    # 공식 compose는 CVAT_VERSION으로 이미지 태그를, CVAT_HOST로 traefik 호스트 규칙을 정한다
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
    # 이미 같은 사용자가 있으면 createsuperuser가 실패한다 (멱등 아님)
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
