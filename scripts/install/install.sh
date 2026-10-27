#!/usr/bin/env bash
# 플랫폼 설치 스크립트 (Mac·Linux·Windows WSL2, ADR 0061 7절).
#
# 저장소 루트에서 실행한다:  scripts/install/install.sh [옵션]
#   --check-only   사전 점검만 하고 끝낸다 (아무것도 설치·실행하지 않는다)
#   --install-uv   uv가 없으면 공식 설치 스크립트로 설치한다 (없으면 안내만)
#   --curation     make install 대신 make install-curation (FiftyOne 포함, 약 1 GB 더)
#   --skip-models  make models(모델 가중치 약 0.6 GB 받기)를 건너뛴다
#   --with-cvat    CVAT도 띄운다 (make cvat-up, 처음이면 make cvat-superuser). 디스크·메모리가 많이 든다
#
# 단계마다 멱등이다 (다시 돌려도 이미 된 것은 건너뛰거나 같은 결과). 실패하면 무엇을 하라는 안내를
# 찍고 멈춘다. 이미 있는 Make 대상(install, up, db-upgrade, models, cvat-up, cvat-superuser)을 그대로
# 부른다. `.env`가 없을 때만 .env.example에서 만들고 비밀번호·키를 무작위로 바꾼다 (값은 찍지 않는다).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

CHECK_ONLY=0; INSTALL_UV=0; CURATION=0; SKIP_MODELS=0; WITH_CVAT=0
for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=1 ;;
    --install-uv) INSTALL_UV=1 ;;
    --curation) CURATION=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --with-cvat) WITH_CVAT=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "모르는 옵션: $arg (--help)"; exit 2 ;;
  esac
done

say()  { printf '\n== %s\n' "$*"; }
ok()   { printf '  [확인] %s\n' "$*"; }
warn() { printf '  [주의] %s\n' "$*"; }
fail() { printf '  [실패] %s\n' "$1"; [ -n "${2:-}" ] && printf '  → 할 일: %s\n' "$2"; exit 1; }

# 단계 실행: 이름, 실패 안내, 명령...
step() {
  local name=$1 hint=$2; shift 2
  say "$name"
  if ! "$@"; then fail "$name" "$hint"; fi
}

# ── 1. 사전 점검 ──
OS=$(uname -s); ARCH=$(uname -m); WSL=0
if [ "$OS" = Linux ] && grep -qi microsoft /proc/version 2>/dev/null; then WSL=1; fi

say "사전 점검"
case "$OS" in
  Darwin) ok "macOS ($ARCH)" ;;
  Linux) if [ "$WSL" = 1 ]; then ok "Windows WSL2 안의 Linux ($ARCH)"; else ok "Linux ($ARCH)"; fi ;;
  *) fail "지원하지 않는 운영체제: $OS" "Windows는 WSL2(Ubuntu) 안에서 이 스크립트를 실행하세요 (scripts/install/README.md)" ;;
esac

for cmd in git make curl; do
  command -v "$cmd" >/dev/null 2>&1 || fail "$cmd 없음" "패키지 관리자로 $cmd를 설치하세요 (Ubuntu: sudo apt install $cmd, Mac: xcode-select --install)"
done
ok "git·make·curl"

if ! command -v docker >/dev/null 2>&1; then
  fail "Docker 없음" "Docker Desktop(Mac·Windows, WSL2 연동 켜기) 또는 Docker Engine(Linux)을 설치하세요"
fi
docker info >/dev/null 2>&1 || fail "Docker가 응답하지 않음" "Docker Desktop을 켜거나 'sudo systemctl start docker' 뒤 다시 실행하세요 (권한 문제면 docker 그룹에 사용자 추가)"
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 없음" "Docker Compose 플러그인(docker compose)을 설치하세요"
ok "Docker·Compose v2"

if command -v ffprobe >/dev/null 2>&1; then ok "FFmpeg/ffprobe"; else warn "ffprobe가 없습니다. 영상 진단에 씁니다 (Ubuntu: sudo apt install ffmpeg, Mac: brew install ffmpeg)"; fi
if [ "$OS" = Linux ] && ! ldconfig -p 2>/dev/null | grep -q libEGL; then
  warn "EGL 라이브러리가 없을 수 있습니다. MediaPipe 손 모델에 필요합니다 (Ubuntu: sudo apt install libegl1 libgles2)"
fi

# 디스크 (저장소가 있는 곳, 60 GB 권장: 이미지·볼륨·모델·영상)
FREE_GB=$(df -Pk . | awk 'NR==2 {printf "%d", $4/1024/1024}')
if [ "$FREE_GB" -lt 60 ]; then warn "여유 디스크 ${FREE_GB} GB (60 GB 이상 권장). 저장소(SeaweedFS)는 디스크 여유가 1% 아래면 쓰기를 거부합니다"; else ok "여유 디스크 ${FREE_GB} GB"; fi

# 메모리 (16 GB 권장: 긴 영상 프리라벨·CVAT)
if [ "$OS" = Darwin ]; then MEM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )); else MEM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo); fi
if [ "$MEM_GB" -lt 16 ]; then warn "메모리 ${MEM_GB} GB (16 GB 이상 권장). 긴 영상은 단계를 나눠 돌리세요"; else ok "메모리 ${MEM_GB} GB"; fi

# GPU (있으면 쓴다: 사내 VLM 서버·학습 서버. 개발 서비스 compose 자체는 GPU를 쓰지 않는다)
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  ok "NVIDIA GPU: $(nvidia-smi -L | head -1)"
  if docker info 2>/dev/null | grep -qi nvidia; then ok "nvidia-container-toolkit (Docker GPU 런타임)"; else warn "nvidia-container-toolkit가 없어 컨테이너가 GPU를 못 씁니다 (사내 VLM 서버를 이 컴퓨터에서 띄울 때 필요, docs/handover/vlm-server.md)"; fi
else
  ok "NVIDIA GPU 없음: CPU로 돕니다 (긴 영상은 처리 시간이 영상 길이의 수십 배)"
fi
if [ "$OS" = Darwin ] && [ "$ARCH" = arm64 ]; then
  warn "Apple Silicon: CVAT 이미지는 amd64라 에뮬레이션으로 돌아 느립니다 (Docker Desktop의 Rosetta 사용 켜기 권장)"
fi

# uv
if ! command -v uv >/dev/null 2>&1; then
  if [ "$INSTALL_UV" = 1 ] && [ "$CHECK_ONLY" = 0 ]; then
    step "uv 설치" "https://docs.astral.sh/uv/ 의 설치 안내를 따르세요" sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    export PATH="$HOME/.local/bin:$PATH"
  else
    fail "uv 없음" "이 스크립트를 --install-uv와 다시 실행하거나 'curl -LsSf https://astral.sh/uv/install.sh | sh' 뒤 새 터미널에서 다시 실행하세요"
  fi
fi
ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"

if [ "$CHECK_ONLY" = 1 ]; then say "사전 점검만 했습니다 (--check-only)"; exit 0; fi

# ── 2. 의존성 (uv.lock 그대로) ──
if [ "$CURATION" = 1 ]; then
  step "의존성 설치 (FiftyOne 포함)" "네트워크·디스크를 확인하고 다시 실행하세요" make install-curation
else
  step "의존성 설치" "네트워크·디스크를 확인하고 다시 실행하세요 (uv.lock과 pyproject가 어긋나면 저장소를 최신으로)" make install
fi

# ── 3. .env (없을 때만 만든다. 있으면 그대로 둔다) ──
rand_hex() { python3 -c "import secrets,sys; print(secrets.token_hex(int(sys.argv[1])))" "$1"; }
rand_upper() { python3 -c "import secrets,string,sys; a=string.ascii_uppercase+string.digits; print(''.join(secrets.choice(a) for _ in range(int(sys.argv[1]))))" "$1"; }
make_env() {
  local pg s3 lab ls_pw ls_tok lk_ak lk_sk lk_enc hook cvat exp
  pg=$(rand_hex 16); s3=$(rand_hex 20); lab=$(rand_hex 20); ls_pw=$(rand_hex 12); ls_tok=$(rand_hex 20)
  lk_ak="AKIA$(rand_upper 16)"; lk_sk=$(rand_hex 20); lk_enc=$(rand_hex 24); hook=$(rand_hex 24)
  cvat=$(rand_hex 12); exp=$(rand_hex 24)
  umask 077
  PG="$pg" S3="$s3" LAB="$lab" LSPW="$ls_pw" LSTOK="$ls_tok" LKAK="$lk_ak" LKSK="$lk_sk" LKENC="$lk_enc" \
  HOOK="$hook" CVATPW="$cvat" EXP="$exp" python3 - <<'PY'
import os, re
from pathlib import Path
text = Path(".env.example").read_text("utf-8")
values = {
    "DLP_POSTGRES_PASSWORD": os.environ["PG"],
    "DLP_S3_SECRET_KEY": os.environ["S3"],
    "DLP_S3_LABELER_SECRET_KEY": os.environ["LAB"],
    "DLP_LABEL_STUDIO_PASSWORD": os.environ["LSPW"],
    "DLP_LABEL_STUDIO_TOKEN": os.environ["LSTOK"],
    "DLP_LAKEFS_ACCESS_KEY": os.environ["LKAK"],
    "DLP_LAKEFS_SECRET_KEY": os.environ["LKSK"],
    "DLP_LAKEFS_ENCRYPT_SECRET": os.environ["LKENC"],
    "DLP_WEBHOOK_SECRET": os.environ["HOOK"],
    "DLP_CVAT_ADMIN_PASSWORD": os.environ["CVATPW"],
    "DLP_EXPORT_ID_SECRET": os.environ["EXP"],
}
for key, value in values.items():
    text, n = re.subn(rf"^{key}=.*$", f"{key}={value}", text, flags=re.M)
    assert n == 1, key
# DB URL의 비밀번호도 같은 값으로
text = re.sub(r"^(DLP_DATABASE_URL=postgresql\+psycopg://[^:]+:)[^@]*(@.*)$",
              lambda m: m.group(1) + values["DLP_POSTGRES_PASSWORD"] + m.group(2), text, flags=re.M)
Path(".env").write_text(text, "utf-8")
PY
  chmod 600 .env
}
say ".env"
if [ -f .env ]; then ok ".env가 이미 있습니다 (바꾸지 않습니다)"; else make_env && ok ".env를 만들었습니다 (비밀번호·키 무작위, 권한 0600, 값은 Git·화면에 내지 마세요)"; fi

# `dlp` 명령은 .env를 스스로 읽지 않는다 (.env.example 머리말). 아래 make 대상이 같은 값을 쓰게 불러온다
set -a; . ./.env; set +a

# ── 4. 서비스·DB·모델 ──
step "개발 서비스 띄우기 (make up)" "Docker 디스크·포트 충돌(5432·8333·8081·4200·5000·8000)을 확인하고 'make ps'·'make logs'로 원인을 보세요" make up
step "DB 마이그레이션·온톨로지 등록 (make db-upgrade)" "종료 코드 3이면 물리 사전값 승인이 필요합니다 (CLAUDE.md, ADR 0053). 그 밖에는 'make health'로 DB를 확인하세요" make db-upgrade
if [ "$SKIP_MODELS" = 0 ]; then
  step "모델 가중치 받기 (make models)" "네트워크를 확인하고 다시 실행하세요 (이미 받은 파일은 건너뜁니다)" make models
else
  warn "모델 가중치를 건너뛰었습니다. AI 라벨 전에 'make models'를 실행하세요"
fi

# ── 5. CVAT (선택) ──
if [ "$WITH_CVAT" = 1 ]; then
  step "CVAT 띄우기 (make cvat-up)" "디스크·메모리를 확인하세요. 처음에는 이미지 받기에 수 분 걸립니다" make cvat-up
  say "CVAT 관리자 계정 (make cvat-superuser)"
  if make cvat-superuser; then ok "관리자 계정을 만들었습니다"; else warn "이미 있으면 실패해도 괜찮습니다 (그 밖이면 'services/cvat/cvat.sh status')"; fi
else
  warn "CVAT는 띄우지 않았습니다. 블러 검수·공간 라벨 검수 전에 'make cvat-up'·'make cvat-superuser'를 실행하세요"
fi

say "설치 끝"
cat <<'MSG'
  운영 콘솔 띄우기:   uv run dlp console serve   (콘솔은 .env를 읽어 단계 명령에 넘긴다)
  터미널에서 dlp 명령을 직접 쓸 때는 먼저:  set -a; . ./.env; set +a
  찍힌 주소(http://localhost:8090/#token=…)를 브라우저로 엽니다. 콘솔은 이 컴퓨터에서만 열립니다.
  검수 계정 연결(블러 검수 담당자 = 원본 접근 권한자)은 검수 정책 파일에 적고
  DLP_REVIEW_POLICY_FILE로 지정합니다. 안내: docs/handover/operator-guide.md
MSG
