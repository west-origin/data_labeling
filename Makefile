# 모든 패키지에 같은 명령으로 검증한다. 에이전트와 CI는 `make check`를 기준으로 삼는다.
# 개발 명령 모음 (CLAUDE.md "명령어" 절과 같은 목록).
#
# 처음 시작하는 순서:
#   make install            의존성 설치 (uv, 잠금 파일 기준)
#   make check              린트·타입·계약·단위 테스트 (서비스 없이)
#   make up                 개발 서비스 기동 (docker 필요, .env 자동 생성)
#   make db-upgrade         DB 마이그레이션 + 온톨로지 v1 등록
#   make models             모델 가중치 받기 (실제 모델로 돌릴 때)
#   make test-services      서비스 통합 테스트
#
# 공통 전제: uv가 설치되어 있고 저장소 루트에서 실행한다. `uv run`은 작업공간 가상환경(.venv)에서 돈다.

# 개발 서비스 compose 명령 (up/down/ps/logs/clean이 쓴다). .env가 있어야 한다 (make env)
COMPOSE := docker compose --env-file .env -f services/docker-compose.yml

# 계약 검사 (make check에 포함): 온톨로지 YAML 검증, JSON Schema(schemas/)가 코드와 같은지,
# 정책이 쓰는 모델의 상업 사용 분류 위반이 없는지(위반 줄만 보여 주고 실패). 서비스 불필요.
contracts:
	uv run dlp ontology validate
	uv run dlp schema export --check
	@uv run dlp models licenses > /dev/null || (uv run dlp models licenses | grep 위반; exit 1)

# 정책이 쓰는 모델의 라이선스·학습 데이터·상업 사용 분류 표 (판매 실사 자료)
# 출력은 Markdown 표. 위반이 있으면 종료 코드 1 (ADR 0010). 서비스 불필요.
licenses:
	uv run dlp models licenses

# 계약 타입(dlp_schema)을 바꾼 뒤 schemas/*.schema.json을 다시 만든다 (직접 편집 금지).
# 계약 변경은 ADR + Alembic 마이그레이션 + 계약 테스트와 함께 한다.
schemas:
	uv run dlp schema export

# config/models.yaml의 가중치를 data/models/에 받고 sha256을 확인한다 (네트워크 필요, 약 360 MB).
# 이미 받은 파일은 건너뛴다. 가중치는 저장소에 넣지 않는다.
models:
	uv run dlp models fetch

# 공개 ONNX가 없는 모델(깊이)을 공식 가중치에서 변환한다 (일회용 PyTorch CPU 환경, 수 분)
# 결과: data/models/*.onnx. 이미 있으면 건너뛴다. 네트워크 필요 (Hugging Face, PyTorch 색인).
export-models:
	uv run dlp models export

# 실제 모델을 아직 연동하지 못한 곳 (TODO(real-model) 표시)
# packages/·config/의 .py·.yaml에서 표시 줄을 모두 찾는다. 하나도 없으면 grep이 실패(종료 코드 1)한다.
todo-models:
	@grep -rn "TODO(real-model):" --include=*.py --include=*.yaml packages config | grep -v "/.venv/"

# 정답을 아는 합성 픽스처를 data/fixtures/에 만든다 (저장소에 넣지 않음, 시드 0 고정 → 결정적)
fixtures:
	uv run dlp fixtures generate --out data/fixtures

# 개발 DB에 Alembic 마이그레이션을 끝까지 적용하고 온톨로지 1.0.0을 등록한다.
# 전제: make up (PostgreSQL), DLP_DATABASE_URL(없으면 개발 기본값). 여러 번 돌려도 된다.
db-upgrade:
	uv run dlp db upgrade
	uv run dlp ontology register 1.0.0

# 파일이 아닌 명령 대상 (같은 이름의 파일이 있어도 항상 실행)
.PHONY: install install-curation test-isolated lint fmt typecheck test contracts schemas models export-models todo-models licenses fixtures db-upgrade check env up down ps logs health test-services \
        cvat-up cvat-down cvat-superuser clean

# 작업공간 의존성 설치 (uv.lock 그대로, --locked: 잠금 파일과 pyproject가 어긋나면 실패).
# 선택 그룹(curation)은 빠진다 — FiftyOne이 필요하면 make install-curation.
install:
	uv sync --locked

# 격리된 일회용 환경을 받는 테스트 (LeRobot 공식 쓰기·읽기, PyTorch CPU판 약 1.5 GB, 처음 한 번 수 분).
# 환경은 scripts/lerobot-env/uv.lock에 고정한다. 일부 테스트(내보내기 종단)는 DB가 필요하다: make up 먼저.
# services 표시도 있는 테스트는 여기서만 돈다 (test-services는 isolated_env를 뺀다, 두 번 돌지 않게).
# isolated_env 표시 테스트만 돈다 (기본 pytest 실행에서는 빠진다, pyproject addopts).
test-isolated:
	uv run pytest -m isolated_env

# FiftyOne(데이터 큐레이션, 약 1 GB). make install을 다시 하면 빠진다
# 작업공간 의존성 + curation 그룹 설치. CI는 이 대상을 써서 FiftyOne 연동 테스트까지 돌린다.
install-curation:
	uv sync --locked --group curation

# 린트(ruff check)와 포맷 검사(ruff format --check). 고치지는 않는다 (make fmt). 줄 길이 100.
lint:
	uv run ruff check .
	uv run ruff format --check .

# 자동 포맷과 고칠 수 있는 린트 수정. 파일을 바꾼다.
fmt:
	uv run ruff format .
	uv run ruff check --fix .

# pyright strict (pyproject [tool.pyright]: packages/와 tests/). scripts/·services/는 범위 밖.
typecheck:
	uv run pyright

# 단위 테스트 (pyproject addopts가 services·isolated_env 표시 테스트를 뺀다). 서비스 불필요.
test:
	uv run pytest

# PR 전에 반드시 통과해야 하는 전체 검사 (CI check 작업과 같다).
check: lint typecheck contracts test

# .env가 없을 때만 .env.example을 복사한다 (있으면 그대로 둔다)
env:
	@test -f .env || cp .env.example .env

# 개발 서비스(PostgreSQL, SeaweedFS S3, Label Studio, Prefect, MLflow, lakeFS)를 띄우고 헬스체크 통과까지
# 기다린 뒤(--wait), Label Studio 레거시 API 토큰을 켠다. 전제: docker. 처음에는 이미지 받기에 수 분.
up: env
	$(COMPOSE) up -d --wait
	set -a; . ./.env; set +a; python3 services/label-studio/bootstrap.py

# 서비스 중지 (볼륨은 남는다)
down:
	$(COMPOSE) down

# 서비스 상태 (끝난 일회성 작업 bucket-init 포함)
ps:
	$(COMPOSE) ps -a

# 서비스 로그를 따라 본다 (Ctrl-C로 끝낸다)
logs:
	$(COMPOSE) logs -f --tail=100

# `dlp services check`: 서비스 포트·헬스 엔드포인트 점검 (CVAT 제외). 하나라도 실패하면 종료 코드 1.
health:
	uv run dlp services check

# 격리 환경이 필요한 테스트는 make test-isolated에서 돈다
# 실행 중인 서비스 대상 통합 테스트 (@pytest.mark.services). 전제: make up (+ CVAT 테스트는 make cvat-up).
test-services:
	uv run pytest -m "services and not isolated_env"

# CVAT 공식 compose를 고정 버전(DLP_CVAT_VERSION)으로 받아 띄우고 API 응답까지 기다린다 (git·curl 필요).
# 처음이면 관리자 계정을 만든다: make cvat-superuser
cvat-up: env
	services/cvat/cvat.sh up

# CVAT 중지 (볼륨은 남는다)
cvat-down:
	services/cvat/cvat.sh down

# CVAT 관리자 계정 생성 (DLP_CVAT_ADMIN_*). 이미 있으면 실패한다.
cvat-superuser:
	services/cvat/cvat.sh superuser

# 볼륨까지 지운다 (개발 데이터 초기화)
# 주의: DB·버킷·lakeFS·Label Studio 데이터가 모두 사라진다. CVAT 볼륨은 지우지 않는다.
clean:
	$(COMPOSE) down -v
