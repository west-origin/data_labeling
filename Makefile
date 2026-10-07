# 모든 패키지에 같은 명령으로 검증한다. 에이전트와 CI는 `make check`를 기준으로 삼는다.
COMPOSE := docker compose --env-file .env -f services/docker-compose.yml

contracts:
	uv run dlp ontology validate
	uv run dlp schema export --check
	@uv run dlp models licenses > /dev/null || (uv run dlp models licenses | grep 위반; exit 1)

# 정책이 쓰는 모델의 라이선스·학습 데이터·상업 사용 분류 표 (판매 실사 자료)
licenses:
	uv run dlp models licenses

schemas:
	uv run dlp schema export

models:
	uv run dlp models fetch

# 공개 ONNX가 없는 모델(깊이)을 공식 가중치에서 변환한다 (일회용 PyTorch CPU 환경, 수 분)
export-models:
	uv run dlp models export

# 실제 모델을 아직 연동하지 못한 곳 (TODO(real-model) 표시)
todo-models:
	@grep -rn "TODO(real-model):" --include=*.py --include=*.yaml packages config | grep -v "/.venv/"

fixtures:
	uv run dlp fixtures generate --out data/fixtures

db-upgrade:
	uv run dlp db upgrade
	uv run dlp ontology register 1.0.0

.PHONY: install install-curation test-isolated lint fmt typecheck test contracts schemas models export-models todo-models licenses fixtures db-upgrade check env up down ps logs health test-services \
        cvat-up cvat-down cvat-superuser clean

install:
	uv sync --locked

# 격리된 일회용 환경을 받는 테스트 (LeRobot 공식 쓰기·읽기, PyTorch CPU판 약 1.5 GB, 처음 한 번 수 분).
# 환경은 scripts/lerobot-env/uv.lock에 고정한다. 일부 테스트(내보내기 종단)는 DB가 필요하다: make up 먼저.
# services 표시도 있는 테스트는 여기서만 돈다 (test-services는 isolated_env를 뺀다, 두 번 돌지 않게).
test-isolated:
	uv run pytest -m isolated_env

# FiftyOne(데이터 큐레이션, 약 1 GB). make install을 다시 하면 빠진다
install-curation:
	uv sync --locked --group curation

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run pyright

test:
	uv run pytest

check: lint typecheck contracts test

env:
	@test -f .env || cp .env.example .env

up: env
	$(COMPOSE) up -d --wait
	set -a; . ./.env; set +a; python3 services/label-studio/bootstrap.py

down:
	$(COMPOSE) down

ps:
	$(COMPOSE) ps -a

logs:
	$(COMPOSE) logs -f --tail=100

health:
	uv run dlp services check

# 격리 환경이 필요한 테스트는 make test-isolated에서 돈다
test-services:
	uv run pytest -m "services and not isolated_env"

cvat-up: env
	services/cvat/cvat.sh up

cvat-down:
	services/cvat/cvat.sh down

cvat-superuser:
	services/cvat/cvat.sh superuser

# 볼륨까지 지운다 (개발 데이터 초기화)
clean:
	$(COMPOSE) down -v
