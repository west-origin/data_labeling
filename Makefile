# 모든 패키지에 같은 명령으로 검증한다. 에이전트와 CI는 `make check`를 기준으로 삼는다.
COMPOSE := docker compose --env-file .env -f services/docker-compose.yml

contracts:
	uv run dlp ontology validate
	uv run dlp schema export --check

schemas:
	uv run dlp schema export

models:
	uv run dlp models fetch

fixtures:
	uv run dlp fixtures generate --out data/fixtures

db-upgrade:
	uv run dlp db upgrade
	uv run dlp ontology register 1.0.0

.PHONY: install lint fmt typecheck test contracts schemas models fixtures db-upgrade check env up down ps logs health test-services \
        cvat-up cvat-down cvat-superuser clean

install:
	uv sync --locked

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

test-services:
	uv run pytest -m services

cvat-up: env
	services/cvat/cvat.sh up

cvat-down:
	services/cvat/cvat.sh down

cvat-superuser:
	services/cvat/cvat.sh superuser

# 볼륨까지 지운다 (개발 데이터 초기화)
clean:
	$(COMPOSE) down -v
