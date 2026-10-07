# data_labeling

청소·요양돌봄·간호 작업자의 1인칭 바디캠 영상을 로봇 모방학습용 데이터로 라벨링하는 플랫폼.

- 계획: [docs/labeling-platform-plan.md](docs/labeling-platform-plan.md)
- 구현 계획: [docs/ai-implementation-plan.md](docs/ai-implementation-plan.md)

## 구성

- `packages/schema` (`dlp_schema`): 세션·스트림·라벨·에피소드 그래프 계약 타입, 온톨로지 로더·검증, 온톨로지 이관, DB 스키마
- `packages/fixtures` (`dlp_fixtures`): 정답을 아는 합성 데이터 생성기 (`dlp fixtures generate`)
- `packages/media` (`dlp_media`): 세션 수집, PTS 인덱스, 프록시 영상, IMU·장갑 정규화 (`dlp ingest`, `dlp media`)
- `packages/sync` (`dlp_sync`): 멀티스트림 동기화 (`dlp sync run`, `dlp sync adjust`)
- `packages/cli` (`dlp`): 명령줄 도구
- `config/ontology/v1`: 온톨로지 v1 초안, `config/defaults.yaml`: 미결정 사항 기본값
- `schemas/`: 계약 타입의 JSON Schema (생성 파일)

## 시작하기

필요: [uv](https://docs.astral.sh/uv/), Docker (Compose v2), make

```sh
make install        # Python 3.12 환경과 의존성
make check          # 린트·타입·테스트
make up             # PostgreSQL, SeaweedFS(S3), Label Studio, Prefect, MLflow
make health         # 헬스체크
make db-upgrade     # 메타데이터 DB 마이그레이션 + 온톨로지 v1 등록
make cvat-up        # CVAT (선택, 이미지가 커서 별도)
```

| 서비스 | 주소 | 비고 |
| --- | --- | --- |
| PostgreSQL | localhost:5432 | DB: dlp, labelstudio, prefect, mlflow |
| SeaweedFS S3 | http://localhost:8333 | 버킷: dlp-raw, dlp-labeling, dlp-datasets, dlp-mlflow |
| Label Studio | http://localhost:8081 | 계정은 `.env`의 `DLP_LABEL_STUDIO_*` |
| Prefect | http://localhost:4200 | |
| MLflow | http://localhost:5000 | 산출물은 S3 `dlp-mlflow` |
| CVAT | http://localhost:8080 | `make cvat-up`, 관리자는 `make cvat-superuser` |

포트와 계정은 `.env`(최초 `make up` 시 `.env.example`에서 복사)로 바꾼다. 기본값은 개발용이다.

TLS를 가로채는 프록시 뒤에서 MLflow 이미지 빌드가 실패하면, CA 묶음을 빌드 시크릿으로 넘겨 먼저 빌드한다:

```sh
docker build --secret id=extra_ca,src=<ca-bundle.crt> -t dlp-mlflow:3.16.1 services/mlflow
```
