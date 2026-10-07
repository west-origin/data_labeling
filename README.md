# data_labeling

청소·요양돌봄·간호 작업자의 1인칭 바디캠 영상을 로봇 모방학습용 데이터로 라벨링하는 플랫폼.

- 계획: [docs/labeling-platform-plan.md](docs/labeling-platform-plan.md)
- 구현 계획: [docs/ai-implementation-plan.md](docs/ai-implementation-plan.md)
- 개발자 안내 (처음 맡는 개발자용 전체 지도): [docs/developer-guide.md](docs/developer-guide.md)

## 구성

- `packages/schema` (`dlp_schema`): 세션·스트림·라벨·에피소드 그래프 계약 타입, 온톨로지 로더·검증, 온톨로지 이관, DB 스키마
- `packages/fixtures` (`dlp_fixtures`): 정답을 아는 합성 데이터 생성기 (`dlp fixtures generate`)
- `packages/media` (`dlp_media`): 세션 수집, PTS 인덱스, 프록시 영상, IMU·장갑 정규화 (`dlp ingest`, `dlp media`)
- `packages/sync` (`dlp_sync`): 멀티스트림 동기화 (`dlp sync run`, `dlp sync adjust`)
- `packages/privacy` (`dlp_privacy`): 프라이버시 게이트 (`dlp privacy detect|approve|render|audit-sample`)
- `packages/review` (`dlp_review`): 검수 도구 연동·검수 운영 (`dlp review create|collect|verify|serve|plan|assign|queue|qa|quality`)
- `packages/datasets` (`dlp_datasets`): 데이터셋 버전·분할·골든셋·계보 (`dlp dataset`, `dlp lineage`)
- `packages/models` (`dlp_models`): 모델 레지스트리(`config/models.yaml`, 해시·라이선스)와 공용 ONNX 런타임
- `packages/prelabel` (`dlp_prelabel`): 자동 프리라벨 (`dlp prelabel run`). 미연동 모델은 `make todo-models`
- `packages/relations` (`dlp_relations`): 관계 도출·도구-표면 접촉·표면 커버리지 (`dlp relations run`)
- `packages/actions` (`dlp_actions`): 행동 구간 경계 후보 + VLM 분류·설명 (`dlp actions run`)
- `packages/evaluation` (`dlp_eval`): 지표 라이브러리, 골든셋 평가, 배포 게이트 (`dlp eval golden`)
- `packages/training` (`dlp_train`): 재학습 루프와 모델 버전 배포 (`dlp train run|models|approve`)
- `packages/active` (`dlp_active`): 액티브 러닝 세션 순위, FiftyOne 연동 (`dlp active rank|fiftyone`)
- `packages/export` (`dlp_export`): COCO·구간 JSON·LeRobot v3.0 내보내기 (`dlp export coco|intervals|lerobot`)
- `packages/ops` (`dlp_ops`): 주간 운영 지표, 원본 접근 감사, 보관 만료 (`dlp ops weekly|audit-report|retention|...`)
- `packages/cli` (`dlp`): 명령줄 도구
- `config/ontology/v1`: 온톨로지 v1 초안, `config/defaults.yaml`: 미결정 사항 기본값
- `schemas/`: 계약 타입의 JSON Schema (생성 파일)

## 시작하기

필요: [uv](https://docs.astral.sh/uv/), Docker (Compose v2), make

```sh
make install        # Python 3.12 환경과 의존성
make models         # 모델 가중치 (YuNet 얼굴, MediaPipe 손·전신·객체). MediaPipe는 libegl1 libgles2 필요
make export-models  # 공개 ONNX가 없는 모델(메트릭 깊이)을 공식 가중치에서 변환 (일회용 PyTorch 환경)
make check          # 린트·타입·테스트
make up             # PostgreSQL, SeaweedFS(S3), Label Studio, Prefect, MLflow, lakeFS
make health         # 헬스체크
make db-upgrade     # 메타데이터 DB 마이그레이션 + 온톨로지 v1 등록
make cvat-up        # CVAT (선택, 이미지가 커서 별도)
```

| 서비스 | 주소 | 비고 |
| --- | --- | --- |
| PostgreSQL | localhost:5432 | DB: dlp, labelstudio, prefect, mlflow |
| SeaweedFS S3 | http://localhost:8333 | 버킷: dlp-raw, dlp-labeling, dlp-datasets, dlp-mlflow |
| Label Studio | http://localhost:8081 | 계정은 `.env`의 `DLP_LABEL_STUDIO_*`. `make up`이 API 토큰을 켠다 |
| Prefect | http://localhost:4200 | |
| MLflow | http://localhost:5000 | 산출물은 S3 `dlp-mlflow` |
| lakeFS | http://localhost:8000 | 데이터셋 버전. 키는 `.env`의 `DLP_LAKEFS_*` |
| CVAT | http://localhost:8080 | `make cvat-up`, 관리자는 `make cvat-superuser` |

포트와 계정은 `.env`(최초 `make up` 시 `.env.example`에서 복사)로 바꾼다. 기본값은 개발용이다.

TLS를 가로채는 프록시 뒤에서 MLflow 이미지 빌드가 실패하면, CA 묶음을 빌드 시크릿으로 넘겨 먼저 빌드한다:

```sh
docker build --secret id=extra_ca,src=<ca-bundle.crt> -t dlp-mlflow:3.16.1 services/mlflow
```
