# data_labeling

바디캠 영상에서 손·객체·도구 라벨 초안을 만들고, CVAT·Label Studio에서 사람이 수정한 이력을
데이터셋으로 저장하는 Python CLI 플랫폼이다.

**전문가에게 넘길 때는 [인수인계 첫 화면](docs/handover/README.md)부터 읽는다.**
현재는 단일 환경의 청소 영상 파일럿을 준비한 단계다. 실제 재학습기는 모의 구현이며, 다중 고객
SaaS의 인증·격리·복구와 미해결 의존성 취약점은 [남은 과제](docs/handover/known-issues.md)에 정리했다.

| 목적 | 안내 |
| --- | --- |
| 프로그램 구조·기능 상태 | [인수인계](docs/handover/README.md), [구조 지도](docs/handover/architecture.md) |
| 다른 환경에서 설치·검사 | [환경 준비](docs/handover/development.md) |
| 코드 수정·장애 복구 | [변경 및 복구 안내](docs/handover/change-guide.md) |
| 실제 촬영 영상으로 시험 | [청소 영상 파일럿](docs/weekend-video-pilot.md) |
| 개발 규칙·요구사항·결정 이유 | [CLAUDE.md](CLAUDE.md), [제품 계획](docs/labeling-platform-plan.md), [구현 계획](docs/ai-implementation-plan.md), [ADR](docs/adr/README.md) |

## 개발 시작

필요: Python 3.12, uv, make, Git, Docker/Compose v2, FFmpeg/ffprobe, MediaPipe용 EGL/GLES.
저장소 루트에서 실행한다. 모델·검수 계정·MongoDB 등 전체 준비 순서는 [환경 준비](docs/handover/development.md)를 따른다.

```bash
make install-curation
make env
# .env의 포트·개발 계정·저장소 설정을 확인한 다음
set -a
. ./.env
set +a
make up
make db-upgrade
make health
make check
```

`make check`는 서비스·격리 환경 테스트를 제외하며 모델이 없으면 생략되는 검사도 있다.
전체 검사에는 FiftyOne과 사용 가능한 내장 또는 외부 MongoDB가 필요하다.
실제 파일럿 전에는 `dlp pilot check`와 검수 계정의 권한 확인이 필요하다.
현재 작은 클라우드는 저장소 밖의 별도 시작 구성을 사용하므로 [파일럿의 시작 절차](docs/weekend-video-pilot.md#이-클라우드-환경-시작)를 따른다.

## 서비스와 포트

아래는 개발 기본값이다. `.env`로 포트를 바꾼다. 서비스 헬스체크 통과는 인증·계정·실제 기능의
준비 완료를 의미하지 않는다. CVAT와 외부 큐레이션 MongoDB는 `make up`에 포함되지 않는다.

| 서비스 | 기본 주소 | 비고 |
| --- | --- | --- |
| PostgreSQL | localhost:5432 | 메타데이터 및 개발 서비스 DB |
| SeaweedFS S3 | http://localhost:8333 | 원본·라벨링·데이터셋·MLflow 버킷 |
| Label Studio | http://localhost:8081 | `make up`이 서비스 API 토큰을 활성화 |
| Prefect | http://localhost:4200 | 전체 CLI 자동 재개 큐는 미구현 |
| MLflow | http://localhost:5000 | 산출물은 S3 `dlp-mlflow` |
| lakeFS | http://localhost:8000 | 데이터셋 버전 |
| CVAT | http://localhost:8080 | 일반 개발 호스트의 `make cvat-up`; 현재 클라우드는 별도 절차 |
| MongoDB | localhost:27017 | 선택 큐레이션의 외부 DB; 별도 준비 |

TLS 프록시 뒤에서 MLflow 이미지 빌드에 추가 CA가 필요하면 빌드 시크릿으로 전달한다.
인증서 검증을 비활성화하지 않는다.

```bash
docker build --secret id=extra_ca,src=/path/to/ca-bundle.crt -t dlp-mlflow:3.16.1 services/mlflow
```

실제 영상·동의 정보·계정·모델 가중치는 Git에 넣지 않는다. 개인정보 검수와 모델 라이선스의
상업 이용 검토는 사람이 완료해야 한다.
