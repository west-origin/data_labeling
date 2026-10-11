# 환경 준비와 검증

[인수인계 첫 화면](README.md) · [청소 영상 실행](../weekend-video-pilot.md)

## 다른 개발 환경에서 재현

아래는 저장소 자체의 준비 순서다. 실제 고객 데이터가 없는 개발 환경에서 실행한다.
필요 도구는 Python 3.12, uv, make, Git, Docker/Compose v2, FFmpeg/ffprobe이며 MediaPipe에는
EGL/GLES 시스템 라이브러리가 필요하다(Ubuntu: `libegl1`, `libgles2`). 잠금 파일은 갱신하지 않는다.

```bash
make install-curation
make env
```

전체 타입 검사와 큐레이션 테스트에는 선택 그룹 FiftyOne이 필요해 `install-curation`을 쓴다.
FiftyOne을 설치하지 않으면 전체 타입 검사가 실패할 수 있으므로 테스트 생략만으로 끝나지 않는다.
`make install`을 다시 하면 선택 그룹이 제거된다. `.env`는 기존 파일을 덮어쓰지 않는다.
서비스 기동 전에 `.env.example`의 설명을 따라 포트 충돌·개발 계정·S3/lakeFS 설정을 확인하고,
공유 서버에서는 접근 범위를 제한한다. `.env`나 계정 파일의 내용을 로그·Git에 넣지 않는다.

```bash
set -a
. ./.env
set +a
make up
make db-upgrade
make health
```

`make up`에는 PostgreSQL·SeaweedFS S3·Label Studio·Prefect·MLflow·lakeFS와 버킷 초기화가
포함된다. **CVAT와 외부 MongoDB는 포함되지 않는다.** DB 마이그레이션은 현재 head를 따르고,
온톨로지 1.0.0을 등록한다. 기존 운영 DB는 별도 백업·이관 절차를 거친다.

기본 단위 테스트에도 FiftyOne의 실제 데이터셋 쓰기·읽기가 포함된다. 사용 가능한 내장 또는 외부
MongoDB가 필요하며, DB 연결 실패는 테스트 생략이 아니라 실패가 될 수 있다. 외부 MongoDB 7을
사용하려면 별도로 준비하고 `FIFTYONE_DATABASE_URI`를 설정한다. 기존 서버가 없을 때의 로컬 예시는 아래와 같다.
볼륨은 중지 후에도 유지한다. 이름/포트가 이미 사용 중이면 기존 서비스를 확인한다.

```bash
docker run -d --name dlp-curation-mongo --restart unless-stopped \
  -p 127.0.0.1:27017:27017 -v dlp-curation-mongo:/data/db mongo:7.0
export FIFTYONE_DATABASE_URI=mongodb://127.0.0.1:27017
export FIFTYONE_DO_NOT_TRACK=true
docker exec dlp-curation-mongo mongosh --quiet \
  --eval 'quit(db.adminCommand({ping:1}).ok === 1 ? 0 : 1)' && make check
```

Mongo가 아직 초기화 중이면 위 ping이 실패한다. 준비 상태를 확인한 뒤 ping과 검사를 다시 실행한다.

모델 없이도 정적 검사·대부분의 단위 테스트를 실행할 수 있지만 실제 추론 테스트는 생략될 수 있다.
실제 손·객체·도구 파일럿에는 다음 준비가 필요하다. CPU에서도 동작하지만 처리시간은 실측한다.
깊이 모델 변환은 첫 파일럿의 필수 항목이 아니다.

```bash
make models
uv run --no-sync dlp pilot check --profile cleaning --models-only
```

네트워크 차단·인증서 오류는 프록시/CA 설정으로 해결하고 TLS 검증을 끄지 않는다.
MLflow 이미지 빌드의 추가 CA는 [서비스 설명](../../README.md#서비스와-포트)을 참고한다.

## 검수 도구와 실제 계정 연결

충분한 이미지·볼륨 공간이 있는 일반 개발 호스트에서 CVAT를 별도로 기동한다.
아래 전체 CVAT Compose는 현재의 작은 클라우드에서 사용할 절차와 다르다.

```bash
make cvat-up
make cvat-superuser
uv run --no-sync dlp services check --include-cvat
```

관리자 생성은 최초 한 번이며 이미 있으면 실패한다. 관리자는 설정에만 사용하고, 실제 검수는
CVAT의 일반 user/worker 계정으로 한다. 원본 권한자와 일반 라벨러를 분리한다.
실제 라벨러에게 작업을 열기 전에 `dlp services check --security`로 라벨링 버킷 버전 관리가 꺼져
있고 라벨러 키가 라벨링 버킷 읽기만 하는지 확인한다 (라벨러 키로 쓰기·삭제를 시도해 403을 본다.
허용되면 써진 빈 점검 객체를 서비스 키로 지운다, F-REV-19).
Label Studio에도 서비스 계정과 별개의 실제 검수 계정을 준비한다.

기본 검수 정책은 권한자 목록이 비어 있다. 실제 계정 연결을 Git의 기본 정책에 넣지 않고,
Git에서 제외된 로컬 파일로 분리한다. 아래 복사는 파일이 없을 때만 수행한다.

```bash
mkdir -p data/local-config
if [ ! -e data/local-config/review.yaml ]; then
  cp config/policies/review.yaml data/local-config/review.yaml
fi
export DLP_REVIEW_POLICY_FILE="$PWD/data/local-config/review.yaml"
```

이 파일의 `reviewers.privacy`에 승인받은 원본 검수자 ID를, `cvat.users`에 두 검수자 ID와 실제
CVAT 사용자 이름을 연결한다. [파일럿 안내](../weekend-video-pilot.md)의 명령을 그대로 쓰려면
ID는 `pilot-privacy`, `pilot-labeler`로 맞춘다. 일반 검수자 ID는 `config/policies/pilot.yaml`의
`labeler`와 일치해야 한다. 계정 생성과 원본 열람 승인 자체는 이 YAML이 대신하지 않는다.

```bash
uv run --no-sync dlp pilot check --profile cleaning
```

준비 검사는 서비스 응답·DB head·모델 실행·CVAT 일반 계정·Label Studio 서비스 계정 인증을
확인한다. 일반 라벨러의 원본 작업 접근 거부와 Label Studio 검수 계정 로그인도 별도로 확인한다.
`DLP_REVIEW_POLICY_FILE`은 검수 정책을 읽는 `dlp` CLI 명령이 모두 같은 로더
(`dlp_cli.review_config.load_review_policy`)로 따른다: 준비 검사, 검수 명령(`dlp review create|collect|
serve|assign|register-webhooks|plan|qa|quality`, 작업의 CVAT 계정 배정·웹훅 담당자 확인·워터마크
매체 설정·표본 판정 포함), 운영 명령(`dlp ops …`). 고른 파일이 없거나 깨졌으면 기본 정책으로
되돌리지 않고 실패한다. 한 명령 안에서는 정책을 한 번 읽어 `ReviewSetup`과 명령이 같은 값을 본다.
매체 회수(`dlp review revoke`, cron `revoke --pending`, `dlp privacy detect|render`와 검수 명령의 커밋
뒤 회수)는 검수 정책 값을 쓰지 않는다. 회수가 스스로 만드는 설정은 이 파일을 읽지 않으므로 파일이
깨져도 낡은 라벨링 매체는 지운다 (ADR 0039). 예외로 고아 정리(`dlp review revoke --orphans`)는 나이
하한(`orphans.min_age_s`)을 이 파일에서 엄격히 읽는다. 이 키는 필수라 별도 정책 파일에도 넣어야 한다.
반면 라이브러리 기본 로더(`dlp_review.ops.policy.load_policy(root)`,
정책을 넘기지 않은 `ReviewSetup`의 지연 로더)는 이 변수를 보지 않고 Git의 기본 `review.yaml`을 읽는다.
그래서 라이브러리 함수를 직접 부르는 테스트는 기본 정책을, CLI 명령 함수를 부르는 테스트는 이 변수를 따른다.
테스트에서는 파일럿 권한 설정이 테스트 픽스처를 덮어쓰지 않도록 서브셸 안에서만 해제한다.
검사가 끝나면 현재 셸의 정책 연결이 유지되므로 파일럿 명령을 이어서 실행할 수 있다.
첫 검사가 실패하면 서비스 검사를 실행하지 않고 실패를 반환한다.

```bash
(
  unset DLP_REVIEW_POLICY_FILE
  make check && make test-services
)
```

## 검사 범위와 자원

| 명령 | 보장하는 범위 | 별도 준비·생략 가능성 |
| --- | --- | --- |
| `make check` | ruff·포맷·pyright·온톨로지·생성 스키마·라이선스 분류·단위 테스트 | 모델 없으면 일부 생략; 전체 검사에 FiftyOne·사용 가능한 Mongo 필요; services/isolated_env 제외 |
| `make test-services` | DB·S3·lakeFS·MLflow·검수 도구 통합 | 도구 미기동 시 일부 생략; isolated_env 제외 |
| `make test-isolated` | 공식 LeRobot 격리 환경 쓰기·읽기 등 | 일부는 DB 필요; 추가 PyTorch 환경 약 1.5 GB |
| `make export-models` | 깊이 모델 ONNX 변환 준비 | 큰 다운로드·추가 CPU 환경; 정확도 판정 아님 |
| `make coverage` | 기본 테스트의 줄 커버리지(모듈별, 실행 안 된 모듈은 0%) | 하한 없음; 서비스 테스트는 `--cov-append`로 합침 |
| `make review-scope` | 추적 파일별 영역·위험 등급·관련 미해결 문제 수 (reports/review-scope.csv) | 위험 등급 규칙은 scripts/review_scope.py |
| `dlp pilot check` | 선택한 파일럿의 모델·서비스·검수 계정 준비 | 동의·실영상 정확도·사람 검수 완료는 판정하지 않음 |

`pytest` 결과의 passed/failed/skipped/deselected를 함께 기록한다. 성공 종료만으로 모든 모델·도구를
검사했다고 쓰지 않는다. CI는 [ci.yml](../../.github/workflows/ci.yml)의 check/services와
일일·수동 CVAT 작업으로 나뉜다. 이 클라우드에서의 로컬 통과는 CI 실행 결과를 대신하지 않는다.

`.venv`·모델·영상·서비스 이미지·볼륨·PyTorch 격리 환경은 모두 디스크를 쓴다. Docker VFS는 이미지
복제로 사용량이 커진다. 일괄 설치 전에 `df -h /workspace`(일반 호스트는 해당 디스크)와
`docker system df`를 확인한다. 실패 복구에 `make clean`을 쓰면 개발 DB·버킷 볼륨이 삭제된다.

## 현재 클라우드에만 있는 준비

`/workspace/.cloud-onboarding/data-labeling/`은 저장소 밖의 환경별 설치·시작·CA/라이브러리·
스토리지 설정이다. 해당 파일과 실제 계정 파일은 다른 개발자의 clone에 따라가지 않는다.
현재 환경에서는 [파일럿 안내의 시작 절차](../weekend-video-pilot.md#이-클라우드-환경-시작)를 사용한다.

| 항목 | 현재 클라우드 | 새 환경에서 필요한 확인 |
| --- | --- | --- |
| CVAT | v2.78.0 최소 서버/UI와 필수 큐를 단일 서버 컨테이너에 구성 | 공식 전체 Compose와 자원·작업자 구성이 다름 |
| MongoDB | 저장소 밖 Compose로 별도 기동 | 외부 Mongo URI와 실제 CRUD 검사 |
| Python/미디어 | Python 3.12.14, uv 0.12.19, FFmpeg, 별도 GLES 런타임 | 시스템 라이브러리·PTS·모델 실행 |
| 모델 | CPU 파일럿 모델 준비, 깊이 변환 제외 | 해시·라이선스 분류·실제 실행 검사 |
| 계정 | 분리된 일반 검수 계정·로컬 정책, 비밀번호 별도 0600 파일 | 실제 계정 생성·권한·로그인·촬영 승인 |
| 설치 재사용 | 설치/시작 절차 초안 저장 | 새 환경 복원 검증·공개 설정 적용은 별도 |

현재 환경에서 일반 `make cvat-up`으로 구성을 덮어쓰지 않는다. 이전 준비 결과를 새 호스트의
실행 성공으로 간주하지 않고, 위 준비 검사를 다시 통과시킨 뒤 실제 영상을 넣는다.
