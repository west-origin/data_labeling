# 개발자 안내 (외주 검수·리팩토링 인수인계용)

이 문서는 이 저장소를 처음 맡는 개발자가 하루 안에 전체 그림을 잡고, 검수·리팩토링을 시작할 수 있게
쓴 안내서다. 세부 결정의 이유는 `docs/adr/`에, 무엇을 왜 만드는지는 `docs/labeling-platform-plan.md`에,
작업 패키지(WP0~WP16)별 완료 기준은 `docs/ai-implementation-plan.md`에 있다. 코드 규칙은 `CLAUDE.md`가
기준이다 (이 문서와 다르면 `CLAUDE.md`가 맞다).

모든 소스 파일에는 한국어 모듈 docstring(역할, 파이프라인 위치, 관련 WP·ADR, 주의점)과 함수·클래스
docstring(인자·단위·반환·예외·부작용)이 달려 있다. 이 문서는 그 위의 지도다.

---

## 1. 무엇을 만드는 시스템인가

- 돌봄·청소 작업자가 몸에 단 카메라(바디캠, 1인칭)와 3인칭 카메라, 장갑 압력 센서, IMU로 찍은 세션을
  받아, 사람이 검수한 라벨 데이터셋을 만들어 **판매**한다 (소프트웨어가 아니라 데이터를 판다).
- 라벨 종류: 손 키포인트(hand21), 전신(coco17), 객체·도구 박스, 손 상태(접촉·파지), 3D 궤적, 관계
  (손-도구, 도구-표면 접촉), 표면 커버리지, 행동 구간·설명.
- 출력 형식: COCO, 구간 JSON(`dlp_schema.export.IntervalFile`), LeRobot v3.0 에피소드.
- 두 가지 강한 제약
  1. **프라이버시**: 얼굴·문서·화면·송장 등은 사람이 검수·승인한 블러를 입힌 영상(블러본)만 라벨러·
     모델·구매자에게 간다. 원본 영상 접근은 모두 감사 기록에 남는다.
  2. **상업 사용**: 데이터를 팔기 때문에 프리라벨에 쓰는 모델의 가중치·학습 데이터 라이선스를 분류한다
     (`config/models.yaml`, ADR 0010). `forbidden`은 쓰지 않고, `review`는 판매 전 법률 검토 대상이다.

## 2. 개발 환경 시작하기

| 단계 | 명령 | 비고 |
|---|---|---|
| 의존성 | `make install` (FiftyOne까지는 `make install-curation`) | uv 잠금 파일 기준. pyright가 `dlp_active.curation`을 검사하려면 curation 설치가 필요하다 |
| 정적 검사 + 단위 테스트 | `make check` | ruff, 포맷, pyright strict, 계약(JSON Schema), 라이선스, pytest. PR 전에 반드시 통과 |
| 서비스 기동 | `make up` → `make db-upgrade` | PostgreSQL, SeaweedFS(S3), Label Studio, Prefect, MLflow, lakeFS. 처음 한 번 온톨로지 v1 등록 |
| 서비스 테스트 | `make test-services` | `@pytest.mark.services` (실행 중 서비스 대상, 테스트마다 임시 DB를 만든다) |
| 격리 환경 테스트 | `make test-isolated` | LeRobot 공식 API를 `scripts/lerobot-env/uv.lock` 고정 환경에서. `make up` 필요 |
| CVAT | `make cvat-up` / `make cvat-superuser` | 공식 compose, 고정 버전. 블러 검수·공간 라벨 검수 도구 |
| 모델 가중치 | `make models` (`make export-models`는 메트릭 깊이 ONNX 변환) | 저장소에 넣지 않는다. 없으면 관련 테스트는 건너뛴다 |
| 합성 데이터 | `make fixtures` | 실제 영상·개인정보는 저장소에 넣지 않는다 |

환경 변수는 `.env.example`에 설명과 함께 있다. 개발용 가명화 비밀값은 `DLP_ENV=dev`에서만 허용된다.

## 3. 저장소 구조

```
packages/<이름>/src/<모듈>/   uv 워크스페이스 멤버 (각자 tests/)
config/                       온톨로지·정책·기본값 YAML (코드에 값 하드코딩 금지)
schemas/                      생성된 JSON Schema (`make schemas`, 직접 편집 금지)
services/                     개발용 docker compose, CVAT 실행 스크립트
scripts/                      격리 환경 스크립트(LeRobot), 모델 변환
tests/                        패키지를 가로지르는 통합 테스트 (종단 테스트 포함)
docs/adr/                     아키텍처 결정 기록 (0001~)
```

### 패키지와 의존 방향

아래에서 위로만 의존한다. `dlp_schema`는 모든 패키지가 쓰는 계약이고, `dlp_cli`는 맨 위에서 엮기만 한다.

| 층 | 패키지 | 한 줄 역할 | 주요 명령 |
|---|---|---|---|
| 계약 | `dlp_schema` | 계약 타입, 온톨로지, DB 테이블·마이그레이션·저장소 함수, 이력 도우미 | `dlp schema`, `dlp ontology`, `dlp db` |
| 기반 | `dlp_fixtures` | 정답을 아는 합성 데이터 생성기 (테스트용) | `dlp fixtures generate` |
| 기반 | `dlp_media` | 수집: 원본 저장(불변·멱등), PTS 인덱스, 프록시, IMU·장갑 정규화, 원본 접근 감사 저장소 | `dlp ingest`, `dlp media` |
| 기반 | `dlp_models` | 모델 레지스트리(해시 확인)와 공용 ONNX 런타임(OWLv2, 메트릭 깊이) | `dlp models` |
| 단계 | `dlp_sync` | 스트림 시계를 바디캠(마스터 타임라인)에 맞춤 | `dlp sync run\|adjust` |
| 단계 | `dlp_privacy` | 블러 탐지·추적, 승인, 블러본 렌더, 잔여 누락 감사 | `dlp privacy detect\|approve\|render\|audit-sample` |
| 단계 | `dlp_prelabel` | 자동 프리라벨(손·전신·객체·도구), 3D 궤적, 접촉, 3인칭 착용자 매칭 | `dlp prelabel run` |
| 단계 | `dlp_relations` | YAML 규칙 엔진 관계 도출, 도구-표면 접촉, 표면 커버리지 | `dlp relations run` |
| 단계 | `dlp_actions` | 행동 구간: 경계 후보 → VLM 분류(JSON Schema 강제) → 병합 | `dlp actions run` |
| 단계 | `dlp_review` | CVAT·Label Studio 변환(무손실 왕복), 작업 생성·수거, 웹훅, 검수 운영(배정·표본·QA·품질), 검수 완료 판정 | `dlp review …` |
| 단계 | `dlp_datasets` | 데이터셋 버전(lakeFS), 작업자·장소 단위 분할, 골든셋, 사용 중지, 계보 | `dlp dataset …`, `dlp lineage` |
| 단계 | `dlp_eval` | 지표 라이브러리(참조 구현 일치), 골든셋 평가 하네스, 배포 게이트 | `dlp eval golden` |
| 단계 | `dlp_train` | 재학습 루프, MLflow 기록, 모델 레지스트리, 게이트 후 배포 | `dlp train run\|models\|approve` |
| 단계 | `dlp_active` | 수정률 기반 세션 순위, FiftyOne 큐레이션(선택 설치) | `dlp active rank\|fiftyone` |
| 단계 | `dlp_export` | COCO·구간 JSON·LeRobot 내보내기 (데이터셋 버전에서만) | `dlp export …` |
| 단계 | `dlp_ops` | 주간 운영 지표, 원본 접근 월간 감사, 원본 보관 만료 | `dlp ops …` |
| 진입점 | `dlp_cli` | `dlp` 명령. `<단계>_cmds.py`가 정책·저장소·DB를 엮어 각 패키지 함수를 부른다 (로직 없음) | `dlp …` |

## 4. 데이터 흐름과 운영 순서

### 세션 생애주기와 프라이버시 상태

- 생애주기(`LifecycleState`, 한 방향으로 한 칸씩, 어디서든 `withdrawn` 가능):
  `raw_ingested → privacy_approved → prelabeled → human_verified → split_assigned → exported`.
  전이는 `session_lifecycle_events`에 시각·실행자와 함께 추가만 된다 (ADR 0028, 0029).
- 프라이버시 상태(`PrivacyState`): `pending → auto_blurred → approved`. 블러 라벨이 바뀌면(재탐지·검수
  수거) `auto_blurred`로 돌아가고 렌더 기록이 무효가 된다 (ADR 0024).

### 정상 운영 순서

1. 처음 한 번: `make up` → `dlp db upgrade` → `dlp ontology register 1.0.0` (선택: `dlp models fetch`)
2. 세션마다
   1. `dlp ingest <매니페스트>` — 원본 불변 저장, 파생 파일, 세션 등록
   2. `dlp sync run <세션>` (필요하면 `dlp sync adjust`)
   3. `dlp privacy detect <세션>` → `dlp review create <세션> --stage privacy --assignee <권한자>`
      (또는 `dlp review plan <세션> --privacy --reviewer <권한자>` → `dlp review assign <배정>`) →
      CVAT에서 블러 검수 → `dlp review collect <작업 키> --reviewer <권한자>`(또는 웹훅 `serve`) →
      `dlp privacy approve` → `dlp privacy render`
   4. `dlp prelabel run` → `dlp relations run` → `dlp actions run --vlm-url …`
   5. `dlp review plan <세션> --reviewer …` → 배정마다 `dlp review assign` → 검수 → 수거 →
      `dlp review qa`(뽑힌 QA 배정도 assign·수거) → `dlp review verify` (열린 배정·작업이 남으면
      완료가 아니다)
3. 데이터셋 이후: `dlp dataset golden --domain <도메인> --create <버전>`(도메인마다 처음 한 번,
   프라이버시 승인 세션에서 제안한 뒤 사람이 처음부터 라벨링) → `dlp dataset build <버전> --golden
   <골든셋>` → `dlp train run`(→ `deploy: approve` 과제는 `dlp train approve`) / `dlp eval golden` →
   `dlp export coco|intervals|lerobot`. 빌드 후보는 프라이버시 승인 세션 전체이고, 검수 완료
   (`human_verified`) 세션만 `split_assigned`로 옮겨진다 (ADR 0031)
4. 주기적으로: `dlp ops weekly`, `dlp privacy audit-sample` → `dlp ops privacy-audit`,
   `dlp ops audit-report`(매월), `dlp ops retention`, `dlp active rank`

### 버킷

| 버킷 | 내용 | 누가 |
|---|---|---|
| 원본 (`dlp-raw`) | 원본 영상·센서, 프록시, PTS 인덱스, 장갑·IMU 정규화본, 동기화 보고서, 검수 우선 구간, 탐지 표시 | 원본 접근 권한자만. 모든 접근이 감사 기록에 남는다 |
| 라벨링 (`dlp-labeling`) | 블러본과 렌더 기록(`.render.json`), 검수용 워터마크 영상, 시계열 CSV | 라벨러는 이 버킷 읽기 전용 자격 증명으로 서명된 URL만 |
| 데이터셋 (`dlp-datasets`) | 데이터셋 스냅샷(lakeFS), 내보내기 결과, 내부 가명 대응표 | 운영자 |
| MLflow (`dlp-mlflow`) | 학습 산출물 | 재학습 루프 |

## 5. 꼭 알아야 할 규약 (위반하면 판매 데이터가 틀어진다)

1. **시각** — 모든 시간은 정수 ms, 프레임 번호는 저장하지 않는다. 시간 구간 라벨(행동·접촉·관계 등)은
   마스터 타임라인(바디캠 시계), 공간 라벨(박스·마스크·키포인트·블러·3D 궤적) 키프레임은 그 스트림
   영상의 PTS 시각이다 (ADR 0019). 영상 시각은 PTS 인덱스로만 계산한다.
2. **라벨 불변** — 덮어쓰지 않는다. 수정은 새 레코드 + `parent_label_id`, 삭제는 `retracted` 레코드.
   DB 트리거가 DELETE·TRUNCATE와, 검수 상태(verification) 밖 열의 UPDATE를 막는다 (검수 상태만
   `record_review`로 갱신된다).
3. **운영 라벨** — 다른 단계의 입력은 `dlp_schema.episode.current_labels()`로 고른다. 오류 삽입
   (`seeded_error`)·측정(`measurement`) 레코드와 그 후손은 학습·내보내기에 들어가면 안 된다.
4. **멱등과 모델 버전** — 각 단계는 다시 돌려도 같은 결과다. 모델 버전에는 가중치 해시, 정책 절 해시,
   (필요하면) 입력 라벨 ID 해시가 들어간다. 다시 돌릴지는 전체 이력(`get_labels`)으로 정하고, 버전이
   바뀌면 검수 전인 이전 결과만 `retractions()`로 지운다. **검수자가 승인·수정·표본 검증한 라벨은 어떤
   단계도 지우지 않는다** (ADR 0015, 0019, 0026).
5. **원본 접근** — 원본 버킷 저장소는 `dlp_cli.raw_access.raw_store`(DB 없으면 `raw_store_offline`)로만
   만든다. 정적 검사 테스트(`tests/test_raw_access_audited.py`)가 소스 텍스트(주석 포함)를 훑어 우회를
   막으므로, 소스 주석에 원본 버킷 이름을 따옴표로 쓰지 않는다.
6. **블러본만** — 라벨러·VLM·FiftyOne·내보내기는 지금 승인 기준으로 렌더된 블러본인지 확인한 뒤에만
   쓴다 (`dlp_privacy.runner.assert_render_current`, 내보내기는 `expected_render_hash` +
   `check_fetched`).
7. **정책 값은 config에서** — 비율·허용 오차·임계값은 `config/policies/*.yaml`, `config/defaults.yaml`.
   정책 해시는 파싱된 값으로 계산하므로 YAML 주석은 해시에 영향이 없다.
8. **분할 격리** — 골든·학습·검증 사이에 작업자나 장소가 겹치면 안 된다. 분할은 `dlp_datasets.splitter`
   로만 만든다.
9. **계약 변경** — 모듈 사이 데이터는 `dlp_schema` 타입으로만. 계약을 바꾸면 ADR + Alembic 리비전 +
   계약 테스트 + `make schemas`를 함께 한다. Pydantic 클래스 docstring과 `Field(description=…)`는 JSON
   Schema에 들어가므로 바꾸면 `make schemas`가 필요하다.
10. **내보내기** — 데이터셋 버전에서만 만든다. 기본은 사람이 만들었거나 승인·수정·표본 검증한 라벨만.
    블러 라벨·원본 위치·검수자 ID는 어떤 형식에도 없다. 작업자·장소·세션·라벨 ID는 내보내기마다 다른
    HMAC 가명이다 (ADR 0027). 이력을 먼저 커밋하고 파일을 올린 뒤 manifest를 마지막에 올린다.

## 6. 테스트 전략

- 알고리즘 테스트는 `dlp_fixtures`의 정답을 아는 합성 데이터로 쓰고, 생성기 정답과 비교한다
  (동기화 오프셋·드리프트, 블러 대상 박스, 행동 경계, 접촉 구간 등).
- 마커: 기본(`make check`), `services`(실행 중 서비스), `isolated_env`(LeRobot·PyTorch 일회용 환경).
- 종단 테스트 `tests/test_end_to_end.py`: 합성 세션 하나를 수집부터 내보내기까지 패키지 함수로 통과시킨다.
- 지표는 참조 구현과 일치 테스트가 있다 (pycocotools, TrackEval, MS-TCN, ActivityNet, scikit-learn).
- CI(`.github/workflows/ci.yml`): check 잡(FiftyOne·모델 가중치까지 받고 `make check`), services 잡
  (`make up` → 서비스·격리 테스트), CVAT까지 띄우는 services-cvat 잡(매일 예약·수동 실행 때만).

## 7. 아직 실제 모델이 아닌 곳과 사람이 정할 값

`make todo-models`가 `TODO(real-model):` 표시 목록을 보여 준다. 현재:

- 재학습기: `oracle-stub`만 있다 (객체 YOLOX·RTMPose 미세조정, 접촉 영상 분류기 등이 후보).
- VLM 서버(OpenAI 호환, GPU면 vLLM, CPU면 llama.cpp): 요청 형식만 테스트했다.
- 글자 탐지(OCR) 블러 탐지기, 도구 작용부 마스크, 바디캠 6자유도 궤적(Basalt 등), 영상만으로 접촉을
  판정하는 분류기, 실제 영상의 작용부·표면 꼭짓점 3D 궤적.
- 배포된 행동 모델을 `dlp actions run`이 아직 불러오지 않는다.

사람이 정할 값: 시간당 인건비(`ops.yaml cost.hourly_cost`), 원본 보관 기간
(`defaults.yaml retention.raw_retention_days`), 블러 검수 권한자(`review.yaml reviewers.privacy`와
`cvat.users`), 성공 기준, `review` 등급 모델의 법률 검토.

## 8. 검수 이력

전체 검수를 네 번 했고 결과는 ADR에 남아 있다: 0015(1차), 0019(2차), 0021~0023(3차), 0024~0028(4차).
0029는 4차 뒤의 검수 완료 판정 결정이다.
주석 작업 중 발견한 의심 22건은 5차 정정으로 ADR 0030·0031에 남겼다. 아래 9절은 아직 고치지 않은 것이다.

## 9. 알려진 한계와 고치지 않은 의심

주석 작업에서 나온 의심 22건 중 영향이 큰 것은 ADR 0030·0031에서 고쳤다. 아래는 일부러 남겨 둔
한계와, 영향이 작아 고치지 않은 의심이다. 실제 영상으로 시험한 뒤 우선순위를 다시 정하길 권한다.

### 운영·보안 한계 (알고 운영해야 함)

| 항목 | 내용 | 관련 |
|---|---|---|
| CVAT에 올린 원본 프록시 | 블러 검수용으로 CVAT에 올린 원본 사본은 보관 기간·사용 중지 뒤에도 지우지 않는다 (CVAT 삭제 연동 없음) | ADR 0021 |
| 접근 기록의 실행자 | `DLP_ACTOR`·OS 사용자로 스스로 밝히는 값이라 강한 신원 증명이 아니다 | ADR 0020, 0021 |
| CVAT 프로젝트 공개 범위 | `dlp-privacy` 프로젝트를 CVAT 조직·권한으로 막는 것은 문서로만 안내 (자동화 안 함) | ADR 0024 |
| 가명 연결 | ID 가명은 ID로 잇는 것만 막는다. 같은 세션의 두 내보내기는 내용(프레임·라벨 값)으로 맞춰 볼 수 있다 | ADR 0027 |
| lakeFS 스냅샷 | 여러 파일을 올리다 중간에 실패하면 커밋되지 않은 객체가 브랜치에 남아 다음 버전에 섞일 수 있고, 빌드 DB 롤백 뒤 lakeFS 커밋이 고아로 남는다 | `dlp_datasets.snapshot`, `build` |
| 검수 중 재생성 | 상자 수정으로 접촉이 다시 만들어질 때, 열린 Label Studio 작업에서 검수자가 새로 그린 구간이 새 모델 접촉과 겹칠 수 있다 (다음 검수에서 정리) | ADR 0026 |

### 정확도 한계 (실제 영상에서 확인 필요)

- 카메라 내부 파라미터: 화각 근사에서 주점을 `width/2`로 두고 렌즈 왜곡을 무시한다
  (`dlp_models.depth`, `dlp_prelabel.lift3d`). 광각 액션캠은 화면 가장자리 3D 오차가 크다 — 보정값을
  세션 메타데이터로 받는 것이 다음 단계.
- 평가: 블러 정밀도와 객체 오탐은 정답 키프레임 시각에서만 센다 (ADR 0013), 커버리지 평가는 정답에 없는
  (표면, 도구) 예측을 무시한다, 파지·동사 macro F1은 놓친 정답을 "missing" 클래스로 넣어 벌점을 준다
  (의도된 설계로 보이나 확인 필요). `cohen_kappa`는 기대 일치도 1이면 1.0 (scikit-learn은 NaN).
- 접촉: merge_gap으로 이은 두 접촉 구간의 점을 이어 커버리지가 조금 커질 수 있다
  (`dlp_relations.contact`, `coverage`). 장갑 히스테리시스 시작·끝 경계 조건이 `>`/`>=`로 비대칭.
- 3인칭 착용자 매칭: `wearer.py`의 `offset_ms` 인자 미사용, 키프레임 2개면 속도를 0으로 본다.
- OWLv2 질의가 16토큰을 넘으면 잘리며 EOS가 빠진다 (지금 질의는 모두 짧다).
- RTMPose: rtmlib YOLOX 출력이 점수 순이라는 보장을 확인하지 못했다 (사람이 `max_people`보다 많을 때).
- 동기화: 끝 슬레이트를 찾을 때 seek 직후 첫 프레임에 슬레이트가 이미 보이면 버릴 수 있다
  (`dlp_sync.slate`). `sync adjust`는 ms를 실수로 받는다 (계약도 실수).
- 행동: VFR 구간에서 VLM에 요청한 장수보다 적은 프레임을 보낼 수 있다, VLM 4xx(설정 오류)도 장애로
  처리돼 백오프 후 중단된다, gap 응답의 `tool_id`는 위반으로 잡지 않는다.

### 작은 의심

- `dlp_review.reconcile`: 128자를 넘는 ID는 앞을 잘라 접두사가 사라질 수 있다 (지금 판정에는 영향 없음).
- `dlp_review.clients.lead_seconds`: 취소·서비스 계정 주석의 lead_time도 더한다 (프리라벨은 예측으로
  올리므로 현재 영향 없음).
- `dlp_review.timeseries`: `rate_hz`가 1000의 약수가 아니면 `time_ms`가 정수가 아니다 (기본 50 Hz는 정상).
- `dlp_schema.episode.EpisodeGraph`: 공간 라벨 구간(스트림 PTS)을 마스터 구간과 그대로 비교한다.
- `dlp_schema.db.repository`: `_is_additive`가 목록 원소 추가를 덧붙이기로 보지 않는다 (의도 확인),
  `set_privacy_state`는 전이 순서를 검사하지 않고 기록도 남기지 않는다.
- `dlp_privacy`: 여유를 더한 박스가 화면 밖으로 잘린 관측 프레임도 score가 남아 저신뢰 구간에 들어갈 수
  있다 (검수 구간이 늘어나는 쪽), 렌더 결과의 `applied`가 실제로 칠하지 않은 박스도 센다.
- `dlp_train`: 누적 기준(`min_new_examples`)이 rejected 모델 뒤에도 걸린다 (의도 확인), MLflow
  실험 조회가 200이 아니면 오류여도 생성을 시도한다.

## 10. 리팩토링 후보

주석 작업 중 각 패키지 담당이 모은 목록이다. 동작은 바꾸지 않았다. 위험도는 리팩토링 자체의 위험이다.

### 우선 검토 (위험도 중)

| 위치 | 무엇 | 왜 |
|---|---|---|
| `dlp_cli` 여러 `*_cmds.py` | 온톨로지를 `config/ontology/v1`로 고정해 읽음 | 세션 온톨로지 버전이 바뀌면 어긋난다. 세션의 `ontology_version`으로 고른다 |
| `dlp_review.clients`, `dlp_datasets.snapshot` | CVAT·Label Studio·lakeFS 호스트가 `localhost` 고정 (포트만 설정) | 원격 배포가 안 된다. S3·MLflow처럼 URL 환경 변수로 |
| `dlp_ops.metrics.weekly_metrics`, `dlp_review.ops.runner`(`known_classes`, `_open_loads`, `existing`), `dlp_active.select.rank_sessions`, `dlp_privacy.audit.audit_candidates` | 매번 모든 세션의 전체 라벨 이력·배정을 읽음 | 세션 수에 비례해 느려진다. 집계 쿼리나 증분 계산으로 |
| `dlp_export.runner.run_export` (약 190줄) | 고르기·쓰기·manifest·이력·올리기·전이가 한 함수 | 단계 함수로 나눈다. **이력 선커밋 → 파일 → manifest 순서는 반드시 보존** |
| `dlp_export.lerobot.build_episode` | 프레임 루프 안에 손 2D·3D·상태·도구·행동·작업이 섞임 | 묶음별 함수로 |
| `dlp_review.tasks` Label Studio 작업 생성 | `dlp review create` 직접 경로에서 바디캠마다 작업, 시계열 CSV 키 덮어씀 | 기준 스트림 하나만 |
| `config/defaults.yaml golden_set` 절 | `split_unit`, `min_instances_per_class`를 코드가 읽지 않음 | 연결하거나 지운다 |
| `dlp_schema.validation.check_label` | 테스트에서만 호출, 운영 경로(저장·검수 수거)는 온톨로지 대조를 안 거침 | 저장 경로에 연결 |

### 정리 (위험도 하)

- 중복 코드: 마스터→스트림 시각 변환(`dlp_export.frames.stream_ms`, `dlp_active.curation._stream_ms` →
  `Stream` 메서드로), 영상 종류 상수(`VIDEO_KINDS`/`VIDEO`) 10곳, 박스 IoU 4곳(`dlp_models.owlv2._iou`,
  `dlp_prelabel.common.iou`, `dlp_review.ops.priority._iou`, `dlp_privacy.geometry.Box.iou`),
  `version_tag` 2곳(`dlp_actions.assemble`, `dlp_schema.episode`), P/R/F1 계산 3곳(`dlp_eval.harness`),
  오디오 상관 호출(`dlp_sync.pipeline`), 임시 PostgreSQL `pg` 픽스처 여러 벌(공용 conftest로).
- 하드코딩: `dataset_cmds` 도메인·온톨로지 기본값, `train_cmds` 과제 선택지, `dlp_media.proxy` 인코더
  rate 30, `dlp_review.tasks` 골격 이름(`kp_hand21` 등), `dlp_fixtures.sessions`의 `rng.choice(3)`.
- 경계·타입: `dlp_train.deployed`가 `loop._download`(비공개)를 import, `boxes_to_labels`가
  mediapipe 모듈에 있고 `Any` 타입, `dlp_eval.gate`의 `type: ignore`, `dlp_review.collect`가
  `cvat.http`를 직접 호출(4xx가 KeyError로).
- 성능(작은 데이터에선 무해): `dlp_datasets.splitter.assign_splits` O(작업자²×세션),
  `dlp_eval.metrics.detection` O(C·I·N), `dlp_sync.taps.match_taps` O(R²·T²), `dlp_media.pts.nearest`,
  `dlp_privacy.tracker` 보간, `dlp_fixtures` `_wrist_track`.
- 기타: `dlp_train.tracking`의 `httpx.Client` 미종료, `dlp_cli.review_cmds.cmd_register` 웹훅 중복 등록,
  `dlp_schema.db.repository` 예외 종류 불일치(`NoResultFound`/`KeyError`), `insert_*`의 datetime 되돌리기
  반복, `dlp_actions.runner.KINDS` 미사용, 이름 혼동(`dlp_review.collect`의 `retracted`,
  `VERIFIED_STATES`/`VERIFIED`, `dlp_media.ingest`의 변수 `ms`).

## 11. ADR 색인

| 번호 | 주제 |
|---|---|
| 0001 | 개발 도구 체인과 개발 서비스 구성 |
| 0002 | 계약 타입과 라벨 저장 방식 |
| 0003 | 세션 수집과 미디어 처리 |
| 0004 | 멀티스트림 동기화 (결정 3은 0028이 대체) |
| 0005 | 프라이버시 게이트 |
| 0006 | 검수 도구 연동 |
| 0007 | 데이터셋 버전과 계보 |
| 0008 | 자동 프리라벨 어댑터 |
| 0009 | 모델 레지스트리와 실제 모델 연동 |
| 0010 | 모델의 상업 사용 분류 |
| 0011 | 관계 도출과 표면 커버리지 |
| 0012 | 행동 구간 2단 구조 |
| 0013 | 지표와 평가 하네스, 배포 게이트 |
| 0014 | 검수 운영 로직 |
| 0015 | 1차 감사 정정 |
| 0016 | 재학습 루프와 모델 레지스트리 |
| 0017 | 액티브 러닝 |
| 0018 | 내보내기 |
| 0019 | 공간 라벨 시각 규약과 2차 검수 정정 |
| 0020 | 운영 지표, 원본 접근 감사, 원본 보관 만료 |
| 0021~0023 | 3차 검수 정정 (내보내기·운영, 계약·DB·동기화, 프라이버시·검수) |
| 0024~0028 | 4차 검수 정정 (프라이버시·검수, 평가·재학습, 프리라벨·행동·관계, 내보내기 ID·검증 시각, 동기화·온톨로지·생애주기) |
| 0029 | 검수 완료 판정과 생애주기 전이 |
| 0030~0031 | 5차 정정 (주석 작업 중 발견: 프라이버시·수집·검수 운영·운영 지표 / 저장소·데이터셋·동기화·프리라벨·관계·평가·가명) |
