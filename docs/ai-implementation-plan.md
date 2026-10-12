# AI 에이전트 구현 계획

2026-10-07 · 기준 문서: [labeling-platform-plan.md](labeling-platform-plan.md)

## 목적과 범위

이 문서는 플랫폼 계획(기준 문서)을 **AI 코딩 에이전트가 실제로 구현할 수 있는 단위**로 다시 설계한 것이다. 기준 문서가 "무엇을, 왜"를 정한다면, 이 문서는 "어떤 순서로, 어떤 계약 아래, 어떻게 검증하며" 만드는지를 정한다.

AI 에이전트가 맡는 일과 사람이 맡아야 하는 일을 처음부터 나눈다.

| AI 에이전트가 할 수 있는 일 | 사람이 해야 하는 일 |
| --- | --- |
| 스키마·온톨로지 패키지, 파이프라인 코드, 변환기, 지표 라이브러리, 검수 도구 연동, 큐·샘플링 로직, 내보내기, 대시보드, 테스트·CI, 인프라 구성 파일 | 촬영과 현장 프로토콜 실행, 장비 구매·캘리브레이션, 골든셋 수동 라벨링, 블러·작업 라벨 검수, 미결정 사항 결정, 성공 기준 수치 확정, GPU·서버 확보, 온톨로지 내용(작업·하위 단계·클래스) 확정 |

에이전트가 사람의 결정을 기다리며 멈추지 않도록, 모든 미결정 사항은 **설정값으로 분리하고 기본값을 둔다.** 결정이 나면 설정만 바꾼다.

## 에이전트 친화 설계 원칙

1. **계약 우선.** 모든 모듈 사이의 데이터는 `schema` 패키지의 타입(Pydantic 모델 + 생성된 JSON Schema)으로만 주고받는다. 모듈을 만들기 전에 입출력 타입을 먼저 확정한다. 에이전트가 여러 모듈을 병렬로 만들어도 계약이 맞으면 합쳐진다.
2. **온톨로지와 정책은 코드가 아니라 데이터.** 온톨로지, 블러 대상, 허용 오차, 큐 가중치, 샘플링 비율, 내보내기 정책은 버전 관리되는 YAML이다. 코드에는 값이 하드코딩되지 않는다.
3. **모델은 어댑터 뒤에 둔다.** 모든 자동 모델(탐지, 포즈, SLAM, VLM)은 공통 인터페이스 `Predictor.run(clip) -> list[LabelRecord]`를 구현한다. 각 어댑터에는 CPU에서 수 초 안에 도는 **가짜(stub) 구현**과 실제 구현을 둔다. CI와 다른 모듈 개발은 stub으로 진행한다.
4. **모든 단계는 멱등적인 CLI.** 각 파이프라인 단계는 `dlp privacy detect <세션ID>` 등의 명령으로 단독 실행되고, 같은 입력에 같은 출력을 낸다. 세션 ID 위치와 옵션은 각 명령의 `--help`를 따른다. 오케스트레이터(Prefect)를 추가할 때는 이 CLI를 묶는다. 에이전트가 단계 하나를 독립적으로 실행·디버깅할 수 있다.
5. **정답을 아는 합성 데이터로 검증한다.** 실제 영상이 없어도 테스트할 수 있도록 오프셋을 아는 합성 신호, 위치를 아는 합성 블러 대상, 경계를 아는 합성 행동 시퀀스를 생성하는 픽스처 생성기를 먼저 만든다. 모든 알고리즘 모듈의 완료 기준은 이 픽스처에 대한 자동 테스트다.
6. **작은 작업 카드.** 각 작업 패키지(WP)는 하나의 PR로 끝나는 크기의 작업 카드로 나눈다. 카드에는 입력 계약, 출력 계약, 완료 기준(자동 검증 가능), 금지 사항이 들어간다.
7. **사람 확인 지점을 명시한다.** 결과 품질을 코드로 판정할 수 없는 지점(UI 사용성, 실제 영상에서의 모델 품질)은 카드에 "사람 확인"으로 표시하고, 에이전트는 확인용 산출물(스크린샷, 샘플 렌더, 지표 리포트)을 만들어 둔다.

## 저장소 구조

> 아래는 계획 당시의 목표 구조다. `bridges/`, `metrics/`, `flows/`, `dashboards/`처럼 실제와 다른 이름이 있다.
> 현재 구조와 패키지 의존은 [구조 지도](handover/architecture.md)가 기준이다 (의존 표는
> `tests/test_import_boundaries.py`가 코드와 대조한다). Python 버전은 `pyproject.toml`의 `>=3.12,<3.14`가 기준이다.

```
data_labeling/
├── CLAUDE.md                  # 에이전트 작업 규칙, 명령어, 금지 사항
├── docs/                      # 기준 문서, 이 문서, ADR, 작업 카드
├── config/
│   ├── ontology/v1/           # 작업·행동·객체·도구·이벤트·프라이버시 사전 YAML
│   ├── policies/              # 블러 대상, 허용 오차, 큐 가중치, 샘플링, 내보내기 정책
│   └── defaults.yaml          # 미결정 사항의 기본값
├── packages/
│   ├── schema/                # 세션·스트림·라벨·에피소드 그래프 타입, 검증, 온톨로지 로더·마이그레이션
│   ├── fixtures/              # 합성 데이터 생성기
│   ├── media/                 # 수집, PTS 인덱스, 프록시, IMU·장갑 수집
│   ├── sync/                  # QR 슬레이트, 두드림 검출, 상호상관, 드리프트 보정
│   ├── privacy/               # 블러 탐지 어댑터, 추적·유지, 렌더, 감사 표본
│   ├── prelabel/              # 객체·도구, 손, 전신, 접촉, 자세·깊이, 행동 어댑터
│   ├── relations/             # 관계 도출 규칙, 커버리지 계산
│   ├── review/                # 우선순위 큐, 블라인드·오류 삽입, 일치도
│   ├── bridges/               # CVAT·Label Studio 변환기와 작업 생성
│   ├── datasets/              # 버전, 분할, 골든셋, 계보, 사용 중지 전파
│   ├── metrics/               # 평가 지표 라이브러리
│   ├── training/              # 재학습 작업, 골든셋 평가, 배포 게이트
│   ├── active/                # 세션 선택 점수
│   ├── export/                # COCO, 구간 JSON, LeRobot 에피소드
│   └── cli/                   # `dlp` 진입점
├── flows/                     # Prefect 플로우 (CLI 단계 조합)
├── services/                  # docker compose: PostgreSQL, 객체 저장소, CVAT, Label Studio, Prefect, MLflow
├── dashboards/                # 운영 지표 대시보드
└── tests/                     # 계약 테스트, 통합 테스트(합성 세션 종단 실행)
```

언어는 Python 3.11+, 패키지 관리는 uv, 린트·포맷은 ruff, 타입 검사는 pyright, 테스트는 pytest로 고정한다. 하나의 도구 체인으로 통일해야 에이전트가 모든 패키지에서 같은 명령으로 검증할 수 있다.

## 핵심 계약 (WP1에서 가장 먼저 확정)

| 타입 | 주요 필드 |
| --- | --- |
| `Session` | session_id, domain, worker_id(가명), site_id(가명), consent_version, streams[], calibration, sync_result, privacy_state, lifecycle_state |
| `Stream` | stream_id, kind(bodycam, imu, third_person, glove_l, glove_r, audio), uri, sample_rate, pts_index_uri, offset_ms, clock_scale, sync_method, sync_confidence |
| `LabelRecord` | label_id, session_id, kind(box, mask, keypoint, contact, action, state, pose3d, blur, event, relation), t_start_ms, t_end_ms, payload, ontology_version, provenance(source, model_version), evidence, confidence, verification(state, reviewer_id, reviewed_at), frame(좌표계), source_3d, parent_label_id(수정 이력) |
| `Action` payload | action_id, hand(left/right/both), verb, target_id, tool_id, t_approach_ms, t_contact_start_ms, t_contact_end_ms, t_end_ms, contact_held, pre_state, post_state, skill_id, task_id |
| `Relation` payload | subject_id, predicate(grasp, contact, support, contain), object_id, derived_by(rule_id) |
| `EpisodeGraph` | episode_id, session_id, entities[], relations[], events[], state_transitions[], coverage[] |
| `Ontology` | version, tasks, substeps, verbs(층위별), objects, tools(작용부·파지부), surfaces, states, events, privacy_targets, body_parts, postures |
| `DatasetVersion` | version_id, label_snapshot, ontology_version, splits(golden/train/val, 작업자·장소 단위), excluded_sessions, parent_version |

계약 라벨의 시각은 정수 ms다. 동기화 오프셋·드리프트 계산과 일부 시계열은 실수 값을 사용한다. 시간 구간 라벨은 마스터 타임라인, 공간 라벨의 키프레임·3D 표본은 스트림 PTS 기준이다 (ADR 0019). 계약 라벨에 프레임 번호를 시각으로 저장하지 않고 PTS 인덱스로 필요할 때 계산한다. 라벨은 덮어쓰지 않고 새 레코드를 만들며 `parent_label_id`로 수정 이력을 잇는다.

계약 테스트: 모든 타입의 왕복 직렬화, 온톨로지 밖의 동사·클래스 거부, 시간 순서 위반(t_approach > t_contact_start 등) 거부, 온톨로지 버전 불일치 거부.

## 작업 패키지

의존 관계 순서로 나열한다. 같은 단계 안의 WP는 계약만 맞으면 병렬로 진행할 수 있다.

### 단계 A · 토대 (기준 문서 Phase 0과 병행)

**WP0 저장소와 개발 환경**
- 산출물: 저장소 골격, `CLAUDE.md`, uv 워크스페이스, ruff·pyright·pytest 설정, GitHub Actions CI, `services/` docker compose.
- 완료 기준: `make check`(린트, 타입, 테스트)가 CI에서 통과. `make up`으로 PostgreSQL, 객체 저장소, Label Studio, Prefect, MLflow, lakeFS를 기동하고, CVAT는 `make cvat-up`으로 별도 기동한 뒤 헬스체크 통과.

**WP1 스키마와 온톨로지 패키지**
- 산출물: 위 핵심 계약의 Pydantic 모델, JSON Schema 생성, 온톨로지 YAML 로더·검증기, 온톨로지 버전 간 마이그레이션 프레임워크(매핑 표 기반), PostgreSQL 테이블과 마이그레이션(Alembic).
- 완료 기준: 계약 테스트 전체 통과. 기준 문서의 온톨로지 v1 초안이 YAML로 들어가 검증 통과.
- 사람 확인: 온톨로지 YAML 내용 검토(작업·하위 단계·클래스 목록은 파일럿 후 사람이 채움).

**WP2 합성 픽스처 생성기**
- 산출물: 다음을 생성하는 함수와 CLI.
  - 알려진 오프셋·드리프트를 가진 다중 스트림(오디오 두드림, QR 슬레이트 프레임, 장갑 압력 스파이크, IMU).
  - 알려진 위치의 블러 대상(얼굴 이미지 패치, 텍스트 문서, 거울 영역)이 움직이는 짧은 합성 영상.
  - 알려진 경계를 가진 손 키포인트 궤적과 접촉 신호, 정답 `Action` 레코드.
  - 작업자·장소·도메인 메타데이터를 가진 가짜 세션 수백 개.
- 완료 기준: 생성기 자체 테스트, 시드 고정 시 결정적 출력.

### 단계 B · 수집과 프라이버시 (Phase 1)

**WP3 수집과 미디어**
- 산출물: 세션 업로드·등록, ffprobe 기반 PTS 인덱스, 프록시 영상 생성, GPMF 등에서 IMU 추출(어댑터, 기종별 확장), 장갑 데이터 수집(Parquet/HDF5 어댑터), 원본 저장소와 라벨링 저장소 분리.
- 완료 기준: 가변 프레임레이트 합성 영상에서 PTS↔ms 변환 오차 0. 중복 업로드 멱등.

**WP4 동기화**
- 산출물: QR 슬레이트 검출, 오디오 피크·장갑 스파이크 두드림 검출, 오디오 상호상관, 장갑-운동 상관, 다지점 선형 드리프트 보정, 방법·신뢰도 기록, 사람 미세 조정값 반영 API.
- 완료 기준: 합성 픽스처에서 오프셋 오차가 슬레이트·두드림 ≤ 1프레임 시간, 상호상관 ≤ 2프레임 시간. 드리프트 추정 오차 ≤ 10ppm. 방법 우선순위 대체 동작 테스트.

**WP5 프라이버시 게이트**
- 산출물: 탐지 어댑터(얼굴, 반사면, 송장·문서·화면·사진물·주소 표지: 오픈 보캐뷸러리 + OCR + 바코드 결합), 추적·보간·200ms 유지, 블러 트랙을 `LabelRecord`로 저장, 모자이크·단색 렌더러, 오디오 제거, 검수 우선 구간 산출(트랙 끊김, 경계 신뢰도, 탐지기 불일치, 반사면 존재), 잔여 누락 감사 표본 추출, 전수 검수 종료 조건 판정기.
- 완료 기준: 합성 영상에서 정답 대상 재현율 100%(stub 탐지기가 정답을 내는 조건에서 추적·유지·렌더 로직 검증). 렌더본의 대상 영역 픽셀이 원본과 복원 불가능하게 다름을 테스트. 렌더본에 오디오 트랙 없음.
- 사람 확인: 실제 파일럿 영상의 블러 샘플 렌더와 탐지 재현율 리포트.

**WP6 검수 도구 연동**
- 산출물: `LabelRecord` ↔ CVAT 작업·주석, `LabelRecord` ↔ Label Studio 작업 변환기, 세션→작업 생성, 검수 결과 수집 웹훅, 역할 기반 권한(원본 권한자, 일반 라벨러, QA), 라벨러 ID 워터마크, 시간 구간·장갑·IMU 동기 재생 설정.
- 완료 기준: 두 도구 모두 왕복 변환 무손실(ID, 시간, 속성). 일반 라벨러 계정이 원본 URI에 접근 불가함을 통합 테스트로 확인.
- 사람 확인: 검수 화면 사용성(스크린샷 + 파일럿 검수자 피드백).

**WP7 데이터셋·계보**
- 산출물: 데이터셋 버전 커밋(lakeFS 또는 DVC 어댑터), 작업자·장소 단위 분할기(골든 배정 시 해당 작업자·장소를 학습에서 제외), 도메인별 골든셋 관리와 버전, 생애주기 상태 머신, 사용 중지 전파(이후 버전·내보내기 제외, 영향받은 학습 실행 목록 산출), 계보 조회 API.
- 완료 기준: 합성 세션 수백 개에서 골든·학습 간 작업자·장소 교집합 0. 사용 중지 후 새 버전과 내보내기에 해당 세션 0건. 계보 조회가 세션→버전→학습 실행→내보내기를 모두 반환.

### 단계 C · 프리라벨 (Phase 2)

**WP8 프리라벨 어댑터**
- 산출물: 공통 `Predictor` 인터페이스 아래 객체·도구(작용부 마스크 포함), 손 21관절, 전신, 접촉(장갑 임계값 분할, 영상 기반 추정, 도구-표면 겹침·깊이), 카메라 자세·깊이(시각-관성 SLAM, 시각 SLAM + 단안 깊이, 표면 평면 추정), 3인칭 착용자 매칭. 각각 stub과 실제 구현.
- 완료 기준: 모든 어댑터가 스키마 검증을 통과하는 `LabelRecord`를 출력. stub으로 종단 플로우 통과. 실제 구현은 GPU 환경에서 샘플 세션 처리 시간·메모리 리포트 생성.
- 사람 확인: 실제 영상에서의 품질은 골든셋 평가(WP11)로 판정.

**WP9 관계 도출과 커버리지**
- 산출물: 규칙 엔진(규칙은 YAML, 규칙 ID를 `derived_by`에 기록), 손-객체·도구-표면 접촉 구간에서 관계 생성, 작용부 3D 궤적을 표면 평면에 투영해 커버리지 계산.
- 완료 기준: 합성 시나리오(걸레질 반복 접촉 등)에서 정답 관계·커버리지와 일치. 규칙 변경 시 관계 재생성 멱등.

**WP10 행동 구간 2단 구조**
- 산출물: 손 속도·접촉 신호 기반 경계 후보 생성기, VLM 분류 어댑터(온톨로지 목록을 프롬프트에 주입, JSON Schema로 출력 강제, 스키마 위반·온톨로지 밖 응답 시 재시도 후 "미상"), 인접 동일 분류 병합, 사이 구간을 대기·미상·범위 외로 채우기, 한 문장 설명 초안.
- 완료 기준: 합성 행동 시퀀스에서 경계 후보 재현율 ≥ 95%(허용 오차 내). VLM stub으로 병합·채우기 규칙 테스트. 타임라인 공백 0.

### 단계 D · 검수 운영과 학습 루프 (Phase 2~3)

**WP11 지표와 평가 하네스**
- 산출물: 지표 라이브러리(mAP, HOTA, IDF1, IoU, PCK, 접촉 시점 오차·F1, 구간 F1, temporal mAP, 상태 전이 정확도, 커버리지 오차, macro F1, ECE, 카파, 경계 일치율), 골든셋 평가 실행기, 장갑·맨손 등 하위 집단별 분리 리포트, 클래스별 최소 표본 미달 표시, 배포 게이트(기존 모델 대비 우열 판정).
- 완료 기준: 각 지표를 손으로 계산한 작은 예제와 대조하는 단위 테스트. 가능한 지표는 기존 참조 구현(예: TrackEval, pycocotools)과 결과 일치 테스트.

**WP12 검수 운영 로직**
- 산출물: 우선순위 큐(낮은 신뢰도, 모델 불일치, 신규 객체, 장갑-영상 접촉 불일치), 높은 신뢰도 구간 표본 검수와 "표본 검증" 상태 부여, 블라인드 과제 배정, 오류 삽입 과제 생성(경계 이동, 클래스 교체, 블러 삭제)과 재학습 제외 표시, 이중 라벨링 배정과 일치도 계산, QA 표본 추출.
- 완료 기준: 정책 YAML의 비율대로 배정되는지 통계 테스트. 오류 삽입 레코드가 데이터셋 버전의 학습 분할에 0건.

**WP13 재학습과 모델 레지스트리**
- 산출물: 데이터셋 버전에서 모델별 학습 데이터 추출(자동 원본과 수정본 차이 포함), 학습 작업 템플릿, MLflow 기록, 레지스트리, 게이트 통과 시 프리라벨·프라이버시 어댑터에 배포.
- 완료 기준: stub 모델로 "누적 → 학습 → 평가 → 게이트 실패 시 미배포 / 통과 시 배포" 전체 루프가 통합 테스트에서 동작.

**WP14 액티브 러닝**
- 산출물: 클래스별 수정률 기반 세션 점수, 이후 추가할 점수 항목을 플러그인으로 붙이는 구조, FiftyOne 연동.
- 완료 기준: 합성 수정률 분포에서 기대 순서대로 세션이 선택됨.

**WP15 내보내기**
- 산출물: COCO, 구간 JSON, LeRobot 에피소드 변환기(손·작용부 궤적, 도구-표면 접촉, 검증 상태 포함), 내보내기 검증 정책 적용, 내보내기 이력 기록.
- 완료 기준: 각 형식의 공식 로더(pycocotools, LeRobot 데이터셋 로더)로 읽기 성공. 기본 정책에서 미검수 라벨 0건, 사용 중지 세션 0건.

**WP16 운영 대시보드와 보안**
- 산출물: 주간 운영 지표(검수 시간, 수정률, 자동 승인율, 프리라벨 편향, 오류 삽입 발견율, 잔여 블러 누락률, 에피소드당 원가), 원본 접근 감사 로그와 월간 리포트, 원본 보관 기간 만료 알림.
- 완료 기준: 합성 운영 이벤트로 지표 값이 기대와 일치. 원본 열람 이벤트가 빠짐없이 감사 로그에 기록됨을 테스트.

### WP17 AI 1차 검수(로컬 VLM)

사내 OpenAI 호환 서버(NVIDIA DGX Spark의 vLLM)의 VLM이 **승인된 블러본만** 보고, 검수 전 자동 라벨마다
맞음·의심·틀림·판단 불가와 한국어 이유, 온톨로지 안의 수정 제안을 남긴다. 판정은 라벨도 검수 상태도
아니다(추가 전용 표, 사람이 명시적으로 적용할 때만 사람 레코드). AI는 사람 검수를 **늘리는 방향**
(우선순위, 표본 제외, 늦은 지적의 재검수)으로만 쓴다. 결정과 불변 조건 I1~I10은
[ADR 0040](adr/0040-ai-first-pass-review.md)이다.

운영 순서: `dlp prelabel run` → `dlp relations run` → `dlp actions run` → `dlp aireview run` →
`dlp review plan` → `dlp review assign` → 사람 검수 → `dlp review collect` → `dlp review verify`.

| WP | 산출물 | 완료 기준 | 상태 |
| --- | --- | --- | --- |
| 17.0 결정·준비 | ADR 0040, `config/policies/aireview.yaml` 골격(`TODO(real-model)`), `config/models.yaml external` 후보(모델 카드로 라이선스 확인), `used_models`에 `aireview.backend.model`, `ops.yaml audit.service_purposes`에 `aireview.run`, Spark vLLM 실행 안내(`services/vlm/`, [VLM 서버 안내](handover/vlm-server.md))와 Arm64 재현 점검표, 이 절 | `make licenses` 통과(후보 모두 분류 기재), `make todo-models`에 aireview 표시 | 완료 (ADR 리뷰 대기) |
| 17.1 공통 전송 | `dlp_models.vlm`(요청 본문, 오류 분류, 백오프·재시도), 행동 단계 어댑터화, 4xx → `VlmRequestError`·종료 5 | 행동 기존 테스트 무수정 통과, **리팩토링 전 요청 본문과 바이트 동일**, 버전 문자열 불변, import 경계 표 갱신, A-08의 4xx 부분 반영 | 완료 |
| 17.2 판정 핵심(순수) | `dlp_media.frames`, `dlp_aireview`의 policy·plan·prompt·schema·validate·suggest·judge·overlay·sensors·perturb·oracle·backend(프로토콜), `dlp_schema.aireview` 계약(DB 없음), `ReviewReason` AI 사유, `make schemas` | 계획(대상 정확히 한 번, 긴 라벨 분리, 같은 프레임 분할, 선택 결정성, 단위 키 국소성, 공간 대상마다 표식 1프레임 이상, 장갑 센서 근거 라벨 제외)·프롬프트(틀 전부를 골든 파일로 바이트 고정: system·고정 블록·응답 스키마·가변 글·재시도 글, 재시도는 끝에만, `kinds`는 요청 본문을 바꾸지 않음)·스키마(portable 키워드)·재검사(어떤 응답 값도 예외 없음, 기록 문장에 모델 값 없음)·**합성 오류 100% 탐지** 테스트, 복잡도 15 이하 | 완료 |
| 17.3 벤치·점검·추정 + 장비 결정 | `FakeOpenAIServer`, `dlp aireview bench`·`probe`·`estimate`(DB 읽기만, `--include-actions`), 합성 벤치 입력 `dlp_fixtures.aireview`, Spark 실측 절차·기록 양식([VLM 서버](handover/vlm-server.md#spark-실측-절차)) | CI에서 가짜 서버로 보고서 생성. **Spark 1대에서** 후보 모델 2~3개의 프리필·디코드 실측, 파일럿 영상의 행동 후보 밀도·aireview 단위 수, VLM 밖 단계(OWLv2·깊이·렌더) 처리 시간 실측, 아래 공식으로 대수를 정한 기록 | 코드 완료 (가짜 서버 검사), Spark 실측 대기 |
| 17.4 계약·DB | `tables.py` + Alembic `0019_ai_review.py`(down 0018, 5개 표, 추가 전용·TRUNCATE 금지 트리거), `aireview_repository`, `locks.try_lock_aireview`, `record_review`·`insert_labels`의 `aireview:` 검수자 거부 | 마이그레이션 일치, SQLite 저장소 테스트, 트리거 services 테스트, `make contracts` 깨끗 | 완료 |
| 17.5 실행기·CLI | `runner`(생산자·작업자·기록자, 차단기, 재개, 렌더 재확인, `DbSink`·`MemorySink`, 세션 잠금), `backend.EndpointPool`(창 키 고정·엔드포인트별 동시성 상한·백오프·차단기·반개방 ping 뒤 시험 요청·모두 막히면 중단), `media.open_blurred`·`frames_for`(블러본 경계, `BlurredFrame`), `status`(판정률·`--pending` 선택), 장갑 문장(CLI가 프리라벨 함수로, 감사 원본 저장소 `aireview.run`), `dlp aireview run`·`status`·`show`·`waive`(`dlp_cli.aireview_run_cmds`), `--pending`(조용한 시간·판정 없는 대상 또는 끝나지 않은 실행·승인·보류·사용 중지 제외, 최근 실패 세션은 뒤로, `--limit`은 실제 실행만 셈), 멈춤 신호(`--max-minutes`·Ctrl-C·`--parallel-sessions`가 묶음 대기까지 멈춤), 실행 중 사용 중지·보류 확인, `bench --source session:<id>`, 정적 경계·격리 테스트(CLI 조립 층 포함) | 재개·멱등·차단기·종료 코드 1·2·3·5·6·7·8·130, `label_records` 불변, 원본 접근이 `aireview.run` 용도로만(블러본 게이트 뒤) | 완료 (가짜 서버·OracleJudge 검사, 잠금 경합·ON CONFLICT는 services) |
| 17.6 검수 화면 표시 | LS 실현 가능성 services 테스트(`test_ls_ai_feasibility.py`: 같은 TimeSeries에 라벨 컨트롤 둘, readonly AI 구간, 구간별 결정, 작업 요약 예측, 컨트롤 추가 갱신 성공·제거 거부), 수집의 from_name 경계(F-AIR-02), `ai_display`(노출 집단: 블라인드·오류 삽입·QA 제외, 이중은 짝 루트, 재검수는 루트 배정, 대조군 0.1 / LS 요약·AI 구간·메모·결정 단축키 r·h(자동 키 후보 밖, 라벨 구간에서만 칸 표시, AI 구간 결정은 원 라벨로 되돌림) / CVAT 1단계 이슈), 노출 기록, `review.yaml ai_review`(코드 기본 off, 저장소 advisory)·`review_ui.yaml ai`, `ReviewSetup.ai`(CLI 경유), 우선순위 사유·표본 제외·`<배정>:ai-flagged` 마무리 보호(증가 프로필), `required` 게이트(`dlp review plan` 종료 6, waived:timeout), 회수 재생성 안내의 `dlp aireview run` | 단위 + LS·CVAT services 테스트, 실화면 사람 확인(Playwright, LS 1.23.2·CVAT v2.78.0 두 모드), 판정 없는 작업의 데이터·예측 불변, 모드와 무관한 같은 설정 | 완료 |
| 17.7 수집·적용 | `ai_collect.begin`·`ls_decisions`·`apply_decisions`·`settle`·`record_outcome`(검수자 명시 결정만, 제안은 DB에서 다시 읽어 검사, 구간 그대로일 때만, 세션 검증에 걸리면 AI 적용만 되돌림), `collect._outcome_for` 추출(`collect_task` 복잡도 기준선 제거), `reconcile.same_as_sent` 공개, 수집 출력의 AI 적용·경고 | 적용 표의 모든 행(`test_ai_collect.py`), 세션 검증 실패 적용의 되돌림, 적용 실패에도 수집 성공, 실제 도구 종단 services 테스트(Label Studio 결정 포함 주석 → 수집 → 사람 수정 레코드·결과 기록, CVAT 결정 없음) | 완료 |
| 17.8 품질·운영 | `dlp_review.ops.ai_quality`(대조군 기준 지표·층화 ok 놓침률과 Clopper-Pearson 상한·제안 적용·자동화 편향·누출), `QualityReport.ai`·`dlp review quality --ai [--ai-thresholds]`, `WeeklyMetrics.ai`·`ops.yaml alerts.ai_*`·주간 Markdown AI 절, `dlp aireview eval`(perturb·history, `dlp_aireview.evaluate`)·게이트(`gate.block: false`라 실패해도 경고만) | 합성 자료로 지표 손 계산 일치, eval이 DB에 쓰지 않음(쓰기 문장 감시), 주간 Markdown 새 절 | 완료 |
| 17.9 실장비 파일럿 | 실서버 `real_vlm` 테스트, 실세션 10개 이상 파일럿(holdout 0.1), 표시 임계값·선별·누락 표시 결정, mode를 required로(`max_wait_h` 24) | probe 스키마 위반율 < 1%, 골든 게이트 통과, 실측 처리량·대수와 품질 지표를 ADR 0040 후속·인수인계 상태표에 기록 | 예정 |

의존: 17.1·17.2 병행 → 17.3·17.4 → 17.5 → 17.6 → 17.7 → 17.8. LS 실현 가능성 테스트는 17.2와 병행할 수
있고, 17.9의 bench는 장비가 오는 대로 17.3 뒤부터 한다. 범위 밖 후속 과제는
[알려진 문제 레지스트리](handover/issues.yaml)의 `F-AIR-*`다.

장비 대수 (실측으로 정한다, 아래 값은 모두 가정)

```
U_s  = estimate가 낸 단계 s의 영상 1시간당 요청 수   (s ∈ {aireview 묶음별, actions(--include-actions)})
T_s  = bench 무릎 처리량(req/s) × t(벤치) ÷ t(작업량),  t = 새 프롬프트 토큰 ÷ 프리필 용량 + 출력 토큰 ÷ 생성 용량
       (프리필·생성 용량은 bench가 무릎 동시성에서 따로 잰다. 혼합 실행의 관측 tok/s를 한계로 쓰지 않는다)
D    = Σ_s U_s / T_s                                 (영상 1시간당 장비 초)
대수  = ceil( H × (1+ρ) × D / (86,400 × u) )          H=하루 영상 시간(164), ρ=재실행 여유(0.2), u=가동률(0.7)
```

aireview 전량은 영상 1시간당 약 530요청(가정)으로 Spark 1대가 1 req/s를 내면 2대 안팎이다. 행동 VLM은
후보 구간마다 직렬로 물어 영상 1시간당 수천 요청이 될 수 있어 대수를 정할 수 있다 (F-AIR-01).

## 종단 검증

각 단계가 끝날 때마다 **합성 세션 종단 테스트**를 확장한다. 합성 세션 하나를 수집 → 동기화 → 블러 → 프리라벨 → (검수 시뮬레이터가 정답으로 수정) → 데이터셋 버전 → 재학습(stub) → 평가 → 내보내기까지 CI에서 실행하고, 내보낸 결과가 픽스처 정답과 일치하는지 확인한다. 이 테스트가 깨지면 어떤 WP도 병합하지 않는다.

실제 파일럿 영상이 들어오면 같은 플로우를 실제 어댑터로 실행하는 **스모크 런**을 GPU 환경에서 주 1회 돌리고, 결과 리포트를 사람이 확인한다.

## 에이전트 운영 방식

- **작업 카드:** `docs/cards/WPxx-yy.md`에 카드 하나당 PR 하나. 카드 형식은 목표, 입력 계약, 출력 계약, 완료 기준(실행할 테스트 명령), 금지 사항(예: 계약 타입 변경 금지, 하드코딩 금지), 사람 확인 항목.
- **병렬화:** WP1과 WP2가 끝나면 WP3~WP7은 병렬 진행 가능. WP8~WP10은 WP1·WP2와 stub 어댑터만 있으면 병렬 진행 가능. 계약 변경이 필요하면 별도 카드로 먼저 처리하고 다른 카드는 그 뒤에 리베이스한다.
- **계약 변경 규칙:** `packages/schema` 변경은 ADR(`docs/adr/`) 한 장과 마이그레이션, 계약 테스트 갱신을 함께 요구한다.
- **리뷰:** 에이전트 PR은 CI 통과 + 자동 코드 리뷰 + 사람 승인(계약 변경, 프라이버시, 권한 관련 PR은 반드시 사람 승인).
- **CLAUDE.md에 둘 규칙:** 계약 라벨 시각은 ms 정수(동기화 계산·일부 시계열의 실수 값과 구분), 라벨 덮어쓰기 금지, 원본 URI를 일반 라벨러 경로에 노출 금지, 정책 값 하드코딩 금지, 새 모델은 반드시 어댑터와 stub을 함께 추가.

## 기준 문서 로드맵과의 대응

| 기준 문서 단계 | 이 문서의 작업 패키지 | 같은 기간의 사람 작업 |
| --- | --- | --- |
| Phase 0 · 준비 | WP0, WP1, WP2 | 온톨로지 내용, 카메라·장갑 선정, 촬영 프로토콜, 파일럿 촬영, 성공 기준 잠정치 |
| Phase 1 · 기반 | WP3, WP4, WP5, WP6, WP7 | 블러 검수, 잔여 누락 감사, 골든셋 수동 라벨링, 라벨 원가 기준선 측정 |
| Phase 2 · 프리라벨 | WP8, WP9, WP10, WP11, WP12 | 프리라벨 검수, 블라인드·오류 삽입 과제 수행, 실제 모델 품질 확인 |
| Phase 3 · 루프 | WP13, WP14, WP15, WP16 | 배포 승인, 내보내기 샘플 확인, 블러 전수 검수 종료 판정 |
| Phase 3 이후 · AI 1차 검수 | WP17 | DGX Spark 1대 구매·실측, 대수 결정, 파일럿 판정 확인, required 전환 결정 |

에이전트 작업은 사람 작업보다 앞서 끝나는 것을 목표로 한다. 예를 들어 Phase 1의 골든셋 수동 라벨링이 시작되는 시점에는 WP6(검수 도구 연동)과 WP7(분할·골든셋 관리)이 이미 동작해야 한다. 따라서 에이전트 일정의 실제 병목은 코드가 아니라 **실제 데이터가 필요한 확인 지점**(WP5·WP8의 실제 영상 품질, WP6의 사용성)이며, 이 지점에 쓸 파일럿 영상을 Phase 0 안에 확보하는 것이 가장 중요하다.

## 에이전트가 시작하기 전에 사람이 줄 것

1. 기본값으로 진행해도 되는지에 대한 승인: 객체 저장소(SeaweedFS 또는 MinIO), 데이터 버전 도구(lakeFS 또는 DVC), 검수 도구 하이브리드 시작.
2. GPU 환경 접근 방법(실제 어댑터 스모크 런용).
3. 파일럿 영상 몇 개(가능하면 IMU 포함, 장갑·맨손 각각)와 그 사용 범위.
4. 온톨로지 v1 초안 검토자 지정.
