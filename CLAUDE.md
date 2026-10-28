# CLAUDE.md

1인칭 돌봄 영상 라벨링 플랫폼. 기준 문서는 `docs/labeling-platform-plan.md`(무엇을, 왜),
구현 계획은 `docs/ai-implementation-plan.md`(작업 패키지 WP0~WP17, 순서, 완료 기준)다.
작업 전에 해당 WP 절을 읽는다. 처음 맡는 개발자는 `docs/handover/README.md`(전체 지도, 기능 상태,
환경 재현, 변경·복구 안내, 남은 과제)부터 본다. 계획 문서의 목표와 현재 구현 상태를 구분한다.
알려진 문제의 단일 목록은 `docs/handover/issues.yaml`, 외주 검수·리팩토링의 인수 기준은 `docs/handover/acceptance.md`다.

## 작업 방식 우선순위 (2026-10-28 소유자 결정)

이 저장소는 pstack 플러그인(`pstack@cursor-plugins`, `.claude/settings.json`에서 켬)을 쓴다. 작업 방식(계획 세우기,
구현·검증 순서, 테스트 작성 방식, 커밋·PR 흐름, 검토 방식, 글쓰기 문체)은 pstack 스킬을 이 문서보다 우선한다.
예외: 아래 "규칙" 절은 데이터·개인정보·라이선스·라벨 이력을 지키는 약속이라 pstack 스킬과 부딪쳐도 이 문서를 따른다
(원본 버킷·블러·사람 승인, 라벨 덮어쓰기 금지, 상업 사용 분류, 분할·내보내기 규칙, 주석·문서 한국어 포함).
`make check` 통과도 그대로 필수다.

## 명령어

- `make install` — 의존성 설치 (uv, 잠금 파일 기준)
- 전체 타입 검사·큐레이션 검사 환경은 `make install-curation`으로 설치한다. `make install` 재실행 시 선택 그룹이 제거된다.
- `make check` — 린트(ruff) + 포맷 검사 + 타입 검사(pyright strict) + 단위 테스트. PR 전에 반드시 통과
- `make fmt` — 자동 포맷·린트 수정
- `make contracts` — 온톨로지 검증 + JSON Schema가 코드와 일치하는지 검사 (`make check`에 포함)
- `make schemas` — 계약 타입을 바꾼 뒤 `schemas/*.schema.json` 재생성
- `make models` — `config/models.yaml`의 모델 가중치를 `data/models/`에 받고 sha256 확인 (저장소에는 넣지 않음)
- `make export-models` — 공개 ONNX가 없는 모델(메트릭 깊이)을 공식 가중치에서 변환 (일회용 PyTorch 환경)
- `make todo-models` — 실제 모델을 아직 연동하지 못한 곳(`TODO(real-model):` 표시) 목록
- `make licenses` — 정책이 쓰는 모델의 라이선스·학습 데이터·상업 사용 분류 표 (위반이 있으면 실패, `make check`에 포함)
- `make fixtures` — 합성 픽스처를 `data/fixtures/`에 생성 (저장소에는 넣지 않음)
- `make db-migrate` — Alembic 마이그레이션만 적용 (운영 배포 단계, 온톨로지 등록 없음)
- `make db-upgrade` — 개발 DB에 Alembic 마이그레이션 적용 + 온톨로지 v1 등록 (온톨로지 이름을 덧붙인 뒤에도 다시 돌린다.
  `en` 없는 예전 등록은 읽을 때 `LegacyOntologyContentError`, ADR 0043). 등록된 초안의 물리 사전값이 바뀌면 승인 없이
  종료 코드 3으로 아무것도 쓰지 않는다. 승인은 `uv run dlp ontology register 1.0.0 --accept-physical-priors` (ADR 0053).
  사전값 없는 객체를 덧붙이는 것은 사전값 변경이 아니다 (승인 불필요, ADR 0059)
- `make up` / `make down` / `make clean` — 개발 서비스 기동 / 중지 / 볼륨까지 삭제
- `make health` — `dlp services check`로 서비스 헬스체크
- `make test-services` — 실행 중인 서비스 대상 통합 테스트 (`@pytest.mark.services`, `isolated_env` 표시 테스트는 뺀다)
- `make test-isolated` — 격리된 일회용 환경(LeRobot·PyTorch CPU판)을 받는 테스트 (`@pytest.mark.isolated_env`).
  내보내기 종단 테스트는 DB도 쓰므로 `make up` 뒤에 돌린다. 환경은 `scripts/lerobot-env/uv.lock`에 고정한다
  (바꾸면 그 디렉터리에서 `uv lock`). YOLOX CPU 스모크도 포함한다 (`scripts/yolox-train-env/uv.lock`, ADR 0062)
- `uv run dlp console serve` — 브라우저 운영 콘솔 (기본 127.0.0.1:8090, 토큰 주소 출력, 작업 기록은 `DLP_CONSOLE_DIR`,
  기본 ~/.dlp-console). 새 컴퓨터 설치는 `scripts/install/install.sh` (Mac·Linux·WSL2, ADR 0061)
- `make install-curation` — FiftyOne(데이터 큐레이션, 약 1 GB) 포함 설치. `make install`을 다시 하면 빠진다
- `make coverage` — 기본 테스트의 줄 커버리지 (reports/coverage, 하한 없음)
- `make review-scope` — 전수 검수 범위표 (추적 파일별 위험 등급·관련 미해결 문제 수, reports/review-scope.csv)
- `make cvat-up` / `make cvat-down` / `make cvat-superuser` — CVAT (공식 compose, 고정 버전, UI는 검수 화면 테마를 얹은 이미지)
- `make ui-theme` — 검수 화면 테마(`config/review_ui.yaml`)를 바꾼 뒤 CVAT 테마 색 파일 재생성 (`make contracts`가 일치 검사, ADR 0038)

## 구조

- `packages/<이름>/` — uv 워크스페이스 멤버. `src/<모듈>/`과 `tests/`를 둔다.
  새 패키지는 루트 `pyproject.toml`의 `[tool.uv.sources]`와 `dependencies`에 등록한다.
- `packages/cli/` (`dlp_cli`) — `dlp` 명령줄 진입점. 하위 명령마다 `<단계>_cmds.py`가 정책·저장소·DB를 엮어 각 패키지의
  실행 함수를 부른다. 모델 선택·온톨로지 선택 로직도 일부 남아 있다. 신규 도메인 로직은 각 패키지에 둔다.
- `packages/schema/` (`dlp_schema`) — 계약 타입, 온톨로지 로더·검증, 이관, 설정 로더, DB 테이블·마이그레이션·저장소.
  테스트용 객체는 `dlp_schema.testing`을 쓴다.
- `packages/fixtures/` (`dlp_fixtures`) — 정답을 아는 합성 데이터: 가짜 세션, 오프셋·드리프트를 아는 동기화 신호와
  QR 슬레이트 MP4, 위치를 아는 블러 대상 VFR 영상, 경계를 아는 행동 시퀀스(손 키포인트·장갑 압력·정답 라벨).
  알고리즘 모듈의 테스트는 이 생성기로 작성하고, 생성기 출력의 정답을 기준으로 판정한다.
- `packages/media/` (`dlp_media`) — 세션 수집: 원본 저장소(불변, 멱등), PTS 인덱스, 프록시, IMU 추출기(GPMF·사이드카),
  장갑 Parquet·HDF5 정규화. `dlp ingest <매니페스트>`로 실행한다.
  카메라 내부 파라미터 등록(`dlp_media.calibration`, ADR 0045): 세션 값 → 기기(`calibration.camera_id`) 등록값
  (`camera_calibrations`, 추가만) → 화각 근사 순으로 고른다. 작업자 손 크기 사전값은 `worker_hand_sizes`(추가만).
  수집 뒤 세션 → 기기 연결은 `session_camera_assignments`(추가만, `dlp calib assign`, 매니페스트 camera_id가 앞선다).
  작업자 손 크기는 유효한 등록 중 최신 → 정책 순이다. 무효 기록 `worker_hand_size_invalidations`(`dlp calib palm-revoke`,
  동의 철회·삭제 요구)와, 추정 출처 세션(`worker_hand_size_sources`)이 사용 중지된 추정은 결정에서 빠진다 (ADR 0052).
  `palm-estimate`는 잰 값을 `--override-measured` 없이 덮지 않는다고, 화면비가 다른 캘리브레이션 세션은 이유를 알리고 건너뛴다 (F-PAR-77). 캘리브레이션은 보드 기울기·타당 범위 검사
  (`calibration.yaml`)를 통과해야 등록한다. 화면비가 다른 캘리브레이션은 쓰지 않고 화각 근사로 떨어진다.
  `dlp calib intrinsics|import|assign|show|palm|palm-estimate|palm-revoke`. 정책은 `config/policies/calibration.yaml`.
  촬영 점검 경고 `dlp_media.capture_check`(prepare·ingest, 기준 `media.yaml capture`, 가이드 `docs/handover/capture-guide.md`),
  TTA·학습 증강 공용 기하 변환 `dlp_media.geometry`, COCO RLE `dlp_media.rle`. 방향 태그가 틀린 영상은
  `dlp media prepare --display-rotation <반시계 각도>`.
- `packages/sync/` (`dlp_sync`) — 멀티스트림 동기화: QR 슬레이트, 두드림, 오디오·운동 상호상관, 드리프트 보정.
  `dlp sync run <세션>`, `dlp sync adjust <세션> <스트림> <ms>`. 정책은 `config/policies/sync.yaml`.
- `packages/privacy/` (`dlp_privacy`) — 프라이버시 게이트: 탐지기(YuNet, QR·바코드, OWLv2 오픈 보캐뷸러리, 반사면, 오라클 stub), 추적·보간·유지,
  모자이크 렌더, 검수 우선 구간, 잔여 누락 감사, 블러 변경 뒤 배포 매체 회수 요청(`dlp_privacy.revocation`, ADR 0039),
  사고 대응 승인 보류(`dlp_privacy.hold`, ADR 0042: DB 단계 커밋 뒤에 저장소 클라이언트를 만들고, 렌더 무효화는
  별도 트랜잭션의 세션 잠금 아래 그 보류가 아직 유효할 때만 쓴다).
  `dlp privacy detect|approve|render|audit-sample|hold`. 정책은 `config/policies/privacy.yaml`.
  블러 오탐 의심 힌트 `dlp_privacy.hints` (ADR 0055).
- `packages/review/` (`dlp_review`) — 검수 도구 연동: CVAT·Label Studio 변환기(무손실 왕복), 작업 생성, reconcile(검수 결과 →
  라벨 이력), 웹훅 수집, 라벨러 워터마크, 역할 경계, 배포 매체 회수 실행기(`dlp_review.revoke`).
  `dlp review create|collect|verify|serve|register-webhooks|revoke|refresh-urls` (verify: 검수가 끝난 세션을 human_verified로. 설명·3D 궤적 따르기를 먼저 적용하고, `review.yaml verify.automatic`(기본 `camera_pose`·`relation`·`coverage`, 모델 출처만, ADR 0011 개정)의 자동 라벨은 검수 전이어도 완료를 막지 않는다(검수 상태는 그대로), ADR 0050, 출처 2D가 바뀌어 지운 3D를 지금 2D로 다시 만들기 전에는 완료를 막고 `dlp prelabel run`을 안내한다(`verify.require_current_3d`, ADR 0051),
  revoke: 블러가 바뀐 뒤 이미 배포한 라벨링 매체 회수, 대기가 남으면 종료 코드 4, ADR 0039. `revoke --orphans`: DB 기록 없이
  남은 도구 작업·사본 정리(기본 보고만, `--apply`일 때 삭제). refresh-urls: 열린 Label Studio 작업의 서명 URL 다시 서명).
  Label Studio 결과는 담당자 계정(`review.yaml label_studio.users`)이 낸 주석만 받는다.
  운영 로직(`dlp_review.ops`): 우선순위 큐, 표본 검증, 블라인드·오류 삽입·이중·QA 배정, 품질 측정.
  `dlp review plan|assign|queue|qa|quality`. 정책은 `config/policies/review.yaml`.
  검수 화면 테마(`dlp_review.ui_theme`, `config/review_ui.yaml`, `services/review-theme/`): 색·로고·단축키·다크/화이트만
  도구 밖에서 덧씌우고 라벨 형식은 바꾸지 않는다 (ADR 0038).
  AI 1차 검수 표시·수집(ADR 0040): `dlp_review.ai_display`(판정 → 검수 화면 표시·우선순위·표본 제외, 노출 기록·대조군),
  `dlp_review.ai_collect`(검수자가 고른 제안 적용·세션 검증·결과 기록 `ai_review_outcomes`),
  `dlp_review.ops.ai_quality`(대조군만으로, 표본 가중 지적 정밀도·재현율·ok 놓침률 95% 상한·ECE·κ). `dlp review quality --ai`.
  판정 표시 보정 `dlp aireview calibrate`(대조군만, `dlp_review.ops.ai_calibration`, ADR 0062): 버전 붙은 산출물을
  `review.yaml ai_review.calibration.artifact`(경로·sha256)로 고정해야 쓰인다. 의심·틀림 표시만 거르고 ok와 검수 대기열은
  바꾸지 않는다.
  설정은 `review.yaml ai_review`(코드 기본 off).
  오탐 의심 힌트 `dlp review hints <세션> [--tta] [--ai]`(블러: 원본 버킷 hints 파일 → 블러 검수 CVAT 이슈, 객체
  `dlp_review.object_hints`: 라벨링 CVAT 이슈, 잘라 낸 박스 AI 확인 `dlp_aireview.crops`). 라벨·블러는 바꾸지 않는다
  (ADR 0055·0056).
  손이 있는 골격(hand21·arm3)의 entity_id는 수집 때 손에서 정하고(`<손>_hand`·`<손>_arm`), 모든 점이 visibility 0인
  키포인트 키프레임은 CVAT outside로 오간다 (F-REV-32·33).
  마스크(`mask_track`)는 CVAT 모양 모드 마스크로 무손실 왕복한다 (`dlp_review.cvat.to_cvat_shapes`, 원본 해상도 작업만,
  블러 검수에서는 받지 않음, ADR 0059).
- `packages/datasets/` (`dlp_datasets`) — 데이터셋 버전(lakeFS 커밋), 작업자·장소 단위 분할(holdout 포함), 골든셋 제안,
  사용 중지 전파, 계보 조회. `dlp dataset golden|build|withdraw`, `dlp lineage <세션>`. 정책은 `config/policies/dataset.yaml`.
  빌드는 지정한 골든셋뿐 아니라 등록된 모든 골든셋 세션을 학습·검증에서 뺀다(지정하지 않은 골든셋은 holdout, F-ETA-19).
- `packages/models/` (`dlp_models`) — 모델 레지스트리(`config/models.yaml`, 해시 확인)와 공용 ONNX 런타임(OWLv2, 메트릭 깊이).
- `packages/prelabel/` (`dlp_prelabel`) — 자동 프리라벨: MediaPipe 손·COCO 객체, RTMPose 전신, OWLv2 도구, 깊이 기반 3D 궤적,
  Oracle stub, 장갑·영상 접촉, 3인칭 착용자 매칭. `dlp prelabel run <세션>`. 정책은 `config/policies/prelabel.yaml`.
  손 키프레임에는 MediaPipe 손 중심 3D 점(`KeypointFrame.world`, 선택, 검수 도구로 보내지 않는다. 사람이 고친 트랙에는
  2D가 그대로인 키프레임에만 parent의 world가 남는다, ADR 0044 1절 개정)을 싣고, 손목·다섯 손가락 끝 3D(카메라 좌표)는
  그 손 모양 + 2D 투영 + 손바닥 깊이·손 크기 사전값으로 맞춘다 (`dlp_prelabel.handfit`, ADR 0044. 손 모델 자세는 손 중심
  회전으로 다듬고 문턱은 해상도와 무관한 상대 재투영 오차다. world가 없거나 맞춤이 나쁘면 손바닥 관절만 픽셀 깊이로
  올리고 손가락 끝은 내지 않는다, `depth.fingertip_fallback`). 표본마다 방식·신뢰도, 궤적마다 내부
  파라미터 출처를 남기고 작업자 손 크기(`dlp_prelabel.palm`)를 우선한다 (ADR 0045).
  바디캠 6자유도 자세(IMU 회전 + 깊이 기반 이동, `dlp_prelabel.campose`)와 월드 좌표 3D 궤적(`dlp_prelabel.worldframe`,
  ADR 0046). 3D 궤적(손 관절·박스 중심, 카메라·월드 좌표)은 `derived_from`의 2D 트랙 검수를 물려받고
  (`dlp_schema.trajectory_follow`, 기록 `trajectory_inheritance`, ADR 0048·0050), 물려받기만 한 3D의 출처가 바뀌거나
  같은 2D에서 입력이 바뀌어 새로 계산한 3D가 생기면 궤적 따르기가 지운다(새 3D가 같은 검수를 물려받는다, ADR 0051).
  단계는 마지막으로 낸 버전으로 다시 돌릴지 정한다(입력이 되돌아오면 `+r<n>`으로 다시 만듦). 3D·월드 단계 뒤와 수집·`dlp review verify`에서 적용한다(세션 잠금, 멱등). 카메라 자세
  (entity `camera`)는 사람 검수 대상이 아닌 자동 라벨이다.
  시간 일관성 `dlp_prelabel.temporal`(박스 트랙 거르기·손 One Euro 평활·짧은 끊김 보간, ADR 0054), 손 3D 물리 제약
  `dlp_prelabel.constraints3d`(표본 `flags`), 장면 분류 `dlp_prelabel.scene`과 `dlp prelabel run <세션> --scene <장면>`
  (도구 질의를 kitchen·bathroom·bedroom·living·unknown으로 고름, ADR 0055), 손 크기 초점 거리 추정 `dlp_prelabel.fov`
  (출처 `estimated_hand_size`, 기본 끔, ADR 0056).
  착용자 팔 골격 arm3(손목·팔꿈치·어깨, 손별 entity `<손>_arm`, 손목은 hand21 손목 기준, `dlp_prelabel.arm`, 전신 단계가
  착용자 자세·합성 착용자 박스 자세에서 고름)와 팔꿈치 3D(`<손>_arm`, 아래팔 길이 표시 `forearm_length`), 바디캠 착용자
  몸 억제(`dlp_prelabel.wearer_suppress`, 판정 규칙 `dlp_schema.wearer_body`, 객체·도구·전신 단계, 정책
  `prelabel.yaml wearer_suppression`·`arm`, ADR 0058).
  객체·손 모양 마스크 `dlp_prelabel.masks`(SAM 2.1 tiny ONNX `dlp_models.sam2`, review 등급, 이미지 모드
  `masks.frame_stride_ms` 간격, 프롬프트 = 현재 박스 트랙마다 하나(박스 번호 재사용 무관, 검수 마스크 보호는 `derived_from`·조상, F-PAR-74)·hand21 손 트랙, 착용자 몸 박스·화면 아래 사람 박스 제외, 손은
  class `hand`·entity `<손>_hand`, `derived_from`은 출처 기록이고 검수를 물려받지 않음, 키프레임 `score`, 보간 없음,
  `dlp prelabel run --skip masks`, CPU stub `BoxEllipseSegmenter`, ADR 0059).
- `packages/retarget/` (`dlp_retarget`) — 고객 로봇 리타기팅(ADR 0047): 로봇 등록(`config/robots/<ID>.yaml`, 모델 sha256,
  고객 모델 파일은 저장소 밖 `DLP_ROBOTS_DIR`), URDF 기구학, 손 → 끝단 목표·평행 그리퍼, 관절·속도 한계 IK(내장, mink는
  격리 환경 `scripts/retarget-env`), 닿지 못한 프레임은 자르지 않고 표시. `orientation_mode`는 관절 수가 아니라 구조로
  고른다(접근 축 둘레 롤 관절이 있으면 full, SO-101은 full). 재시작 프레임·그리퍼에도 속도 한계, 손 목표가 없는 에피소드는
  drop, 설정·URDF 검증(축 단위·직교, 그리퍼, mimic, robot_id 경로) (ADR 0053). `dlp robot check <ID>`,
  `dlp export lerobot --robot <ID>`. 정책은 `config/policies/retarget.yaml`.
- `packages/relations/` (`dlp_relations`) — 관계 도출(YAML 규칙 엔진, `derived_by`=규칙 ID), 도구-표면 접촉(작용부-평면 거리),
  표면 커버리지. `dlp relations run <세션>` (멱등, 규칙 변경 시 차이만 반영). 정책은 `config/policies/relations.yaml`.
- `packages/actions/` (`dlp_actions`) — 행동 구간 2단 구조: 손목 속도·접촉으로 경계 후보 → VLM이 온톨로지 안에서 분류·설명
  (JSON Schema 강제, 재시도 후 미상) → 병합·채우기(공백 0). `dlp actions run <세션> --vlm-url`. 사람이 의미를 고친 행동의 설명은
  `dlp actions describe <세션>|--pending`이 VLM에 다시 물어 만든다(검수를 물려받음, ADR 0043). CPU·CI는 `OracleVlm` stub.
  정책은 `config/policies/actions.yaml`.
  설명 영어 번역(ADR 0043, `dlp_actions.translate`): 현재 한국어 설명마다 영어 형제 레코드(`language=en`,
  `source_label_id`=원문 ID, parent 아님)를 사내 서버(`dlp_models.vlm`, 기본 Qwen3-VL-8B)로 만든다. 온톨로지 ko→en
  이름을 용어집으로 넣는다. 멱등, 원문이 바뀌면 검수 전 번역만 교체, 서버 장애는 대기로 남긴다(예외 없음).
  설명(ko·en)은 검수 도구 경로가 없어 `segment_id`가 가리키는 행동·상위 구간의 검수를 따른다
  (`dlp_schema.description_follow`, ADR 0043 3절): 승인·표본 검증은 물려받고, 의미를 고쳤거나 지운 행동의
  검수 전 설명은 지운다. 물려받은 뒤 행동이 다시 바뀌면 물려받은 설명도 지운다(규칙 6, 물려받음은
  `description_inheritance` 기록으로 판정, Alembic 0021). 수집·`dlp review verify`·번역·재생성 단계가 적용한다.
  재생성 요청은 손마다 고친 의미 필드 전부(동사·대상·부위·도구·방향·전후 상태)를 묶고 손 단위로 검증한다.
  재생성·번역은 쓰기 직전 세션 잠금 아래 사용 중지·승인 보류·(프레임을 보냈으면) 렌더 해시를 다시 보고, 막히면
  아무것도 쓰지 않는다 (`dlp_actions.write_check`).
  `dlp translate run <세션>|--pending`, `dlp translate pending`. `dlp actions run`이 커밋 뒤 자동으로 부른다
  (실패해도 행동 결과·종료 코드 불변). CPU·CI는 `GlossaryStubTranslator`. 정책은 `config/policies/translate.yaml`.
- `packages/aireview/` (`dlp_aireview`) — AI 1차 검수 판정(ADR 0040): 사내 vLLM(OpenAI 호환, 기본 Qwen3-VL-8B)이 블러본
  프레임과 라벨 맥락으로 라벨마다 ok·suspect·wrong·unsure와 이유·제안을 낸다. 참고용 판정만 추가 전용 `ai_review_*` 표에
  쓴다(멱등·재개, 세션 잠금). CPU·CI는 `OracleJudge`·`FakeOpenAIServer`. 골든셋 평가(`evaluate`: 교란·이력, DB에 쓰지 않음,
  게이트 실패는 기본 경고만). `dlp aireview run|status|show|waive|estimate|bench|probe|eval`.
  정책은 `config/policies/aireview.yaml`.
- `packages/evaluation/` (`dlp_eval`) — 지표 라이브러리(mAP·HOTA·IDF1·PCK·접촉·구간 F1·temporal mAP·ECE 등, 참조 구현과 일치 테스트),
  골든셋 평가 하네스, 하위 집단 리포트, 배포 게이트, 정답 없는 대리 지표 `dlp_eval.proxy`(`dlp ops quality-proxy <세션>`,
  기준 `evaluation.yaml proxy`). `dlp eval golden <골든셋> --model <과제>=<버전>`. 정책은 `config/policies/evaluation.yaml`.
- `packages/training/` (`dlp_train`) — 재학습 루프: 데이터셋 버전에서 과제별 학습 예제 추출(자동 원본과 수정본 차이),
  학습 작업 템플릿, MLflow 기록, 모델 레지스트리(DB `model_versions`), 골든셋 평가·게이트 후 배포.
  `dlp train run|models|approve`. CI·CPU는 `oracle-stub` 학습기. 정책은 `config/policies/training.yaml`.
  학습 증강 `dlp_train.augment` (ADR 0057).
  replaces 과제의 첫 배포는 골든셋에 예측을 낸 기본 어댑터마다 비교 버전이 있어야 한다(어댑터는 라벨 ID 접두사로 판별,
  F-ETA-20). 비교 예측 확인과 학습 예제 추출은 같은 과제 대상 판정 `dlp_eval.runner.is_task_target`을 쓴다 (F-ETA-21).
  학습 모델 로더는 출발 가중치 계보의 지금 분류를 다시 본다 (F-ETA-22).
  실제 학습기 `yolox`(objects, `dlp_train.yolox`·`yolox_data`, ADR 0062): 지금 승인된 블러본 키프레임 프레임 → COCO(증강
  포함) → 격리 환경 `scripts/yolox-train-env`(uv.lock, `--extra cpu|cu128`)에서 학습 → `yolox-model.zip`(ONNX
  decode_in_inference false + manifest). 로더는 `YoloxTools`에 끼운다(`trained-objects`). GPU가 없으면
  `--param allow_cpu=true` 없이는 거절한다. 출발 가중치는 `training.yaml yolox.base_model`(models.yaml .pth)이고 그 분류를
  물려받는다. 기본 학습기는 아직 `oracle-stub`이다.
- `packages/console/` (`dlp_console`) — 운영 콘솔 (ADR 0061): 기존 `dlp` 명령을 작업으로 돌리고(셸 없음, 작업 디렉터리
  JSON Lines 추가만) DB는 읽기만 한다. 응답·로그에 원본 위치를 싣지 않고, 미리보기는 승인·렌더된 블러본의 라벨러 서명
  URL만 쓴다. Git이 관리하는 설정 파일은 고치지 않고 패치 미리보기만 낸다(외부 학습 가중치 가져오기: 구조 이름은 식별자 모양만, 가중치 경로는 models_dir 안인지
  confirm에서 다시 본다). 자식 `dlp`는 새 프로세스 그룹으로 돌고(pid·pgid 기록) 콘솔 종료 때 멈추며, 재시작 때 살아 있는
  그룹은 잠금을 유지한다 (`console.yaml jobs`). 설정
  `config/console.yaml`. 도메인 로직을 두지 않는다 (단계 조건·권한 검사는 `dlp` 명령이 하고, 콘솔 활성 규칙은 안내용).
- `packages/active/` (`dlp_active`) — 액티브 러닝: 클래스별 수정률 기반 세션 점수(점수 항목 플러그인 `register_term`),
  FiftyOne 연동(블러본만). `dlp active rank|fiftyone`. 정책은 `config/policies/active.yaml`.
  FiftyOne은 선택 설치(`make install-curation`).
- `packages/export/` (`dlp_export`) — 내보내기: COCO, 구간 JSON(`dlp_schema.export`), LeRobot v3.0 에피소드.
  데이터셋 버전 스냅샷에서, 검증 정책 적용, 블러본만. 작업자·장소 ID는 내보내기마다 다른 가명(HMAC). 내보내기 이력은
  올리기 전에 따로 커밋한다. 세션·라벨 ID도 내보내기마다 다른 가명이고(이력에는 내부 ID, 세션 대응표는 내부 경로),
  manifest에 파일 목록과 sha256을 넣는다 (ADR 0027). `dlp export coco|intervals|lerobot`. 정책은 `config/policies/export.yaml`.
  온톨로지 한·영 이름을 싣는다(COCO 범주 `name_ko`·`name_en`, 구간 JSON `term_names` 형식 1.2, LeRobot 어휘 `term_names`).
  설명은 언어마다 한 레코드(ko·en)이고 번역의 `source_label_id`도 가명 라벨 ID다 (ADR 0043).
  LeRobot은 격리된 일회용 환경(`scripts/lerobot_write.py`, `scripts/lerobot_check.py`)에서 공식 API로 쓰고 읽는다.
  객체 물리 사전값(재질·촉감·질량·마찰, ADR 0049)은 세 형식 모두 `measured: false`로 싣는다 (측정값 아님).
  manifest `verification_policy.trajectory3d`는 3D 검수 근거(2D에서 물려받음, `camera_pose: automatic_unreviewed`,
  `pose_derived`, 근거별 수)를 밝히고, 끊긴 물려받은 3D는 옵션과 관계없이 넣지 않는다 (ADR 0048·0050).
  3D는 `lerobot.interp_max_gap_3d_ms`로 보간하고 궤적 구간 채움 비율이 낮으면 경고한다. LeRobot에 3D 방식·내부 파라미터
  출처·접촉 객체 클래스 특징을 싣는다. 페이로드의 라벨 ID 필드(`pseudonym.LABEL_REF_FIELDS`, 예: `derived_from`)도 라벨
  가명이 된다. 손 부위 설정은 내보내기 시작 때 hand_joints와 대조한다 (ADR 0053).
  COCO `wearer_arm` 범주, LeRobot `annotation.wearer_arm`(arm3 2D + 팔꿈치 3D, ADR 0058).
  COCO에 마스크 `segmentation`(압축 RLE)·`mask_score`·`hand`, 객체 hand 범주가 손 키포인트 범주를 겸한다 (ADR 0059).
- `packages/ops/` (`dlp_ops`) — 운영 대시보드와 보안: 주간 운영 지표(검수 시간·수정률·자동 승인율·편향·발견율·
  잔여 블러 누락·원가, AI 1차 검수 절: 백로그·fallback·캐시 적중·장비 시간·ok 놓침 상한)와 경고, 원본 접근 월간 감사
  리포트, 원본 보관 만료 알림·결정.
  `dlp ops weekly|audit-report|retention|retention-decide|log-work|privacy-audit`. 정책은 `config/policies/ops.yaml`.
- `config/` — 온톨로지·정책·기본값 YAML. 코드에 값을 하드코딩하지 않는다.
  온톨로지 항목은 한국어 `ko`와 영어 `en` 표시 이름이 모두 필수다 (ID는 영어 snake_case, 라벨에는 ID만, ADR 0043).
  객체 물리 사전값(`physical_priors`: 재질·촉감·질량·마찰 범위, 출처는 사전값만)은 `config/ontology/v1/physics.yaml` 어휘를 쓴다 (ADR 0049).
  by_state는 상태 속성 하나만, 젖음 기본 범위 = 건조, prior_literature는 note 필수, 사전값 표에 `content_sha256` (ADR 0053).
- `schemas/` — 생성된 JSON Schema. 직접 편집하지 않는다 (`make schemas`).
- `services/` — 개발용 docker compose (PostgreSQL, SeaweedFS S3, Label Studio, Prefect, MLflow, lakeFS), CVAT 실행 스크립트.
- `tests/` — 패키지를 가로지르는 통합 테스트.
- `docs/adr/` — 아키텍처 결정 기록.

## 규칙

- 계약 라벨의 시각은 정수 ms로 저장하고 프레임 번호로 대체하지 않는다. 동기화 오프셋·드리프트 계산과
  일부 시계열은 실수 값을 사용한다. 시간 구간 라벨은 마스터 타임라인 시각, 공간 라벨
  (박스·마스크·키포인트·블러·3D 궤적)의 키프레임은 그 스트림 영상의 PTS 시각이다 (기준 스트림은 같다, ADR 0019).
- 라벨은 덮어쓰지 않는다. 수정은 새 레코드 + `parent_label_id`. 번역(영어 설명)은 수정이 아니라 별도 운영 라벨이라
  parent가 아니라 `DescriptionPayload.source_label_id`로 원문을 가리킨다 (ADR 0043).
- 다른 단계의 입력은 `current_labels()`(운영 라벨)로 고른다. 오류 삽입 레코드와 그 후손, 블라인드·이중 측정
  레코드(`measurement`)는 운영 라벨이 아니며 학습·내보내기에 들어가면 안 된다.
- 모듈 사이 계약 데이터(세션·라벨·검수·데이터셋 등)는 `dlp_schema` 타입으로 주고받는다. 저장소·렌더 게이트 같은
  인프라 API를 다른 패키지에서 쓰는 것은 `docs/handover/architecture.md` 의존 표 안에서만 한다
  (`tests/test_import_boundaries.py`가 표와 실제 import를 대조한다). 다른 모듈의 `_` 이름은 import하지 않는다.
  계약 변경은 ADR + Alembic 마이그레이션 + 계약 테스트 + `make schemas`를 함께 한다.
- DB 스키마는 `dlp_schema/db/tables.py`와 새 Alembic 리비전을 함께 바꾼다. 테스트가 둘의 일치를 검사한다.
- 모든 datetime은 시간대 정보가 있어야 한다 (UTC 권장).
- 영상 시각은 PTS 인덱스로만 계산한다. 프레임 번호에 프레임 간격을 곱해 계산하지 않는다.
- 원본 버킷(`dlp-raw`) URI를 일반 라벨러 경로(블러본, 검수 작업, 내보내기)에 노출하지 않는다.
  `dlp_review.roles.check_stage_uris`로 검사하고, 라벨러용 URL은 라벨러 자격 증명(라벨링 버킷 읽기 전용)으로 서명한다.
- 정책 값(비율, 허용 오차, 임계값)은 `config/`에서 읽는다.
- 골든·학습·검증 사이에 작업자나 장소가 겹치면 안 된다. 분할은 `dlp_datasets.splitter`로만 만든다.
- 새 모델은 공통 `Predictor` 어댑터와 CPU용 stub 구현을 함께 추가한다. CI는 stub과 준비된 실제 모델 경로를 검사한다.
- 실제 모델을 연동하지 못하고 stub만 둔 곳에는 주석 `TODO(real-model): <무엇이 필요한지>`를 단다 (코드와 정책 YAML 모두).
  연동하면 표시를 지운다.
- 라벨링한 데이터를 판매하므로 모델의 상업 사용 여부를 따진다 (ADR 0010). 새 모델은 `config/models.yaml`에 가중치
  라이선스, 직접 학습 데이터와 그 라이선스, `commercial` 분류를 적는다. 가중치가 비상업(`forbidden`)이면 쓰지 않는다.
  데이터는 당분간 판매하지 않고 사내 모델 개발에 쓰지만(2026-10-24), 라이선스 분류·forbidden 금지·개인정보·내보내기
  규칙은 그대로 적용한다 (ADR 0058).
- 학습 예제는 데이터셋 버전의 학습·검증 분할에서만 뽑는다 (`dlp_train.extract`). 후보 모델의 골든셋 예측은
  DB에 쓰지 않는다. 배포는 게이트를 통과한 모델만, 프라이버시는 사람 승인 후에만 한다.
- 각 파이프라인 단계는 멱등적인 `dlp <단계>` 하위 명령으로 만든다. 모델 버전에는 그 단계가 쓰는 정책 절의 해시를 넣는다. 모델 출처 라벨 ID에는 `version_tag(모델 버전)`을
  넣고, 일반 단계는 전체 이력(`get_labels`)으로 재실행을 정한다.
  프라이버시는 최신 완료 세대의 활성 버전 집합으로 정하며, 과거 정책 복귀도 새 세대다 (ADR 0033). 버전이 바뀌면 검수 전인 이전 버전만
  `retractions()`로 지운다. 검수자가 승인·표본 검증한 라벨은 어떤 단계도 지우지 않는다 (ADR 0015, 0019).
  좁은 예외 둘. (1) 행동 검수를 물려받기만 한 모델 설명은 물려준 행동이 다음 회차에 의미가 다시 바뀌거나 지워지면
  설명 따르기(`review-follow-1`)만 지울 수 있다 (ADR 0043 3절 규칙 6). (2) 2D 트랙(손 키포인트·객체 박스) 검수를
  물려받기만 한 모델 3D 궤적은 출처(`derived_from`의 2D 트랙, 월드 좌표면 카메라 좌표 궤적·카메라 자세)가 바뀌거나
  지워지면, 또는 같은 출처 2D에서 3D 단계가 다른 판(내부 파라미터·작업자 손 크기·depth 정책·추정기 판·자세 입력 변경)으로
  새로 계산한 검수 전 3D가 같은 (스트림, 개체, 부위, 좌표계)에 생기면(새 3D가 같은 2D 검수를 물려받는다) 궤적
  따르기(`trajectory-follow-1`)만 지울 수 있다 (ADR 0048·0050·0051). "물려받기만 함"은 추정하지 않고 명시
  기록(설명 `description_inheritance`, 3D `trajectory_inheritance`, 추가만, 물려받는 `record_review`와 같은
  트랜잭션)으로만 정한다: 그 라벨의 지금 검수 상태가 마지막 물려받기 기록과 같을 때뿐이다. 저장 검증
  (`label_validation`)은 같은 이력으로 다시 계산한 따르기 계획이 지우는 것만 허용한다. 기록 없는 검수(검수자·시각이
  우연히 같아도)와 기록 뒤 직접 검수, 사람이 쓰거나 고친 라벨은 보호한다.
- 내보내기는 데이터셋 버전에서만 만들고(`dlp_export`), 기본은 사람이 만들거나 승인·수정·표본 검증한 라벨만 넣는다.
  미검수는 명시적 옵션으로만. 블러 라벨·원본 위치·검수자 ID는 어떤 형식에도 넣지 않는다.
- 원본 버킷 저장소는 `dlp_cli.raw_access.raw_store`(감사 저장소)로만 만든다 (DB 없는 로컬 개발은
  `raw_store_offline`, 기록은 로컬 JSON Lines). 모든 원본 접근은 `raw_access_log`에 남고, 운영·감사 기록은 추가만
  한다 (ADR 0020, 0021). 정적 검사 테스트가 우회(`buckets.raw`, `raw_bucket()`)를 막는다.
- 블러(원본 영상) 검수는 `review.yaml reviewers.privacy`의 원본 접근 권한자에게만 배정한다.
- AI 1차 검수 판정은 라벨도 검수 상태도 아니다 (ADR 0040). 판정은 추가 전용 `ai_review_*` 표에만 쓰고 `label_records`·
  `record_review`에 쓰지 않는다. 판정 입력 픽셀은 `dlp_aireview.media.open_blurred`(지금 승인된 블러본)로만 받는다.
  AI 제안은 검수자가 명시적으로 적용할 때만 사람 라벨이 되며, 제안 내용은 DB에서 다시 읽어 검증한다 (`dlp_review.ai_collect`).
  AI의 ok는 사람 검수·표본을 줄이지 않는다. 판정 서버는 사내 서버만(외부 API 금지). Label Studio 설정의 AI 컨트롤은
  모드와 무관하게 늘 둔다.
- 테스트는 정답을 아는 합성 픽스처(WP2)로 작성한다. 실제 영상·개인정보를 저장소에 넣지 않는다.
- 함수 복잡도 상한은 15다 (ruff C901). 상한을 넘는 기존 함수의 `# noqa: C901`은 분해 대상 기준선이며 새로 달지 않는다.
- 새로 찾은 문제는 `docs/handover/issues.yaml`에 넣는다. 확정 결함은 `tests/test_known_defects.py`에
  `xfail(strict=True)` 재현을 두고, 고치면 표시를 지우고 status를 fixed로 바꾼다 (`tests/test_issue_registry.py`).
- 1인칭 바디캠의 MediaPipe 손 판정은 뒤집지 않는다 (`hands.swap_handedness: false`, F-PAR-61).
- 프라이버시 탐지는 시간 거르기·오탐 힌트로 지우거나 바꾸지 않는다. 오탐 의심은 검수 화면 표시로만 쓴다 (ADR 0054·0055).
- 보간해 채운 손 키프레임(`interpolated`)은 관측이 아니다. 3D로 올리지 않는다 (ADR 0054).
- 손 크기로 추정한 내부 파라미터는 `estimated_hand_size`로 밝히고 `calibrated`로 쓰거나 실제 캘리브레이션을 덮지
  않는다. 메트릭 깊이 모델의 화각 사전값을 되돌려 주므로 기본 끈다 (`depth.focal_estimate.enabled: false`, F-PAR-65).
- 학습 증강의 입력은 데이터셋 버전의 학습·검증 분할 블러본만이다. 증강본은 라벨 이력·판매 내보내기에 넣지 않는다 (ADR 0057).
- 1인칭 바디캠에서 착용자 자신의 팔·몸통 박스·자세는 객체·사람 라벨로 만들지 않고 단계 요약에만 센다. 프라이버시
  탐지·블러는 이 억제와 무관하다 (ADR 0058).
- 팔꿈치가 화면 밖이거나 판정 검사를 통과하지 못하면 visibility 0으로 두고 위치를 지어내지 않는다. 팔 손목은 같은 손
  hand21 손목이다 (ADR 0058).
- 모양 마스크는 프라이버시 블러와 무관하다 (블러는 박스 그대로). 착용자 팔·몸통은 마스크 프롬프트로 쓰지 않는다 (ADR 0058·0059).
- 코드 주석과 문서는 한국어, 식별자는 영어.
