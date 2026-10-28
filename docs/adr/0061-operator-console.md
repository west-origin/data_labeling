# ADR 0061: 운영 콘솔과 설치 스크립트

- 상태: 제안 (초안, 2026-10-27)
- 날짜: 2026-10-27
- 관련: ADR 0020·0021(원본 접근 감사), ADR 0023·0024(검수 권한), ADR 0033(탐지 세대), ADR 0038(검수 화면
  테마), ADR 0040(AI 1차 검수), ADR 0060(공개 데이터셋 1차 학습), F-COE-74·75.
- 계약 변경: 없음 (DB 스키마·라벨·내보내기 그대로, Alembic 리비전 없음). 새 패키지 `packages/console`
  (`dlp_console`), 새 설정 `config/console.yaml`, 새 하위 명령 `dlp console serve`, 설치 스크립트
  `scripts/install/`.

## 배경

플랫폼은 단계별 `dlp` 명령의 묶음이고, 정상 순서·선후 조건·실패 대응이 문서(weekend-video-pilot.md,
pilot-real-2026-10-22.md)에 흩어져 있다. 사용자는 비개발자 운영자 한 명이다. 터미널 없이 브라우저 한
화면에서 "영상 올리기 → AI 1차 라벨 → 결과 보기 → 검수 도구에서 수정 → 수집·검수 완료 → 데이터셋 버전 →
재학습(골든셋 게이트) → 회차별 정확도·수정률"을 진행해야 한다. 또 어떤 컴퓨터에든 같은 방법으로 설치할
수 있어야 한다.

## 결정

1. **콘솔은 도메인 로직을 갖지 않는다.** 각 버튼은 기존 `dlp <단계>` 명령을 자식 프로세스로 돌리는
   작업(job)이다 (`python -m dlp_cli.main …`, 셸 없음, 인자 목록). 단계 명령은 멱등이라 다시 눌러도
   안전하다. 프라이버시 승인·검수 완료·학습 게이트·원본 접근 권한 검사는 명령이 그대로 하고, 콘솔의
   활성 규칙(`dlp_console.steps`)은 "앞 단계가 끝났는가"를 미리 보이는 안내일 뿐이다 (막힌 단계는
   서버도 거부한다). 블러 승인은 사람이 누르는 버튼이고 확인 대화를 거쳐 `dlp privacy approve`를 부른다.
2. **상태는 저장소 밖 작업 디렉터리에 추가만 한다.** `DLP_CONSOLE_DIR`(기본 `~/.dlp-console`, 권한 0700):
   `jobs.jsonl`(created·started·finished·interrupted 사건), `logs/<작업>.log`, `uploads/<세션>/`(업로드
   사본·매니페스트), `pretrain/`·`pretrain_imports.jsonl`(가중치 가져오기), `reports/`(평가·주간 지표),
   `cache/reports/`(평가 리포트 캐시). **DB 스키마는 바꾸지 않는다.** 이유: 작업 기록은 운영자 한 명의
   화면 상태라 계약 데이터가 아니고, 단계 결과(세션 상태·라벨·검수 작업·모델 버전)는 이미 DB에 있다.
   여러 운영자·여러 컴퓨터가 같은 작업 기록을 봐야 하면 그때 `console_jobs` 표(추가 전용)와 마이그레이션을
   따로 정한다. 같은 잠금 키(세션 하나, 또는 데이터셋·학습·평가·리포트 묶음)의 작업이 돌면 새 작업을 받지
   않는다. 콘솔이 꺼져 끝 사건이 없는 작업은 다음 시작 때 interrupted로 남긴다.
3. **화면 자료는 읽기만 한다.** 세션·탐지 세대·검수 작업·모델 버전은 기존 저장소 함수로 읽는다. 라벨 수는
   화면용 SQL 집계(삭제·오류 삽입·측정·대체된 레코드 제외, 오류 삽입 후손은 근사)이고 단계 입력이 아니다.
   결과 보기의 겹쳐 그리기는 `current_labels()`의 박스·키포인트 트랙만 쓴다 (블러 트랙 제외).
4. **원본·비밀 경계.**
   - 응답에 원본 위치(스트림 `uri`·`pts_index_uri`, 검수 작업 `media_uri`)를 싣지 않는다. 로그 꼬리는
     원본 버킷 URI·경로와 비밀값(이름에 PASSWORD·SECRET·TOKEN·KEY가 든 환경 변수 값)을 가린다.
   - 영상 미리보기는 블러 승인·렌더 뒤에만, 라벨링 버킷 블러본을 라벨러 자격 증명(읽기 전용)으로 서명한다.
     렌더 여부도 라벨러 자격 증명 HEAD로 본다. 콘솔은 원본 버킷을 읽지 않는다 (감사 저장소가 필요 없다).
   - 평가 리포트는 MLflow 산출물 버킷 URI만 읽는다.
   - 기본은 127.0.0.1에만 묶는다. 다른 주소는 `--allow-remote`가 있어야 하고 경고한다. 모든 `/api/*`는
     시작 때 만든 토큰(`X-Console-Token`, 콘솔 주소의 `#token=`로 전달, 주소창에서 곧 지움)이 맞아야
     한다. CORS를 열지 않아 다른 사이트가 머리글을 붙일 수 없다 (CSRF). Host 머리글은 묶은 주소·localhost만
     받는다 (DNS 재바인딩). 인증 체계(S1)는 아니다: 로컬 한 사람용이다.
5. **업로드.** 영상 + 세션·작업자·장소 ID, 도메인, 장면, 확인한 동의서 버전(필수, 지어내지 않음), 촬영
   시각(선택), 카메라 ID(선택), 촬영 방향 회전(선택) → 수집 모델(`SessionManifest`)로 검증한 매니페스트 →
   `dlp media prepare [--display-rotation]` → `dlp ingest`. 사본은 작업 디렉터리에만 두고 수집이 성공하면
   지운다 (`upload.delete_after_ingest`). 원본은 원본 버킷에 불변으로 남는다.
6. **외부 학습 가중치 가져오기 (ADR 0060 1단계 산출물).** ONNX + `model_manifest.json` +
   `data_manifest.json`을 받아 종류·해시 연결·ONNX sha256·`decode_in_inference` false·클래스 순서
   (`config/pretrain/objects.yaml` = `prelabel.yaml closed_vocab_tools.classes`)·입력 크기·데이터셋(목록에
   있고 forbidden 아님)을 검사하고, `config/models.yaml` 항목(path·sha256·license·training_data·
   training_datasets·base_model·commercial = 데이터셋·출발 가중치 중 가장 엄한 분류)을 만든다. 새 항목을
   더한 목록을 메모리에서 `ModelRegistry`·`dataset_violations`로 검사한다.
   **콘솔은 Git이 관리하는 설정 파일(models.yaml·prelabel.yaml·training.yaml)을 직접 고치지 않는다.**
   이유: 정책 파일은 모델 버전 해시·라이선스 검사(`make licenses`)·코드 검토의 대상이고, 화면 버튼 한
   번으로 바뀌면 누가 왜 바꿨는지가 Git 이력에 남지 않는다. 그래서 미리보기(diff)와 패치 파일(작업
   디렉터리), `git apply`·`make licenses` 안내까지만 하고, 운영자가 확인하면 가중치를 `data/models/pretrain/`
   (Git 밖)에 두고 가져오기 기록을 남긴다. 우리 데이터를 밖으로 보내는 기능은 없다 (ADR 0060 6절).
   골든셋 비교는 학습 루프 게이트 경로(`dlp train run objects <버전> --trainer <학습기>`)로만 하고 예측은
   DB에 쓰지 않는다. 지금 저장소에는 외부 ONNX를 그렇게 평가하는 학습기·로더가 없어(F-COE-74; 작업 중인 yolox
   학습기는 .pth 출발 가중치를 우리 데이터로 다듬는 2단계 경로다) 학습기 이름은
   `config/console.yaml pretrain_import.compare.trainer`(기본 null → 버튼 꺼짐)로 둔다. 로더·학습기는
   dlp_train·dlp_prelabel 쪽 구현을 쓰고 콘솔이 만들지 않는다. closed_tools 단계 켜기(prelabel.yaml
   `models.closed_vocab_tools`)는 그 학습기 회차가 게이트를 통과(passed·deployed)했을 때만 안내한다.
   주의: training.yaml `objects.deploy: auto`면 게이트 통과가 곧 배포다.
7. **설치 스크립트** `scripts/install/install.sh`(Mac·Linux·WSL2): 사전 점검(Docker·Compose v2, 디스크
   60 GB·메모리 16 GB 권장 경고, NVIDIA GPU·nvidia-container-toolkit 감지, Apple Silicon CVAT 에뮬레이션
   경고, uv), `make install`, `.env`(없을 때만, 비밀번호·키 무작위, 0600), `make up`, `make db-upgrade`,
   `make models`, 선택 `make cvat-up`·`make cvat-superuser`. 기존 Make 대상만 부른다. Windows는 WSL2
   안에서 같은 스크립트를 쓴다 (README).
8. **화면.** 정적 HTML+JS(빌드 단계·번들러·외부 CDN 없음), 한국어, `config/review_ui.yaml` 테마 색·로고·
   다크/화이트(`dlp_review.ui_theme` 로더). 그래프는 인라인 SVG. 학습기 이름이 `oracle`로 시작하면
   "시험용 학습기" 배지를 단다. 웹 서버는 FastAPI(MIT)·uvicorn(BSD-3)·python-multipart(Apache-2.0),
   정확한 버전 고정.

## 결과

- 운영자가 터미널 없이 파이프라인을 진행한다. 단계 조건·실패 안내가 화면 한 곳에 모인다.
- 콘솔 의존: media(저장소·매니페스트 모델), models(레지스트리·데이터셋 검사), review(테마·검수 정책
  로더), schema. CLI는 `dlp console serve`만 더한다.
- 한계: 작업 기록이 한 컴퓨터의 파일이다. 인증은 토큰 하나다(외부 노출 금지). 결과 보기는 가까운 앞
  키프레임을 그리는 근사다. Label Studio 작업 주소는 프로젝트 목록까지만 연다. 동기화·관계·행동·AI 1차
  검수의 완료는 DB 흔적이 없어 콘솔 작업 기록으로만 본다.
- 남은 일: 외부 ONNX 골든셋 비교 경로(F-COE-74), models.yaml의 로컬 가져온 파일 출처 형식(F-COE-75),
  여러 운영자용 작업 표(필요해지면).

## 개정 (2026-10-28, 전수 검사 후속)

- 외부 가중치 가져오기(F-COE-77): manifest의 구조 이름은 식별자 모양(`^[a-z][a-z0-9_]{0,63}$`)만, 출발 가중치는
  sha256 64자·https 주소만 받는다. 가중치 경로는 resolve 뒤 `pretrain_import` models_dir 안이어야 하고, confirm이
  저장된 계획을 다시 검사한다. 패치 주석에 넣는 문자열은 출력 가능한 한 줄로 바꾼다.
- 작업 실행(F-COE-79): 자식 `dlp`는 새 프로세스 그룹으로 돌고 `spawned` 사건에 pid·pgid를 남긴다. 콘솔이 끝날 때
  그룹을 멈추고(`console.yaml jobs.stop_grace_s` 뒤 SIGKILL) interrupted로 남긴다. 재시작 때 앞 콘솔의 그룹이 살아
  있으면 `adopted`로 실행 중을 유지해 2절의 잠금이 깨지지 않는다.
- Host 검사(F-COE-81): 머리글은 포트를 떼고(`[::1]:8090` → `::1`) IP는 표준형으로 묶은 주소·localhost와 비교한다.
  `--allow-remote`로 모든 주소(0.0.0.0·::)에 묶으면 IP 리터럴 Host만 더 받는다(도메인 이름은 계속 거부, 재바인딩 방지).
  운영 서버는 `testserver`를 받지 않는다 (시험은 `create_app(extra_hosts=…)`). 토큰은 바이트로 비교해 비ASCII도 401이다.
- 입력 오류(F-COE-80·83·84): 업로드 사본 삭제는 정규화한 세션 ID로 실행 중을 검사하고 지운다. 동의서 버전은 자르지
  않고 64자를 넘으면 거부한다. 단계 API 본문이 잘못된 JSON·객체 아님·NaN·무한대면 400이다.
