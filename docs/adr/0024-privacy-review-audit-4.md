# ADR 0024: 프라이버시 게이트·검수 연동 4차 감사 정정

- 상태: 채택
- 날짜: 2026-10-07
- 관련: WP5(프라이버시), WP6·WP12(검수 연동·운영), ADR 0019, ADR 0020, ADR 0023.
  계약(dlp_schema)·DB 변경 없음. 정책 추가: `config/policies/review.yaml cvat`.

## 결정

1. **CVAT 모양(Shape) 주석도 받는다.** CVAT 기본 그리기 모드는 모양이다. 수집이 트랙만 읽어, 검수자가
   블러 검수에서 놓친 얼굴을 직사각형 하나로 그리면 버려지고 승인·렌더가 그대로 통과했다. 이제
   `get_annotations`로 트랙·모양·태그를 모두 읽고(`dlp_review.cvat.annotation_tracks`), 모양 직사각형은
   (그 프레임 키프레임 + 다음 프레임 outside) 트랙으로 바꾼다. CVAT 화면처럼 그 프레임에만 보이고
   렌더도 같다. 점 모양은 키프레임 하나짜리 키포인트 트랙이다. 블러 작업이면 blur_track, 작업 라벨
   작업이면 box_track. 옮길 수 없는 주석(프레임 태그, 다각형·타원·마스크·스켈레톤, 회전 직사각형,
   라벨에 맞지 않는 모양, 영상 밖 프레임)은 조용히 버리지 않고 수집을 멈춘다(`CvatFormatError` →
   `TaskError`). 트랙의 모양 종류도 같은 규칙으로 검사한다.
2. **블러본은 지금 승인된 블러 라벨로 렌더한 것만 쓴다.** 렌더 해시를 렌더할 때만 봐서, 승인이 풀리거나
   (블러 QA·재탐지) 다시 승인된 뒤에도 이전 블러본이 검수 작업·내보내기·VLM·큐레이션에 쓰였다.
   - `dlp_privacy.runner.assert_render_current(conn, labeling, 세션, 스트림, 정책)`: 세션이 **지금 DB
     상태로** 승인(APPROVED)이고, 블러본 옆 렌더 기록(`<스트림>.render.json`)의 해시가
     `render_hash(지금 운영 블러 라벨, 렌더 정책)`과 같아야 한다. 아니면 `RenderNotCurrentError`.
   - 렌더 기록에 블러본 파일 sha256을 함께 둔다. 확인한 뒤 받은 파일이 그 파일인지(`check_fetched`),
     저장소의 블러본이 그 기록의 파일인지 본다 (렌더 밖에서 덮어쓴 파일 금지).
   - 쓰는 곳: `create_labeling_tasks`(TaskError), 내보내기 `load_source`(데이터셋 스냅샷이 아니라 지금
     DB 상태로 승인 확인, 스트림별 기대 해시를 `ExportSession.render_hashes`에 둔다)와
     `fetch_blurred`(ExportError), 행동 VLM 프레임(`run_actions`), 큐레이션 `build_samples`(건너뛰고
     이유를 남긴다).
   - 승인 취소 때 무효화: 블러 검수 수집(`collect_task`)이나 재탐지(`detect_session(labeling=…)`,
     `dlp privacy detect`)로 운영 블러가 바뀌면 그 스트림의 렌더 기록을 `render_hash=null`로 덮어쓴다.
     라벨링 버킷 저장소에는 삭제가 없고, 블러본 파일은 파생물이라 남겨도 된다: 모든 소비자가 위 확인을
     거치므로 무효 기록이나 해시가 다른 블러본은 쓰이지 않는다. 무효화를 거치지 않은 경로가 있어도
     블러 라벨 집합이 다르면 해시가 달라 막힌다.
   - 내보내기·행동·큐레이션 패키지가 `dlp-privacy`에 의존한다 (pyproject, uv.lock).
3. **탐지 모델 버전에 정책 해시.** `dlp_privacy.pipeline.model_version`에 탐지 결과를 정하는 정책 값
   (detection_threshold, review_score·review_priority, 대상별 여유·탐지기, tracker, blur_hold_ms, 쓰는
   탐지기 설정 — 반사면 탐지기가 쓰는 영역·얼굴 탐지기 포함: YuNet nms·top_k, OWLv2 질의·문턱 등)의
   짧은 해시(`+p<8자>`)를 붙인다. 바꾸면 다시 탐지하고 검수 전인 이전 블러는 지운다 (ADR 0019와 같은
   규칙). 렌더 설정은 넣지 않는다 (블러본 해시가 맡는다). label_records.model_version은 Text라 길이
   제한이 없다.
4. **CVAT 작업 배정.** `review.yaml cvat.users`(dlp 검수자 ID → CVAT 사용자 이름, 일대일)로 작업을 만든
   뒤 작업(task)과 모든 job의 담당자를 그 CVAT 계정으로 정한다. 블러 검수는 연결이 없으면 원본 프록시를
   올리기 전에(grant 감사 기록 전에) 멈춘다. 작업 라벨 작업은 연결이 있으면 배정한다.
   웹훅(`resolve_reviewer`)은 CVAT 작업이면 job 담당자 사용자 이름이 담당자의 연결 계정과 같아야 받는다.
   연결이 있는데 job 담당자가 없거나 다르면 거부하고, 블러 검수는 연결이 없어도 거부한다.
   - **dlp-privacy 프로젝트 공개 범위**: CVAT는 관리자(superuser/staff)에게 모든 프로젝트를 보여 준다.
     일반 사용자(user·worker 그룹)는 자기가 만들었거나 담당자인 작업·job만 본다. 그래서
     - dlp-privacy 프로젝트는 서비스 계정(관리자)이 만들고 소유한다. 검수자를 프로젝트 담당자로 두지 않는다.
     - 검수자 CVAT 계정은 관리자가 아닌 일반 사용자(worker 권장)로 만든다. 원본 접근 권한자가 아닌
       라벨러가 관리자 권한을 받으면 원본 프록시가 보이므로 관리자 계정은 플랫폼 운영자만 갖는다.
     - 조직(organization)을 쓰면 dlp-privacy를 원본 접근 권한자만 있는 조직에 두고, 작업 라벨 프로젝트와
       나눈다. 조직·권한 설정은 CVAT 관리 화면에서 하고 코드로는 만들지 않는다 (개발 compose는 조직 없이
       돈다). 운영 배치 점검 목록에 넣는다.
5. **블러 계획은 모든 영상 스트림.** `plan_session --privacy`가 라벨 없는 단위를 건너뛰어 탐지 0개
   스트림은 배정이 없고(승인 불가), 배정 ID가 고정이라 재탐지·승인 취소 뒤 다시 계획해도 새 배정이 없었다.
   블러 단위는 라벨이 없어도 계획하고, 배정 ID에 세대 태그(`g<해시>`: 마지막 자동 탐지 시각 + 현재 운영
   블러 라벨 ID 집합)를 붙인다. 같은 상태에서 다시 계획하면 멱등, 바뀌면 새 배정이 생긴다.
6. **오류 삽입 사본은 마지막 탐지 시각에 넣지 않는다.** 오류 삽입 계획이 모델 출처 블러 사본을 지금
   시각으로 만들어 `last_detection`이 움직여 승인이 막혔다. 운영 라벨이 아닌 레코드(오류 삽입·측정과
   그 후손, `non_operational_ids`)는 세지 않는다.
7. **렌더 보간은 프레임 번호로.** CVAT는 키프레임 사이를 프레임 번호로 보간한다. 블러본이 시각 비율로
   보간해 VFR 영상에서 검수 화면과 박스가 달랐다. 렌더할 때 그 영상의 PTS 인덱스 프레임 순서로
   보간한다 (프레임 번호는 계산에만 쓰고 저장하지 않는다, ADR 0019 시각 규약 유지).
8. **검수 우선 구간.** 재학습 블러 모델만 다시 돌면 `privacy_review/<스트림>.json`을 그 모델 구간으로
   덮어써 탐지기 구간이 사라졌다. 이전 구간과 합친다(`merge_segments`: 다시 돈 쪽만 바꾸고, 더 쓰지 않는
   모델 구간은 뺀다). 블러 검수 작업을 만들 때 구간을 우선순위 순으로 CVAT 이슈(`cvat.privacy_issue_limit`
   개까지)로 남겨 검수자가 먼저 볼 프레임을 안다.
9. **갈래 이력 방지.** 같은 라벨을 보낸 두 작업이 차례로 고치면 한 라벨에 자식이 둘이 되어 두 수정본이
   모두 현재 라벨이 됐다. 수집 때 보낸 라벨에 이미 자식 레코드가 있으면(다른 단계가 지웠거나 다른
   작업이 먼저 고쳤으면) 그 라벨의 검수 결과(승인·수정·삭제)를 빼고 `stale`로 알린다.
10. 검수 시계열 CSV의 장갑 압력 채널 접두사는 `sync.yaml glove.pressure_prefixes`를 읽어 명시해 넘긴다
    (`ReviewSetup.pressure_prefixes`).

## 운영 절차 변화

- 블러 검수 담당자는 `review.yaml reviewers.privacy`와 `cvat.users` 둘 다에 있어야 한다.
- `dlp privacy approve` 뒤에는 항상 `dlp privacy render`를 다시 한다 (렌더 기록이 무효이거나 해시가
  다르면 작업 라벨 작업·내보내기·행동·큐레이션이 멈춘다).

## 남은 일

- CVAT 조직·프로젝트 권한은 문서 절차로만 지킨다 (위 4번). 운영 CVAT에서 조직을 나누면 클라이언트가
  `X-Organization` 헤더를 붙이도록 바꿔야 한다.
- CVAT 작업 시간은 여전히 재지 않는다 (`dlp ops log-work`).
