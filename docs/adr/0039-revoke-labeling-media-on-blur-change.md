# ADR 0039: 블러가 바뀌면 이미 배포한 라벨링 작업 매체를 회수한다

- 상태: 채택
- 날짜: 2026-10-11
- 관련: WP5(프라이버시), WP6·WP12(검수 연동·운영), ADR 0020(원본 접근 감사), ADR 0024(블러본은 지금
  승인된 블러로 렌더한 것만), ADR 0033(완료 탐지 세대), F-REV-02.
  계약 변경: `ReviewTaskStatus.REVOKED`, `ReviewTask` 필드 6개, `review_tasks` 열 6개(Alembic 0015),
  `schemas/review_task.schema.json`.

## 배경

ADR 0024는 블러가 바뀌면 승인을 풀고 렌더 기록을 무효로 둬, 블러본을 **새로 읽는** 모든 소비자가
막히게 했다. 그런데 이미 만든 일반 라벨링 작업은 그 확인을 다시 거치지 않는다.

- CVAT 공간 검수 작업에는 라벨러 워터마크 사본이 CVAT 안에 올라가 있다. 버킷 객체를 지워도 CVAT 쪽
  노출은 끊기지 않는다.
- Label Studio 작업에는 `sessions/<세션>/review/<담당자>/<스트림>.mp4`의 서명 URL(기본 7일)이 박혀 있다.

승인 뒤 블러 재검수(또는 재탐지)가 놓친 얼굴을 찾으면, 일반 라벨러는 그 얼굴이 블러되지 않은 영상을
계속 본다 (CLAUDE.md: 원본 내용을 일반 라벨러 경로에 노출하지 않는다).

## 결정

1. **outbox.** 블러를 바꾸는 트랜잭션 안에서 회수 요청을 `review_tasks`에 기록해 같은 커밋에 넣는다
   (`revoke_requested_at`·`revoke_reason`, 열린 작업은 `status=revoked`). 외부 채널은 커밋 **뒤에**
   실행기(`dlp_review.revoke.execute_media_revocations`)가 끊고, 확인한 채널만 짧은 트랜잭션으로
   표시한다 (`media_revoked_at`, `tool_revoked_at`). 둘 중 하나라도 비면 회수 대기다.
   - 버린 안: 트랜잭션 안에서 도구·S3를 지우고 하나라도 실패하면 전체를 되돌리는 방식(fail-closed).
     도구·S3 장애가 블러 검수 수집 전체를 롤백하면 새 블러 라벨과 승인 해제(AUTO_BLURRED)가 기록되지
     않아 DB는 APPROVED로 남는다. 저장소에는 무효 렌더 기록만 남고 그 오류 문구가 `dlp privacy
     render`를 안내하므로, 운영자가 따르면 이전 라벨로 다시 렌더되어 새로 찾은 얼굴이 빠진 블러본이
     내보내기와 새 작업에 다시 열린다. "새 노출 차단"과 "기존 노출 회수"를 묶으면 도구 장애가 두 쪽을
     모두 막는다. 웹훅 503 뒤 도구가 자동 재전송한다는 보장도 없다.
2. **판정 불변식.** 작업에 저장한 `render_hash`(그 매체를 만든 블러본의 렌더 해시) ≠ 지금 운영 블러의
   `render_hash(operational_blur(...), policy)`이면 낡았다 (`dlp_privacy.revocation.stale_media_tasks`).
   `expected_render_hash`는 승인 상태를 요구해 승인이 풀린 직후에 쓸 수 없어 쓰지 않는다. 블러가 지금
   바뀐 스트림에서는 렌더 해시가 없는 구 작업도 낡은 것으로 본다(legacy): 지금 트랜잭션에서 생긴
   라벨을 이전 작업이 담았을 수 없다. 0건 재탐지처럼 블러 집합이 그대로면 해시가 같아 회수하지 않는다.
   대상은 같은 세션의 비프라이버시 작업(open·collected, 모든 방식)이다. 수집한 CVAT 작업도 담당자가
   다시 열 수 있고 서명 URL도 유효하기 때문이다. 블러 검수 작업(원본 권한자 화면)은 SQL·계약 검증기
   양쪽에서 제외한다.
3. **내용 주소 키와 회수 단위.** 워터마크 사본 키를 `<담당자>/<스트림>-<렌더 해시 앞 16자>.mp4`로
   바꾼다. 블러본이 바뀌면 키도 바뀌어 회수할 사본과 새 사본이 섞이지 않는다. 같은 매체 URI를 쓰는
   작업은 함께 요청한다(`…:shared-media`). 회수 단위는 매체 객체다.
4. **요청 지점.** 모두 세션 잠금 아래에서 요청한다.
   - 블러 검수 수집(`collect._after_blur_change`): 승인 해제 → 그 스트림 요청(legacy) → 렌더 무효.
     승인 상태와 무관하게 부른다 (이전 승인 때 만든 작업이 AUTO_BLURRED 세션에도 남을 수 있다).
   - 재탐지(`detect_session`): 블러 레코드를 쓴 스트림만(legacy). `DetectSummary.changes`·
     `changed_streams`·`revocations`.
   - 렌더 백스톱(`dlp privacy render`): 모든 영상 스트림, 해시가 다른 작업만. 렌더 해시에는
     `privacy.yaml render` 절 전체(`min_block_px`·`blocks_per_box`와 블러와 무관한 인코더 설정
     `encoder_rate`·`crf`)와 `render_mode`가 들어간다. 그래서 블러 강도뿐 아니라 화질(crf)이나 명목
     프레임레이트만 바꿔도, 그 정책으로 다시 렌더하는 세션마다(그리고 create·assign sweep에서) 해시
     있는 라벨링 작업이 모두 회수되고 도구 작업이 지워지며, 다시 만들 때까지 verify가 막힌다. 블러
     강도 변경은 프라이버시 변경이라 의도한 동작이다. 인코더 설정은 블러 내용을 바꾸지 않지만 블러본
     파일이 바뀌어 배포 사본과 달라지므로 같은 규칙을 둔다. 운영상 결과: render 절은 진행 중 라벨링이
     없을 때 바꾸고, 바꾼 뒤 렌더하기 전에 `dlp review revoke <세션> --dry-run`으로 영향받을 작업을
     확인한다.
   - `dlp review revoke <세션>`: 늘 sweep(사유 `sweep`)을 한다. `--task KEY`·`--all`(사고 대응)은
     그 위에 수동 요청(사유 `manual:<행위자>:<사유>`)을 더한다. 수동 선택(`task_keys`·`force`)은
     지정한 것(과 같은 매체를 쓰는 작업)만 고르고 해시 판정을 섞지 않아, 낡은 작업이 수동 사유로
     기록되지 않는다. `--task` 키가 없는 작업·다른 세션의 작업·블러 검수 작업이면 아무것도 쓰지
     않고 거부한다 (조용히 빠져 운영자가 회수됐다고 오해하지 않게).
   - `dlp review create|assign` 시작(sweep).
   - 사용 중지(`dlp dataset withdraw`, `dlp_datasets.lineage.withdraw_session`, F-REV-17 후속):
     생애주기를 withdrawn으로 바꾸는 트랜잭션(세션 행 잠금)에서 그 세션의 비프라이버시 작업 중
     요청이 없는 것을 렌더 해시와 무관하게 모두 요청한다 (사유 `withdrawn:<행위자>`). 동의를 철회한
     영상은 블러가 맞아도 쓰면 안 된다. 쓰기는 `dlp_schema`의 `request_task_revocation`이라 의존 표
     (datasets → schema)는 그대로다. CLI는 커밋 뒤 실행기를 돌리고 대기가 남으면 종료 코드 4다.
     사용 중지한 세션에는 블러 검수·라벨링 작업을 새로 만들지 않는다 (`tasks._check_not_withdrawn`,
     도구 호출 전). 이 변경 전에 사용 중지한 세션은 `dlp review revoke <세션> --all --reason
     withdrawn`으로 회수한다. 블러 검수 작업(원본 권한자 CVAT 작업)은 회수 대상이 아니라 남는다.
   - dlp_privacy가 `review_tasks`에 쓰는 근거: 블러를 바꾸는 트랜잭션(탐지·렌더)이 이 패키지에 있고
     요청은 같은 커밋에 있어야 한다. `generation`도 이미 `review_tasks`를 읽는다. 쓰기는 `dlp_schema`
     저장소 함수(`request_task_revocation`)로만 하고 패키지 의존 표는 그대로다.
5. **실행 순서.** "넓고 확실한 것부터": 라벨링 버킷 사본 삭제(서명 URL이 즉시 끊긴다) → CVAT(배정
   해제 → 스냅숏 → 삭제 → 없음 확인) → Label Studio(스냅숏 → 삭제 → 없음 확인). 채널은 서로
   독립이고 멱등이다. 404는 성공이다. 배정 해제·스냅숏 실패는 경고만 남기고 삭제한다 (노출 차단 우선).
   지우기 전 판정(라벨링 버킷 접두사, 원본 URI 아님, 워터마크 키 모양, 같은 키를 쓰는 미요청 작업 없음,
   삭제를 지원하는 저장소, 도구 클라이언트 있음, 도구 작업의 주인)이 하나라도 맞지 않으면 지우지 않고
   대기로 남긴다. 주인 대조: 외부 ID 숫자만으로 지우지 않는다. CVAT 작업 이름
   `<세션>/<스트림>/<담당자>`(`clients.cvat_task_name`)과 Label Studio data의 `session_id`·`assignee`가
   작업 행과 같아야 한다. 도구 인스턴스를 새로 만들고 DB를 남기면 같은 번호가 다른 세션·라벨러의 진행
   중 작업일 수 있고, 지우면 되돌릴 수 없다. 다르면 대기로 남기고 운영자가 `--mark-tool-gone`으로
   정리한다. 워터마크 키의 해시 글자 수는 `tasks.MEDIA_HASH_CHARS` 하나로 키 생성과 키 모양 판정이
   함께 쓴다 (한쪽만 바뀌면 새 사본이 영구 대기가 된다).
   실행기는 세션 잠금을 잡지 않고, 도구 단계만 작업 행을 `FOR UPDATE SKIP LOCKED`로 잠근다.
   스냅숏(`sessions/<세션>/revoked/<도구>-<외부 ID>.json`, 라벨링 버킷, `review/` 밖)에 진행 중이던
   주석을 남긴다. 자동으로 이어받지는 않는다. 원본 감사 저장소에는 삭제가 없다 (`DeletableStore`).
6. **게이트와 종료 코드.** 회수가 덜 끝난 스트림에는 새 라벨링 작업을 만들지 않고
   (`create_labeling_tasks`가 세션 잠금 뒤 `TaskError`, 대상 스트림만 본다), CLI `dlp review
   create|assign`은 실행기 뒤 세션에 대기가 하나라도 남으면 대상 스트림과 무관하게 멈춘다 (도구·S3
   장애는 대개 세션의 모든 스트림에 함께 걸려 스트림별로 나눠도 얻는 것이 적다). `dlp review verify`는 이유를
   보인다(회수 미완료, 요청 없는 낡은 작업, 회수돼 다시 만들 배정, 배정 없이 만든 열린 작업이 회수된 뒤
   같은 담당자·스트림·도구로 다시 만들지 않은 것). 마지막 것은 회수 전에는 그 열린 작업이 완료를
   막았기 때문이다. 수집 결과의 회수·다시 만들 명령은 `dlp review collect`와 웹훅 서버(`serve`, 서버
   로그)가 같이 출력한다. CLI는 대기가 남으면 종료 코드 4
   (`REVOCATION_PENDING_EXIT`, argparse 사용 오류 2와 다르다). 웹훅 서버는 커밋 뒤 대기가 남으면 503.
   재시도 경로: 수집 재실행·웹훅·create·assign 시작·`dlp review revoke <세션>`·`--pending`(cron).
   `create_labeling_tasks`가 세션을 먼저 잠가 요청과 생성이 겹치지 않는다 (직접 생성 경로의 경합도
   함께 고친다).
7. **회수한 작업.** 회수한 작업은 수집하지 않는다 (`collect_task` None, `mark_review_task_collected`
   거부). 배정 마무리는 회수한 작업을 세지 않고, `lock_assignment`는 회수한 작업만 남은 배정을 "작업
   없음"으로 봐 `dlp review assign`이 지금 블러본으로 다시 만든다. 웹훅 서버는 열린 작업이 아니면 검수자
   확인 전에 무시한다 (회수 실행기의 CVAT 배정 해제가 `update:job` 웹훅을 일으킨다).
8. **블러본 파일은 남긴다** (ADR 0024 유지). 블러본은 라벨러용으로 서명되지 않고, 렌더 기록 무효와 해시
   확인이 막는다. 지우면 현재 렌더를 지울 위험만 생긴다.
9. **승인 게이트는 두지 않는다.** `dlp privacy approve`를 회수 완료에 묶으면 노출을 줄이지 못하면서
   프라이버시 승인이 검수 도구 가용성에 묶인다. 대신 새 작업 생성과 검수 완료 판정을 막는다.

## 운영 절차

- 배포: `dlp review serve`를 멈춘다(구 코드는 status='revoked' 행을 읽으면 실패한다) → `make db-upgrade`
  (0015) → 코드 배포·serve 재기동 → `dlp review revoke <세션> --dry-run --include-legacy`로 구 작업을
  확인하고, 승인 뒤 블러가 바뀐 이력이 있는 세션부터 `dlp review revoke <세션> --include-legacy`를
  돌린다 → cron(또는 Prefect)으로 `dlp review revoke --pending`을 10분마다 돌려 종료 코드 4를 경보로 쓴다.
- 회수 실행이 스스로 만드는 설정(`revoke`, `dlp privacy detect|render` 뒤)은 검수 정책 파일
  (`DLP_REVIEW_POLICY_FILE`)을 읽지 않는다. 실행기는 라벨링 버킷·원본 버킷 이름·도구 클라이언트만 쓰므로
  그 파일이 없거나 깨져도 매체 삭제와 cron 백스톱이 멈추지 않는다. 작업 생성·수집 명령은 그 파일을
  엄격히 읽어 실패한다.
- 재생성: 수집 출력의 회수 내용 확인 → 남은 블러 검수 → `dlp privacy approve` → `dlp privacy render` →
  출력된 배정마다 `dlp review assign <배정>`, 직접 작업은 `dlp review create <세션> --stage labeling
  --assignee <담당자> --stream <스트림> --tool <도구>` → 검수·수집 → 배정 완료 → `dlp review verify`.
- 사고 대응(라벨러가 노출을 신고): `dlp review revoke <세션> --stream <스트림> --all --reason incident`.
  작업을 알면 `--task <작업 키> --reason incident` (그 세션의 라벨링 작업이어야 한다).
- 도구 인스턴스 폐기 등으로 도구 작업을 확인할 수 없으면(실행기가 "주인이 다르다"고 대기로 남긴 경우
  포함) 원래 작업이 없어진 것을 확인한 뒤 `dlp review revoke --mark-tool-gone <작업> --note <사유>`
  (행위자·사유가 남는다. 미리보기가 없어 `--dry-run`과 함께 쓰지 않는다).
- 미리보기: `dlp review revoke <세션> --dry-run`(고를 작업·이유), `dlp review revoke --pending
  --dry-run`(모든 세션의 대기와 남은 채널). 어느 것도 쓰지 않는다.
- 고아 정리(F-REV-15, 후속): 작업 생성 트랜잭션이 롤백되면 CVAT·Label Studio 작업과 워터마크 사본이
  DB 기록 없이 남는다. `dlp review revoke --orphans [<세션>]`이 도구 프로젝트·라벨링 버킷을 나열해
  `review_tasks`와 대조하고 보고만 한다(고아가 남으면 종료 코드 4). 확인한 뒤 `--orphans --apply`로
  지운다. 대상은 이 플랫폼 이름 규칙(프로젝트 `dlp-privacy`·`dlp-labeling-spatial`·
  `dlp-labeling-temporal`과 `-v<버전>-<해시 12자>`, CVAT 작업 이름 `<세션>/<스트림>/<담당자>`, LS
  data의 `session_id`·`assignee`, 사본 키 `sessions/<세션>/review/<담당자>/<스트림>[-<해시>].mp4`)에
  맞고 이 DB에 있는 세션의 것뿐이다. 판정: 행 없음(`no_db_row`), 같은 키의 행이 다른 작업의 것
  (`other_owner`), 회수 완료 표시가 있는데 남음(`revoked`). 회수 대기(실행기 몫)·살아 있는 행이
  쓰는 것은 남긴다. 생성 직후 경합은 나이 하한(`review.yaml orphans.min_age_s`, 기본 3600초)과 세션 행 SKIP
  LOCKED 잠금(생성 중이면 건너뜀) 뒤 DB 재확인으로 피한다. 지우는 순서와 스냅숏(`revoked/orphan-
  <도구>-<ID>.json`)은 실행기와 같고 DB에는 쓰지 않는다 (`dlp_review.orphans`). 주기 실행은 운영
  결정이다 (보고만 cron으로 돌려 종료 코드 4를 경보로 쓰고, 지우기는 사람이 확인한 뒤 실행 권장).
- 0015 downgrade는 회수한 작업이나 회수 대기가 있으면 거부한다 (회수 의도를 잃으면 노출이 잊힌다).

## 한계 (범위 밖)

- 이미 내려받은 사본, 브라우저 캐시, 저장해 둔 서명 URL로 받은 파일, CVAT 작업 내보내기·백업은 회수할
  수 없다. worker 역할에 내보내기 권한을 주지 않는 운영 점검으로 다룬다.
- 라벨링 버킷 버전 관리가 켜져 있으면 지운 객체의 이전 버전이 남는다 (끄거나 비현재 버전 정리).
- 라벨러 자격 증명을 사람에게 나눠 주면 라벨링 버킷 전체를 읽을 수 있다. 서버만 쥔다.
- DB 롤백으로 기록 없이 남은 고아 도구 작업·객체(R4)는 DB 기준 회수가 찾지 못한다. 후속으로
  `dlp review revoke --orphans`가 이름 규칙과 DB를 대조해 찾는다 (F-REV-15, 위 운영 절차). 이 DB에
  없는 세션의 도구 작업(다른 배포)과 이름 규칙 밖의 것은 보지 않는다.
- 커밋과 첫 외부 호출 사이에 죽거나 도구·S3가 장애면 재시도 전까지 기존 노출이 남는다.
- 사용 중지 세션의 블러 검수 작업(원본 프록시를 올린 CVAT 작업)은 회수하지 않는다 (실행기·계약
  검증기가 블러 검수 작업을 회수 대상에서 뺀다). 원본 접근 권한자만 보지만 동의 철회 뒤에도 남는다.
- 진행 중 주석의 자동 이어받기, 승인 보류 명령, 회수 대기 운영 경고, 서명 URL 기간(F-REV-08)은
  후속 과제다.
