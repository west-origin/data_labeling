# ADR 0042: 사고 대응 중 프라이버시 승인을 보류한다

- 상태: 채택
- 날짜: 2026-10-11
- 관련: WP5(프라이버시), ADR 0024(블러본은 지금 승인된 블러로 렌더한 것만), ADR 0039(배포 매체
  회수), F-REV-18.
  계약 변경: `dlp_schema.session.PrivacyHoldEvent`·`HoldAction`, `schemas/privacy_hold_event.schema.json`.
  DB 변경: `privacy_holds` 테이블(추가 전용, Alembic 0017, down_revision 0015).

## 배경

라벨러가 노출(블러되지 않은 얼굴 등)을 신고해도, 블러 재검수를 수집해 블러 라벨이 바뀌기 전까지는
승인(APPROVED)과 렌더 기록이 그대로다. 그 사이 `dlp review create|assign`은 같은 렌더로 새 라벨링
작업을 만들 수 있었고, 블러본을 여는 다른 소비자(내보내기, 행동 VLM, 큐레이션)도 막히지 않았다.
`dlp review revoke --all`은 이미 배포한 작업만 끊고 새 배포는 막지 못한다. 승인을 되돌리는 명령이
없어 운영자가 DB를 직접 고쳐야 했다.

## 결정

1. **보류 명령.** `dlp privacy hold <세션> --reason <사유> [--actor]`(`dlp_privacy.hold.hold_session`)는
   한 트랜잭션(세션 행 잠금)에서
   - 보류 기록을 추가하고(`privacy_holds`, 이미 보류 중이면 더하지 않는다),
   - 승인 상태면 AUTO_BLURRED로 되돌리고(기존 `set_privacy_state`),
   - 모든 영상 스트림의 배포된 라벨링 작업에 해시와 무관하게 회수를 요청하고(기존
     `request_media_revocation(force=True)`, 사유 `hold:<행위자>:<사유>`, 블러 검수 작업 제외),
   - 커밋 **뒤** 렌더 기록을 무효로 둔다(`invalidate_hold_renders` → 기존 `invalidate_render`).
   2026-10-15 수정(F-PMS-14): 처음에는 무효화(라벨링 버킷 쓰기)도 같은 트랜잭션 안이라 저장소 쓰기가
   실패하면 보류·승인 해제·회수 요청까지 되돌아갔다. 이제 DB 단계는 저장소와 무관하게 먼저 커밋하고,
   무효화는 재시도 가능한 다음 단계다. 무효화가 실패하면 CLI가 알리고 종료 코드 5로 끝난다. 같은
   명령을 다시 실행하면 보류 기록은 더하지 않고 무효화·회수만 다시 한다 (멱등). 그 사이에도 승인
   상태가 AUTO_BLURRED라 렌더 게이트(`assert_render_current`)가 거부한다.
   2026-10-15 수정(F-PMS-15·16): 무효화는 별도 트랜잭션에서 같은 세션 잠금(렌더·승인·해제와 같은 잠금)을
   잡고 그 보류가 아직 지금 유효한 보류(마지막 기록)일 때만 쓴다. 늦게 도는 보류 A의 무효화가 A 해제 →
   재승인 → 재렌더 뒤의 새 렌더를 무효로 덮어쓰지 않는다 (`superseded`로 알리고 건너뜀). 라벨링 저장소
   클라이언트도 DB 단계 커밋 뒤에 만든다: 잘못된 AWS 프로필·설정으로 클라이언트를 만들지 못해도 보류는
   기록되고 무효화 실패(종료 코드 5)로 알린다.
   커밋 뒤 CLI가 회수 실행기를 돌린다 (대기가 남으면 종료 코드 4, ADR 0039). 2026-10-15 수정(F-PMS-17):
   회수 실행기가 저장소·도구 클라이언트를 만들지 못해도(잘못된 AWS 프로필 등) 예외로 끝나지 않는다. 커밋된
   보류 결과를 먼저 알리고, 회수 요청은 대기로 남기며, 재시도 명령과 함께 종료 코드 5로 끝난다. 승인 상태를 요구하는
   모든 소비자(`assert_render_current`·`expected_render_hash`, `render_session`, 라벨링 작업 생성,
   내보내기·행동 VLM)가 그대로 막힌다. 블러 검수 작업 생성·수집은 막지 않는다 (보류 중에 블러를
   고친다).
2. **보류 중 승인 거부.** `approve_session`은 세션의 마지막 보류 기록이 hold면 거부한다. 라벨링 작업
   생성(`create_labeling_tasks`)도 보류 이유와 해제 방법을 담아 도구 호출 전에 거부한다.
3. **해제는 재승인 절차를 거친다.** `dlp privacy hold <세션> --release --reason`
   (`release_hold`)은 해제 기록만 남긴다. 승인을 돌려놓지 않으므로 (필요하면 블러 검수 → 수집) →
   `dlp privacy approve`(승인 조건 확인) → `dlp privacy render` → 다시 만들기를 거친다. 보류 중이
   아니면 해제를 거부한다.
4. **기록은 추가만.** `privacy_holds(event_id 일련번호, session_id, action hold|release, reason,
   actor, at)`. 수정·삭제·TRUNCATE는 0009·0010의 공용 트리거 함수로 막는다 (PostgreSQL). 보류
   여부는 일련번호 순 마지막 기록으로 정한다 (시계 역행 무관). 0017 downgrade는 기록이 하나라도
   있으면 거부한다 (보류 중 승인 거부와 사고 대응 감사 기록을 잃는다). 라벨은 쓰지 않는다.
5. **새 상태 값을 두지 않는다.** `PrivacyState`에 HELD를 더하면 승인 상태를 보는 모든 소비자와
   계약·스키마가 바뀐다. 기존 AUTO_BLURRED가 이미 "승인 전"을 뜻해 모든 소비자를 막으므로, 보류는
   별도 기록으로 "재승인 금지"만 더한다.

- 버린 안: 보류 해제 때 자동으로 승인(APPROVED)으로 돌려놓기. 블러를 고치지 않은 채 같은 렌더가
  다시 열린다. 승인 조건 확인을 다시 거치게 한다.
- 버린 안: 해제 조건으로 "보류 뒤 수집한 블러 검수"를 강제하기. 오신고까지 전 스트림 전수 재검수를
  요구하게 되어 운영 결정이 필요하다 (아래 남은 일).

## 운영 절차

1. 노출 신고: `dlp privacy hold <세션> --reason "<신고 내용>"` → 출력의 회수 결과 확인 (4면
   `dlp review revoke <세션>`을 다시 실행하거나 cron `--pending`을 기다린다).
2. 블러 검수: `dlp review create <세션> --stage privacy --assignee <권한자>` → 검수 → `dlp review
   collect`.
3. 해제와 재승인: `dlp privacy hold <세션> --release --reason "<조치 내용>"` → `dlp privacy approve`
   → `dlp privacy render` → `dlp review verify <세션>`이 보이는 다시 만들 배정·직접 작업 명령.

## 한계

- 보류는 이 플랫폼 경로만 막는다. 이미 내려받은 사본·도구 내보내기·라벨러 자격 증명은 ADR 0039
  한계와 같다 (운영 점검, F-REV-19).
- 해제 조건(보류 뒤 블러 재검수 의무 여부)은 운영 결정으로 남는다.
