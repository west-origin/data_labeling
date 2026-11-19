# ADR 0071: 블러 검수용으로 CVAT에 올린 원본 사본을 사용 중지·보관 삭제 결정 때 지운다

- 상태: 채택 (구현됨, SQLite와 가짜 CVAT 클라이언트로 검사. 실제 CVAT 삭제는 ADR 0039 실행기와 같은 호출이다)
- 날짜: 2026-11-19
- 관련: ADR 0039(블러 변경 뒤 배포 매체 회수, outbox와 실행기), 0020·0021(원본 접근 감사·보관), 0007(사용 중지)
- 문제: S2 (`docs/handover/known-issues.md`, CVAT 원본 사본 자동 삭제 없음)
- 계약 변경: `ReviewTask` 검증기만 바뀐다. 블러 검수 작업도 사유가 `raw-copy:`(`dlp_schema.review.RAW_COPY_REASON`)로
  시작할 때만 회수 필드를 가질 수 있다. DB 열은 그대로다 (Alembic 없음, JSON Schema 변화 없음).

## 배경

블러 검수 작업(`tasks.create_privacy_tasks`)은 원본 버킷의 검수 프록시(블러 전 원본 화소)를 CVAT에 바로 올린다.
CVAT는 영상을 자기 안에 보관하므로 작업을 지우기 전까지 원본 사본이 도구에 남는다. ADR 0039의 회수는 블러가 바뀐
일반 라벨러 매체만 다루고 블러 검수 작업은 일부러 뺐다 (계약 검증기와 저장소 SQL 양쪽). 그래서 세션을 사용
중지하거나(동의 철회) 원본 보관을 끝내기로 해도 CVAT의 원본 사본은 사람이 따로 지워야 했다. 보관 만료 안내
(`RETENTION_NOTE`)는 "운영 절차에서 함께 지운다"고만 적었다.

## 결정

1. **언제 지우나.** 두 경우다. 세션 사용 중지(`dlp dataset withdraw`)와 원본 보관 삭제 결정(`dlp ops
   retention-decide <세션> delete`). 수집이 끝난 블러 검수 작업을 바로 지우지는 않는다. 검수자가 다시 열어 고칠 수
   있어야 하고, 지금 보관 규칙은 기간이 끝날 때 원본을 지우는 것이기 때문이다.
2. **같은 outbox를 쓴다.** 요청은 결정과 같은 트랜잭션에서 `review_tasks`에 남긴다
   (`request_session_raw_copy_purges` → `request_raw_copy_purge`). 사유는 `raw-copy:withdrawn:<행위자>`,
   `raw-copy:retention-delete:<결정자>`다. 블러 변경 회수(`request_task_revocation`)는 여전히 블러 검수 작업을 고르지
   않는다. 계약 검증기도 `raw-copy:`가 아닌 사유를 블러 검수 작업에 허용하지 않는다.
3. **매체 단계는 없다.** 블러 검수 작업은 라벨링 버킷 사본을 만들지 않으므로 요청과 함께 `media_revoked_at`을
   남긴다. 원본 버킷의 원본 프록시는 지우지 않는다. 원본 삭제는 보관 운영 절차의 몫이다.
4. **도구 단계는 ADR 0039 실행기 그대로.** 행 잠금 → 주인 대조 → 배정 해제 → 삭제 → 없음 확인 → 표시. 주인
   대조의 담당자 자리는 작업 이름의 `privacy`다 (고아 정리 `orphans._owns`와 같다). 주인이 다르면 지우지 않고
   대기로 남긴다.
5. **스냅숏은 남기지 않는다.** 원본을 본 검수자의 주석이고, 수집한 블러는 이미 DB에 있다. 사용 중지·삭제 결정은
   사본을 줄이려는 것이다.
6. **실패와 증거.** 실행은 커밋 뒤에 한다. CVAT에 닿지 않으면 대기로 남고 명령은 종료 코드 4로 끝난다.
   `dlp review revoke <세션>`이나 cron `dlp review revoke --pending`이 다시 한다. 증거는 `review_tasks`의 요청 시각·
   사유·도구 삭제 확인 시각이다 (추가로만 채운다).

## 결과

- `dlp dataset withdraw`가 블러 검수 작업도 회수 요청 목록에 넣고(`raw-copy:` 사유) CVAT 작업을 지운다.
- `dlp ops retention-decide <세션> delete`가 그 세션 블러 검수 작업의 CVAT 원본 사본을 지운다. `extend`는 그대로다.
- `dlp ops retention`의 안내는 원본 버킷의 원본만 운영 절차라고 적는다.
- 남은 것: 원본 버킷의 원본 프록시·원본 삭제는 여전히 사람이 하는 운영 절차다. 도구에 기록 없이 남은 작업은 고아
  정리(`dlp review revoke --orphans`)가 찾는다.

## 검증

- `packages/review/tests/test_revoke.py::test_raw_copy_purge_deletes_only_the_privacy_cvat_task_without_a_snapshot`:
  블러 변경 회수는 블러 검수 작업을 고르지 않음, 요청 사유와 매체 완료, CVAT 작업만 삭제(저장소 호출 없음, 스냅숏
  없음), 주인이 다르면 대기, 다시 요청해도 새 요청 없음.
- `packages/datasets/tests/test_withdraw_revocation.py`, `packages/cli/tests/test_revoke_cmds.py`(사용 중지와 보관 삭제
  결정의 요청·실행·종료 코드), `packages/schema/tests/test_review_revocation.py`(계약 검증기).
