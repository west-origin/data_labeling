# ADR 0020: 운영 지표, 원본 접근 감사, 원본 보관 만료

- 상태: 채택
- 날짜: 2026-10-07
- 관련: WP16. 계약 변경: RawAccessEvent, ReviewWork, PrivacyAuditRecord, RetentionDecision
  (`dlp_schema.ops`, Alembic 0009, `make schemas`)

## 결정

1. **원본 접근은 모두 감사 기록으로 남긴다.** 원본 버킷 저장소는 `dlp_cli.raw_access.raw_store`로만 만들고,
   이것은 `dlp_media.audit.AuditedStore`(내려받기·서명 URL·올리기)를 돌려준다. 기록은 접근 **전에**, 이벤트마다
   따로 커밋한다 (실패한 시도도 남고, 호출한 작업이 되돌아가도 남는다). 블러 검수 작업에 원본을 올려 사람에게
   보여 주면 grant(담당자)로 남긴다. 실행한 사람은 `DLP_ACTOR`(없으면 OS 사용자). 감사 기록과 다른 운영 기록은
   PostgreSQL 트리거로 수정·삭제를 막는다. 원본 저장소를 감사 없이 만드는 코드는 정적 검사 테스트가 막는다.
   원본 버킷에 올리는 수집은 DB가 있어야 한다 (`--no-db`는 로컬 저장소에만).
2. **월간 감사 리포트** (`dlp ops audit-report <월>`): 사람·용도·동작별 집계와 확인할 것 — 원본 권한이 없는
   사람의 열람·서명 URL, 권한 없는 사람에게 보여 준 원본, 담당자 없이 올린 원본 검수 작업, 업무 시간 밖 사람의
   접근. 권한자 = `review.yaml reviewers.privacy` + `ops.yaml audit.raw_viewers`, 서비스 계정은 파이프라인.
3. **주간 운영 지표** (`dlp ops weekly`, ISO 주, UTC): 영상 1시간당 검수 분(작업 라벨·블러 따로), 수정률,
   자동 승인율, 블라인드 프리라벨 편향, 오류 삽입 발견율, 잔여 블러 누락(영상 1시간당), 검증 에피소드 수,
   에피소드당 생산원가(인건비 정책이 있을 때). 각 지표는 그 주에 일어난 일만 센다 (검수 변화의 시각은
   `dlp_schema.history.ReviewChange.at`). 경고: 자동 승인율 상승 + 발견율 하락(검수 품질 저하), 검수 시간·
   수정률 정체(가이드라인·온톨로지 점검). 표(markdown)와 JSON으로 낸다.
4. **검수 시간**은 Label Studio의 주석 lead_time을 수집할 때 기록한다. CVAT는 작업 시간을 재지 않아
   `dlp ops log-work`로 기록한다. **잔여 블러 감사** 결과는 `dlp ops privacy-audit`로 남긴다 (감사자는 원
   블러 검수자와 달라야 한다, 계약 검증).
5. **원본 보관 만료** (`dlp ops retention`): 기간은 `defaults.yaml retention.raw_retention_days`(미정이면
   알림 없음). 기산점은 사람 검증을 마친 세션의 마지막 라벨 변경 시각. 만료 임박·만료·사용 중지 세션을 알리고,
   연장(기한·사유)·삭제 결정을 `dlp ops retention-decide`로 기록한다. 원본 삭제 자체는 결정 뒤 운영 절차로 한다.

## 남은 것

- 생산원가 인건비, 원본 보관 기간, 성공 기준 수치는 Gate 0·1에서 사람이 정한다 (지금은 비어 있음).
- CVAT의 작업 시간 자동 수집 (CVAT 이벤트 로그 연동)은 하지 않았다.
