# 2026-10-15 외부 검수(Codex) 재현

[인수인계 첫 화면](README.md) · [알려진 문제 레지스트리](issues.yaml) · [이전 최종 검수](full-review-2026-10-12.md)

1차 기준: 커밋 `d24f3e9`. 외부 검수자(Codex)가 재현한 결함 5건, 테스트 환경 견고성 지적 1건, 리팩토링 권고 1건이다.
지적마다 코드를 읽어 확인하고, 고치기 전에 실패하는 회귀 테스트(정답을 아는 합성 데이터, SQLite·로컬 저장소·
stub)를 먼저 두었다. 5건 모두 유효했다. 상태·관련 파일은 레지스트리가 기준이다.
2차 재검수(커밋 `1ebdb0a`)는 [아래 절](#2차-재검수-1ebdb0a)에 있다. 요약 표는 1·2차를 합친 건수다.

## 요약

| 심각도 | 건수 | 대표 항목 |
| --- | --- | --- |
| 상 | 0 | |
| 중 | 9 | 1차: 보류 중 저장소 실패가 DB 되돌림(F-PMS-14), 회수 실행기 경합(F-REV-29), draft 온톨로지 덧붙이기 뒤 AI 재판정 없음(F-AIR-32), 양손 부분 삭제 뒤 설명 승인(F-PAR-16), 번역 중 들어온 설명의 대기 누락(F-PAR-17). 2차: 직접 검수를 물려받음으로 추정(F-PAR-20), 재생성 요청의 의미 필드 누락(F-PAR-21), 사람 설명 번역 삭제(F-PAR-22), 늦은 보류 무효화가 새 렌더를 덮음(F-PMS-15) |
| 하 | 5 | 1차: 테스트 환경(AWS 프로필·MongoDB) 의존(F-COE-14), 큰 모듈 분리 권고(F-COE-15, 미착수). 2차: 보류 전 저장소 클라이언트 생성(F-PMS-16), FiftyOne 건너뛰기 범위(F-COE-16), 재생성·번역 최종 쓰기 재확인(F-PAR-23) |

## 중 등급

- **F-PMS-14** `dlp privacy hold`가 렌더 기록 무효화(라벨링 버킷 쓰기)를 보류 트랜잭션 안에서 했다. 저장소
  쓰기가 실패하면 예외가 DB를 되돌려 승인이 그대로(APPROVED)이고 보류·회수 요청이 없어 렌더 게이트가
  통과했다.
  - 고친 방법: `hold_session`은 DB만 다룬다(보류 기록·승인 해제·회수 요청). CLI는 그것을 먼저 커밋하고,
    `invalidate_hold_renders`로 무효화를 따로 한다. 실패하면 알리고 종료 코드 5로 끝난다. 같은 명령을 다시
    실행하면 기록은 더하지 않고 무효화·회수만 다시 한다. 무효화 전에도 승인 해제로 렌더 게이트가 거부한다.
  - 회귀: `packages/cli/tests/test_revoke_cmds.py::test_privacy_hold_commits_db_even_when_render_invalidation_fails`.
- **F-REV-29** 회수 실행기 A가 "같은 사본을 쓰는 살아 있는 작업이 없다"를 잠금 밖에서 확인한 뒤 멈춘 사이,
  실행기 B가 같은 회수를 끝내 생성 게이트가 열리고 같은 렌더 해시(같은 매체 키)로 작업을 다시 만들면,
  A가 이어서 새 사본을 지웠다.
  - 고친 방법: 매체 삭제를 작업 생성과 같은 세션 잠금 아래로 옮겼다. 잠금 아래에서 다시 확인하고 지운 뒤
    표시한다. 다른 실행기가 이미 표시했으면 건너뛴다.
  - 회귀: `packages/review/tests/test_revoke.py::test_concurrent_runner_never_deletes_media_of_a_recreated_task`
    (두 순서. 행 잠금이 없는 SQLite라 잠금 시점을 대역으로 흉내 낸다).
- **F-AIR-32** 같은 버전의 draft 온톨로지에 덧붙이면 AI 1차 검수 프롬프트·응답 스키마는 바뀌는데 단위 키는
  같았다. 그래서 재실행이 nothing_to_do였고 이전 판정을 다시 썼다.
  - 고친 방법: 단위 키에 온톨로지 내용 해시를 더했다. 이 해시는 검수 화면 프로젝트 이름과 같은
    `ontology_digest`다. 단위 기록에 열을 두었다(Alembic 0020, 계약 `AiReviewUnit.ontology_digest`).
    표시 프로필과 `--pending` 범위는 지금 해시로 판정한 단위만 쓴다.
  - 대가: 모든 단위 키가 바뀌므로 한 번 다시 판정한다(ADR 0040 4절).
  - 회귀: `packages/aireview/tests/test_runner.py::test_draft_ontology_addition_in_same_version_judges_again`,
    `packages/schema/tests/test_aireview_contracts.py::test_valid_verdicts_display_needs_current_ontology_digest`.
- **F-PAR-16** 양손 행동(같은 action_id의 두 손 레코드)에서 검수자가 왼손을 지우고 오른손을 승인하면 검수 전
  설명이 승인을 물려받았다. 설명 따르기 규칙이 현재 주인만 봤기 때문이다.
  - 고친 방법: 설명의 비교 기준(그 설명을 만든 단계 실행의 주인 레코드)과 현재 후손을 대조한다. 기준 레코드
    중 현재 후손이 없는 것이 있으면 의미 변경으로 보고 검수 전 설명(한국어·영어)을 지운다(ADR 0043 3절).
  - 회귀: `packages/schema/tests/test_description_follow.py::test_bimanual_partial_deletion_retracts_description`,
    `::test_baseline_is_the_stage_run_that_made_the_description`.
- **F-PAR-17** 번역하는 동안(잠금 전) 새로 커밋된 한국어 설명은 이번 실행이 번역하지 않는데,
  `summary.pending`이 비어 `dlp translate run`이 0으로 끝났다.
  - 고친 방법: 쓰고 물려받은 뒤 실제 대기를 다시 계산해 요약과 종료 코드의 근거로 삼는다.
  - 회귀: `packages/actions/tests/test_translate.py::test_description_added_during_translation_stays_pending`.

## 하 등급

- **F-COE-14** 테스트 환경 견고성.
  - S3 Stubber 테스트가 셸의 `AWS_PROFILE`이 없는 프로필을 가리키면 `ProfileNotFound`로 실패했다. 저장소 루트
    `conftest.py`가 모든 테스트에서 AWS 프로필·설정 파일 변수를 비운다. 저장소 코드는 끝점·키를 늘 명시한다.
  - FiftyOne 왕복 테스트는 내장·외부 MongoDB에 연결하지 못하면 이제 건너뛴다. 드라이버 왕복 테스트는 이미
    URI가 없으면 건너뛰었다.
  - 확인: `AWS_PROFILE=nope pytest packages/media/tests/test_storage_checks.py`,
    `FIFTYONE_DATABASE_URI=<닫힌 포트> pytest packages/active/tests/test_curation.py`.
- **F-COE-15** (미착수, 리팩토링 권고) 큰 모듈을 나눈다: `dlp_aireview.runner`(약 1,265줄),
  `dlp_cli.review_cmds`(약 1,255줄), `dlp_review.tasks`(약 1,000줄). 동작 변경 없이 책임별로 나눈다. 의존 표
  (`architecture.md`)와 `_` 이름 경계를 지킨다.

## 함께 반영한 사용자 결정

사람이 의미를 고친 행동의 설명 재생성(ADR 0043 3절 "다시 만들기", 레지스트리 F-PAR-18)을 함께 반영했다.
`dlp actions describe <세션> | --pending`이 행동 VLM에 고친 동사·대상으로 묶은 설명만 다시 묻는다. 새 설명은
고친 행동의 검수를 물려받는다. 출처는 `describe-` 모델 버전으로 남는다.

## 2차 재검수 1ebdb0a

기준: 커밋 `1ebdb0a`. 외부 검수자(Codex)가 다시 검수해 중 등급 4건과 견고성 지적("한계" 절) 3건을 재현했다.
책임 개발자가 모두 유효로 판단했다. 1차와 같이 고치기 전에 실패하는 회귀 테스트(정답을 아는 합성 데이터)를
먼저 두었다. 중 등급 4건은 고치기 전 코드(`1ebdb0a`)에서 회귀 테스트가 실패하는 것을 확인했다.

### 중 등급

- **F-PAR-20** (Codex #2) 설명의 "물려받은 검수만 있음"을 검수자·시각 일치로 추정했다. 검수자가 행동과 모델
  설명을 같은 배치에서 직접 승인하면(같은 검수자·시각) 물려받음으로 보아, 다음 회차에 행동의 의미를 고치면
  직접 검수한 설명이 지워졌다. 같은 행동 레코드를 다시 승인하면(시각 바뀜) 물려받음을 잃는 한계도 있었다
  (F-PAR-19).
  - 고친 방법 (사용자 결정 "추천대로"): 물려받음을 명시 기록으로 남긴다. 추가 전용 표 `description_inheritance`
    (Alembic 0021, 계약 `DescriptionInheritance`)에 물려받는 `record_review`와 같은 트랜잭션으로 쓴다. 판정은
    설명의 마지막 기록이 설명의 지금 검수 상태와 같을 때만 물려받음이다. 기록이 없으면 직접 검수다.
  - 이전 데이터: 0021이 올라갈 때 예전 추정으로 한 번 백필한다(`backfill-heuristic`). 추정의 한계는 이전
    데이터에만 남고 경로 값으로 가려낸다 (ADR 0043 3절 규칙 6 개정).
  - 회귀: `packages/schema/tests/test_description_follow.py::test_direct_review_with_same_reviewer_and_time_is_not_inheritance`,
    `packages/review/tests/test_description_follow.py::test_directly_reviewed_description_with_owner_reviewer_and_time_stays_protected`,
    `::test_reapproving_the_giving_owner_keeps_inheritance_for_a_later_change`,
    `packages/schema/tests/test_description_inheritance_migration.py`.
- **F-PAR-21** (Codex #1) 설명 재생성 요청이 동사·대상·도구만 싣고, 의미 변경으로 보는 방향·대상 부위·전후
  상태를 싣지 않았다. 응답 검사도 값 집합만 봐, 방향만 고친 행동에 예전 설명을 받거나 양손 값을 섞은 응답을
  받았다.
  - 고친 방법: 프롬프트·응답 스키마가 손마다 확정한 의미 조합(`hands`)을 묶고, 응답을 손 단위 조합으로 검사한다.
  - 회귀: `packages/actions/tests/test_describe.py::test_direction_correction_rejects_an_answer_for_the_old_direction`,
    `::test_bimanual_answer_is_checked_per_hand_combination`.
- **F-PAR-22** (Codex #3) 사람이 쓴 한국어 설명은 모델 버전이 없어 비교 기준을 찾지 못하고 처음 모델 행동과
  비교했다. 그래서 사람이 고친 행동(grasp → lift)에 맞춰 쓴 설명의 맞는 영어 번역이 바로 지워지고 다시 생기지
  않았다.
  - 고친 방법: 사람이 쓴 설명의 기준은 쓴 시각에 현재였던 주인 레코드다.
  - 회귀: `packages/actions/tests/test_translate.py::test_translation_of_human_description_for_corrected_action_is_kept`.
- **F-PMS-15** (Codex #4) 보류 A의 렌더 무효화(커밋 뒤 단계)가 늦어지는 사이 A가 해제·재승인·재렌더되면, 늦게
  도는 A의 무효화가 새 렌더를 무효로 덮어썼다.
  - 고친 방법: 무효화는 별도 트랜잭션에서 같은 세션 잠금을 잡고 A가 아직 지금 유효한 보류인지 확인한 뒤에만
    쓴다. 보류 DB 커밋이 먼저인 순서(F-PMS-14)는 그대로다.
  - 회귀: `packages/privacy/tests/test_hold.py::test_delayed_invalidation_of_an_old_hold_keeps_a_newer_render`.

### 하 등급 (견고성)

- **F-PMS-16** `dlp privacy hold`가 보류 DB 커밋 전에 저장소 클라이언트를 만들었다. 잘못된 AWS 프로필·설정이면
  보류가 기록되지 않았다. 이제 커밋 뒤에 만들고, 실패는 무효화 실패(종료 코드 5)로 알린다.
  회귀: `packages/cli/tests/test_revoke_cmds.py::test_privacy_hold_is_recorded_even_when_the_store_client_cannot_be_created`.
- **F-COE-16** FiftyOne 왕복 테스트가 모든 예외를 건너뛰었다. 이제 미설치와 연결 예외만 건너뛰고, 의존성
  비호환 등은 실패한다. 회귀: `packages/active/tests/test_curation.py::test_only_connection_errors_skip_the_round_trip`.
- **F-PAR-23** `dlp actions describe`·번역의 최종 쓰기가 사용 중지·승인 보류·렌더 변경을 다시 보지 않았다. 이제
  쓰기 직전 세션 잠금 아래 다시 보고(`dlp_actions.write_check`), 막히면 아무것도 쓰지 않는다.
  회귀: `packages/actions/tests/test_describe.py::test_final_write_rechecks_hold_withdrawal_and_render_under_the_lock`,
  `packages/actions/tests/test_translate.py::test_final_write_rechecks_hold_and_withdrawal_under_the_lock`.
