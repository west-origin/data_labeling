# 불변 조건과 강제 수단

[인수인계 첫 화면](README.md) · [인수 기준](acceptance.md) · [알려진 문제](issues.yaml)

[CLAUDE.md](../../CLAUDE.md) "규칙" 절의 조건마다 무엇이 실제로 막는지 적었다. 리팩토링이 "강제 수단"을 지우거나
우회하면 그 조건은 문서로만 남는다. **문서뿐**으로 표시된 조건은 사람이 검수에서 봐야 하는 곳이다.
강제 수단이 바뀌면 이 표도 함께 고친다.

| 불변 조건 | 강제 수단 | 비고 |
| --- | --- | --- |
| 라벨은 덮어쓰지 않는다 (수정 = 새 레코드 + parent_label_id) | DB 트리거: `label_records` UPDATE·DELETE 금지(검수 필드 제외) — `packages/schema/tests/test_db.py::test_labels_are_immutable_except_review` | 검수 상태 갱신만 허용 |
| 운영·감사 기록은 추가만 | DB 트리거(추가 전용·TRUNCATE 금지) — `test_db.py::test_append_only_tables_cannot_be_truncated` | 일부 표의 TRUNCATE 검사가 약함 (F-SCH-05) |
| 검수자가 승인·표본 검증한 라벨은 어떤 단계도 지우지 않는다 | 저장 검증 `ProtectedLabelRetractionError`(지우는 레코드 자체, 그리고 조상 사슬의 보호 라벨이 마지막 현재 후손을 잃는 경우) — `packages/schema/tests/test_protected_retraction.py`, 각 단계 `retractions()`는 검수 전만 | 단계의 지울 후보 선택은 바로 위 부모만 본다(프리라벨 착용자 사본). 더 깊은 조상은 저장이 거부해 단계가 실패한다 |
| 라벨 사전·세션 버전·구간 참조가 유효해야 저장 | `insert_labels` → `label_validation`, 0014 INSERT 버전 트리거 — `test_ontology_boundaries.py`(트리거는 SQLSTATE 23514·트리거 함수로 확인하고 트리거를 지운 대조와 비교), 양손 행동 노드 `test_label_graph.py` | |
| 다른 단계 입력은 `current_labels()` (오류 삽입·측정 레코드 제외) | 단위 테스트(episode·review ops), 학습 추출·내보내기 테스트 | 호출부가 `get_labels`를 직접 쓰는지는 **문서뿐** |
| DB 스키마는 tables.py와 Alembic을 함께 바꾼다 | `test_db.py::test_migrations_match_table_definitions` | |
| 계약 변경은 ADR + 마이그레이션 + `make schemas` | `make contracts`(`dlp schema export --check`) | ADR 동반은 **문서뿐** |
| 모든 datetime은 시간대가 있어야 한다 | 계약 타입 검증, `set_lifecycle` 사전 검사 — `test_db.py::test_set_lifecycle_rejects_naive_time_before_update` | 다른 저장 함수는 미검사 (F-SCH-12) |
| 영상 시각은 PTS 인덱스로만 (프레임 번호 × 간격 금지) | PTS 인덱스 API, VFR 픽스처 테스트(media·privacy·export) | 새 코드의 금지 계산은 **문서뿐** |
| 원본 버킷 URI를 일반 라벨러 경로에 노출하지 않는다 | `check_stage_uris` — `packages/review/tests/test_units.py::test_stage_uri_guard`, 내보내기 `assert_no_raw` — `packages/export/tests/test_formats.py::test_raw_uri_guard` | 블러 변경 뒤 배포된 작업 회수: outbox 요청 + 실행기 — `packages/review/tests/test_revoke.py`, `packages/review/tests/test_flow.py::test_blur_change_revokes_deployed_labeling_media_end_to_end`, 생성 경합 `tests/test_review_regressions.py::test_direct_labeling_creation_and_blur_collect_never_miss_revocation` (ADR 0039). 이미 내려받은 사본은 회수할 수 없다 |
| 원본 저장소는 감사 저장소(`raw_store`)로만 | 정적 검사 `tests/test_raw_access_audited.py`, `tests/test_raw_access_db.py` | |
| 블러 검수는 원본 접근 권한자에게만 | `check_privacy_reviewers`, CVAT 계정 연결 검사 — review 단위·흐름 테스트 | 정책 파일 선택이 작업 생성에 안 닿음 (F-REV-01) |
| 정책 값(비율·허용 오차·임계값)은 `config/`에서 | 정책 모델(`Contract`, 알 수 없는 키 금지) | 하드코딩 여부는 **문서뿐** (A-18, F-ETA-10, F-COE-11) |
| 골든·학습·검증 사이 작업자·장소 겹침 금지, 분할은 splitter로만 | `packages/datasets/tests/test_splits.py` (격리·누수 보고) | splitter 우회 여부는 **문서뿐** |
| 학습 예제는 학습·검증 분할에서만, 후보 예측은 DB에 안 씀 | `packages/training/tests/test_extract.py`, 학습 루프 테스트 | 철회 재검사 없음 (T3) |
| 배포는 게이트 통과 모델만 | CLI approve가 PASSED만, 게이트 테스트 | 라이브러리 `deploy()` 직접 호출은 미검사 (T4) |
| 내보내기는 데이터셋 버전에서만, 기본은 사람이 검수한 라벨만, 블러 라벨·원본 위치·검수자 ID 금지 | `test_formats.py::test_verification_policy`, 내보내기 정책·`ExportedLabel` 검증기, `test_runner.py` | |
| 내보내기 가명은 내보내기마다 다르다 (HMAC) | `test_formats.py::test_pseudonyms_are_per_export`, `test_ref_replaces_only_whole_session_tokens` | |
| 비상업(`forbidden`) 가중치는 쓰지 않는다 | `make licenses`(위반 시 실패), `tests/test_model_licenses.py` | 하위 탐지기 추적 테스트가 약함 (F-COE-02) |
| stub만 있는 곳에는 `TODO(real-model):` | `make todo-models`(목록) | 누락 검사는 **문서뿐** |
| 새 모델은 `Predictor` 어댑터 + CPU stub | prelabel 어댑터 테스트 | **문서뿐**(구조) |
| 패키지 의존은 architecture.md 표대로, 비공개 이름 import 금지 | `tests/test_import_boundaries.py` | 패키지를 넘는 공개 이름 목록은 고정하지 않음 |
| 검수 화면 테마는 모양만 바꾼다 (라벨 이름·왕복 불변) | `packages/review/tests/test_ui_theme.py`, `make contracts`(생성 파일 일치) | ADR 0038 |
| 코드 주석·문서는 한국어 | 없음 | **문서뿐** (F-SCH-13) |
| 실제 영상·개인정보를 저장소에 넣지 않는다 | 없음 (`data/`는 Git 제외) | **문서뿐** |
