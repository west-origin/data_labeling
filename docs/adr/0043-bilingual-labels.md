# ADR 0043: 라벨을 한국어와 영어로 함께 남긴다 (온톨로지 영어 이름, 설명 자동 번역)

- 상태: 채택
- 날짜: 2026-10-13
- 관련: ADR 0002(계약·라벨 저장), ADR 0010(모델 상업 사용), ADR 0012(행동 2단 구조), ADR 0015(검수된
  라벨 보존), ADR 0018·0027(내보내기·가명), ADR 0028(초안 온톨로지 덧붙이기), ADR 0040(사내 VLM, 외부 API
  금지).
  계약 변경: `dlp_schema.ontology.Term.en`(필수)·`TermName`·`Ontology.display_names`,
  `dlp_schema.labels.DescriptionPayload.source_label_id`, `dlp_schema.export.IntervalFile.term_names`(형식
  1.1). `schemas/ontology.schema.json`, `label_record.schema.json`, `export_intervals.schema.json` 등 재생성.
  DB 변경 없음 (페이로드는 JSON 열, 온톨로지 내용은 `ontology_versions.content`).

## 배경

검수자는 CVAT·Label Studio에서 한국어 화면으로 라벨을 고르지만, 저장되는 값은 이미 영어 온톨로지 ID다
(`verb: wipe`). 한국어로만 남는 것은 두 가지였다.

1. 온톨로지 표시 이름: 사전 항목마다 `ko`만 있고 영어 이름이 없다. 내보내기를 받는 쪽은 ID만 본다.
2. 자유 서술 설명 라벨(`DescriptionPayload`): 행동 VLM이 한국어 한 문장으로 쓴다
   (`dlp_actions.assemble`). 영어 데이터가 필요한 구매자에게 줄 영어 문장이 없다.

사용자 요청: "한국어로 라벨링하면 자동으로 영어로도 함께 저장되게" (두 부분 모두).

## 결정

### 1. 온톨로지 영어 이름

- 이름이 있는 모든 사전 항목(`Term`과 하위 타입, 작업의 하위 단계 포함)에 필수 `en`을 `ko` 옆에 둔다.
  둘 다 앞뒤 공백을 지운 뒤 1자 이상이어야 한다 (`DisplayName`). v1 188개 항목의 영어 이름을 채웠다
  (청소·요양·간호 용어: 예 `rag` → "Cleaning cloth", `transfer_assist` → "Transfer assistance (bed <->
  wheelchair)", `semi_fowler` → "Semi-Fowler's position"). ID는 바꾸지 않는다. 라벨에는 계속 ID만 저장한다.
- **버전을 올리지 않는다.** v1은 초안(draft)이고 `en` 추가는 사전마다 키를 더하기만 하는 변경이라 ADR 0028
  덧붙이기 규칙에 맞는다. `dlp ontology register 1.0.0`(`make db-upgrade`)이 기존 등록 내용을 새 내용으로
  바꾼다. 예전 내용으로 기록된 라벨은 그대로 유효하다 (`packages/schema/tests/test_db.py::
  test_adding_english_names_to_registered_draft_is_additive`).
- **배포 순서**: 새 코드는 `en` 없는 등록 내용을 읽지 못한다 (계약상 필수). 그런 행을 읽으면
  `LegacyOntologyContentError`가 "`make db-upgrade`로 다시 등록하라"고 알린다. 코드를 올린 뒤 라벨을 쓰는
  단계보다 먼저 `make db-upgrade`를 돌린다.
- 검수 화면 프로젝트 이름의 사전 해시(`dlp_review.ontology_projects`)가 바뀌어, 다음 작업 생성부터 새
  CVAT·Label Studio 프로젝트가 생긴다 (초안 덧붙이기의 기존 동작, 기존 프로젝트를 덮어쓰지 않는다).
  라벨 값(ID)·왕복 변환·검수 화면 테마(ADR 0038)는 바뀌지 않는다.
- 조회 도우미: `Ontology.display_names()`(`"<범주>:<ID>"` → `TermName(ko, en)`, 범주가 달라 같은 ID가 있어도
  구별, 예 `objects:person`·`physical_types:person`), `Ontology.name(범주, ID, 언어)`.
- 내보내기에 영어 이름을 싣는다. 기존 소비자를 깨지 않도록 기존 필드는 그대로 두고 덧붙인다.
  - COCO: 온톨로지 객체 범주에 `name_ko`·`name_en` (`name`은 ID 그대로). 합성 범주 `hand`에는 없다.
  - 구간 JSON: `term_names`(파일마다 온톨로지 전체 이름표), 형식 버전 1.0 → 1.1 (하위 호환).
  - LeRobot `meta/dlp_vocab.json`: `term_names`(어휘 목록과 같은 순서의 한·영 이름).

### 2. 한국어 설명의 자동 영어 번역

**위치.** 새 패키지 대신 `dlp_actions`에 모듈 셋을 둔다: `translate`(계획·실행), `translate_policy`,
`translators`(번역기 규약·stub·사내 서버 어댑터). 설명 라벨을 만드는 단계가 행동 단계이고, 같은 공통 전송
(`dlp_models.vlm`)을 쓰며, 의존 표(architecture.md)를 바꾸지 않는다 (actions → models·schema는 이미 있다).
CLI는 `dlp_cli.translate_cmds`다.

**데이터 모델.** 번역은 원문의 수정이 아니라 **별도 운영 라벨**이다.

- 번역 레코드 = 모델 출처 `DescriptionPayload(segment_id=원문과 같음, text=영어, language="en",
  source_label_id=원문 라벨 ID)`. 구간·스트림·온톨로지 버전·근거는 원문과 같고, 신뢰도는 원문 값(없으면
  `translate.yaml default_confidence`).
- `parent_label_id`를 쓰지 않는다. parent로 이으면 `current_labels`가 원문을 "고쳐진 이전 레코드"로 보고
  운영 라벨에서 빼고, 오류 삽입·측정 계보 판정(`non_operational_ids`)·보호 조상 검사(F-SCH-02)도 번역을
  원문의 후손으로 다룬다. 그래서 새 페이로드 필드 `source_label_id`로 원문 레코드를 가리킨다. 원문과
  번역이 둘 다 현재 운영 라벨이다 (`test_translate.py::test_each_korean_description_gets_an_english_sibling`).
- 라벨 ID = `derived_id(원문 ID, "en" + version_tag(번역 버전))`: 원문 레코드·버전마다 결정적이다.
- 번역 버전 = `translate-<정책 해시>+<번역기 버전>+g<용어집 해시 8자>`. 정책 해시는 `server` 절을 뺀
  `translate.yaml` 파싱값, 번역기 버전은 `llm-<모델>`(stub은 `stub-glossary-1`), 용어집 해시는 온톨로지
  한·영 이름에서 만든다 (이름이 바뀌면 번역도 다시 만든다).
- 같은 구간에 ko·en 설명 두 개가 있는 것은 저장 검증(`label_validation`·`label_graph`)이 이미 받는다
  (설명은 구간 그래프의 노드가 아니다). 바꾸지 않고 테스트로 고정했다.

**다시 돌릴 때** (원문 S마다, `plan_translation`)

1. S를 가리키는 현재 번역이 이 버전이거나 검수된 것(사람 출처·승인·표본 검증)이면 됐다.
2. 같은 구간에 검수된 영어 설명이 있으면 새 기계 번역을 만들지 않는다. 검수된 라벨과 겹치는 새 모델
   결과를 버리는 행동 단계(ADR 0015)와 같은 규칙이다. 원문이 바뀌어 낡았을 수 있어 `kept_reviewed`로
   알린다 (검수자가 영어도 고친다).
3. 이 버전 번역 ID가 이력에 있으나 현재 라벨이 아니면(검수자가 지움) 되살리지 않는다 (이력으로 멱등).
4. 그 밖은 번역한다. **성공하면** S를 가리키던 다른 버전의 검수 전 번역을 `retractions()`로 지운다.
   실패하면 이전 번역을 남긴다 (번역이 아예 없어지지 않게).
5. 원문이 더는 현재 원문이 아닌(고쳐졌거나 지워진) 검수 전 번역은 지운다. 검수된 번역은 어떤 경우에도
   지우지 않는다 (저장 검증 `ProtectedLabelRetractionError`가 한 번 더 막는다).

행동 단계와의 관계: 행동 재실행(`dlp actions run`)이 검수 전 행동과 그 설명을 지울 때 그 설명의 **검수 전
영어 번역도 같은 배치에서 지운다** (`runner.with_descriptions`, `is_machine_translation`). 검수된 영어 번역은
검수된 설명처럼 그 행동을 보호한다. 행동 단계의 `_ours`(같은 버전 판정·지울 후보)는 `actions-` 접두사만
보므로 번역 레코드를 자기 결과로 세지 않는다.

**잠금.** 이력 읽기·번역(서버 호출)은 세션을 잠그지 않는다. 쓰기 직전에 세션을 잠그고 이력을 다시 읽어
계획을 다시 세운 뒤, 그 사이 원문이 바뀌었거나 검수된 것은 버리고 쓴다 (긴 세션 잠금 회피, R3).

**번역 백엔드.** 사내 OpenAI 호환 서버만 쓴다 (외부 API 금지, ADR 0040). 기본 모델은
`Qwen/Qwen3-VL-8B-Instruct`(services/vlm `vlm-aireview`, `config/models.yaml` 기존 항목 재사용). 글만 보내는
요청이라(이미지 없음) 같은 서버가 맡는다. 행동 분류 모델(Qwen2.5-VL-7B)과 다른 모델이라 자기 설명을 자기가
옮기는 상관도 없다. 요청: system(번역 지침) + user(용어집 + 원문), 응답 JSON Schema `{"text"}`(서버 강제),
temperature 0. 응답 검사: JSON·비지 않음·`max_chars` 이하·한글이 남지 않음(`reject_hangul`). 위반이면 위반
목록을 붙여 `max_retries`번 다시 묻는다. **용어집**: 온톨로지 ko → en 이름(`glossary: matched`면 원문에 나온
이름만)을 넣어 도메인 용어를 일관되게 옮긴다. 주소는 `--translate-url` 또는 `DLP_TRANSLATE_URL`, 키는
`DLP_VLM_API_KEY`. CLI가 보내기 전에 `translate.yaml server.internal_hosts`로 사내 대역을 검사한다
(`dlp_aireview.backend.check_hosts`, 밖이면 종료 5). CPU·CI는 `GlossaryStubTranslator`(정답 번역 또는 용어집
치환 + `[stub-en]` 표시, `TODO(real-model)`).

**정책 파일.** `config/policies/translate.yaml`을 따로 둔다. `actions.yaml`에 두면 그 해시(행동 모델 버전)가
바뀌어 번역 설정만 고쳐도 모든 세션의 행동 구간을 다시 만든다. 프롬프트 틀 문장은 코드에 두고
`prompt_version`(코드 `PROMPT_VERSION`과 같아야 적재)으로 버전에 넣는다.

**실패 처리.** 번역 실패는 호출자를 실패시키지 않는다. 서버 장애(백오프 뒤)·설정 오류(4xx)면 그 세션의
남은 원문을 묻지 않고 대기(`pending`)로 남기고, 응답 위반이면 그 원문만 남긴다. 번역한 것은 쓴다.
`dlp translate pending`이 대기 세션을 보이고 `dlp translate run --pending`이 채운다.

**자동 실행.** `dlp actions run`이 행동 트랜잭션을 **커밋한 뒤** 같은 세션을 번역한다
(`translate_after_actions`, 최선 노력). 번역 서버 미연동·장애·내부 오류는 안내만 출력하고 행동 결과와 종료
코드에 영향이 없다 (`--no-translate`로 끈다). Label Studio·CVAT는 설명 라벨을 다루지 않아(`LS_KINDS`에 없음)
검수 수집 뒤 번역 고리는 두지 않았다. 사람이 DB에 쓰거나 고친 한국어 설명은 `--pending`이 채운다.

**소비자.**

- 내보내기: 설명은 언어마다 한 레코드라 구간 JSON에 ko·en이 모두 나간다 (`intervals.kinds`에 description).
  번역의 `source_label_id`는 원문의 가명 라벨 ID로 바꾼다 (세션 부분만 바꾸는 `ref`가 아니라 `ids.label`).
  내보내기·학습은 계속 `current_labels`로 고른다. 설명 수를 세는 소비자는 언어별로 두 배가 된다는 점을 안다
  (구매자 문서 갱신 대상).
- AI 1차 검수: 설명은 판정 종류(`TEMPORAL_KINDS`)가 아니고 맥락에도 들지 않아 번역이 판정 단위를 늘리지
  않는다 (`packages/aireview/tests/test_plan.py::test_bilingual_descriptions_do_not_add_judged_units`).
- 라이선스: 번역문도 판매 데이터에 들어가므로 번역 모델은 `make licenses`가 검사한다
  (`used_models`의 `translate.backend.model`). Qwen3-VL-8B는 Apache-2.0, 학습 데이터 미공개라 `review`다
  (S4와 같은 법무 확인 대상).

## 결과

- 명령: `dlp translate run <세션> | --pending`(종료 0 완료, 2 주소 없음, 3 대기 남음, 5 설정 오류),
  `dlp translate pending`, `dlp actions run --translate-url | --no-translate`.
- 남은 일: 실제 사내 서버에서 번역 품질 확인(용어집 적중·한글 잔존 비율), 구매자 문서에 형식 1.1·
  `term_names`·설명 언어 표기. 설명 라벨(한·영 모두)은 검수 도구 경로가 없어 모델 설명이 `dlp review
  verify`를 막는 기존 공백이 번역으로 커진다 (F-PAR-15).
