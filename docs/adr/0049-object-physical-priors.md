# ADR 0049: 온톨로지 객체에 물리 사전값(재질·촉감·질량·마찰)을 둔다

- 상태: 채택
- 날짜: 2026-10-17
- 관련: ADR 0002(계약), ADR 0018·0027(내보내기), ADR 0028(초안 온톨로지 덧붙이기), ADR 0037(온톨로지 검증
  경계), ADR 0040(AI 1차 검수 단위 키, F-AIR-32), ADR 0043(한·영 이름, 용어집). 문제 레지스트리
  F-SCH-14·15·16(남은 일).
  계약 변경: `dlp_schema.ontology`의 `PriorSource`·`PriorRange`·`StatePriorRange`·`PhysicalPriors`,
  `ObjectClass.physical_priors`(선택), `Ontology.materials`·`Ontology.tactile_properties`(기본 빈 사전),
  `Ontology.physics_names`, `PHYSICS_CATEGORIES`. `dlp_schema.export`의 `PhysicalPriorTable`·
  `physical_prior_table`·`PHYSICAL_PRIOR_CONVENTIONS`, `IntervalFile.physical_priors`(형식 1.1 → 1.2).
  초안 덧붙이기 판정(`db.repository._is_additive`)에 `physical_priors` null → 사전 예외.
  `schemas/ontology.schema.json`·`export_intervals.schema.json` 재생성. DB 변경 없음 (온톨로지 내용은
  `ontology_versions.content` JSON 열).

## 배경

사용자 요청 "온톨로지에 물리 성질을 넣자"의 1단계(사용자 승인). 로봇 학습 구매자는 영상 속 물체의 재질,
손이 느끼는 촉감, 대략의 질량·마찰을 알고 싶어 한다(파지력·미끄럼 사전 분포, 시뮬레이터 물체 설정).
지금 데이터에는 물체 클래스 ID와 상태(젖음·채움 등)만 있다. 실제 측정 장비는 없으므로, 먼저 클래스 수준의
**사전값(prior)** 을 온톨로지에 두고 측정값으로 오해되지 않게 계약·내보내기에 명시한다.

## 결정

### 1. 계약 (`dlp_schema.ontology`)

- 새 어휘 범주 `materials`(재질 10개)·`tactile_properties`(촉감 11개)를 `config/ontology/v1/physics.yaml`에
  둔다. 항목은 `Term`이라 `ko`·`en`이 필수다 (ADR 0043).
- `ObjectClass.physical_priors: PhysicalPriors | None` (선택).
  - `source`(필수): `prior_literature`(문헌·사양에서 옮김) | `prior_estimate`(일반 지식 추정). 측정값을 뜻하는
    값은 없다. `note`(선택): 근거·가정.
  - `material`: 대표 재질 ID. `part_materials`: 도구 부분 ID → 재질 (그 객체의 작용부·파지부만, 대표와
    다른 부분만. 예 밀대 `mop_head: fabric`, 쓰레받기 `pan_edge: rubber`).
  - `tactile`: 촉감 ID 목록 (중복 금지). 도구는 파지부, 그 밖은 주로 닿는 면 기준.
  - `mass_kg`, `static_friction`: `StatePriorRange` = `min`·`max`(0보다 큰 유한한 수, min ≤ max) +
    `by_state`(상태 속성 ID → 상태 값 ID → 범위). 기본 범위는 기본 조건(건조, 상태를 따로 적지 않음)이고,
    상태별 범위는 그 객체의 `states`에 있는 속성과 그 속성의 `values`만 쓸 수 있다. 상태별 범위는 기본
    범위 밖이어도 된다 (젖은 바닥의 마찰).
  - 정지 마찰 계수의 상대면은 건조한 맨손 피부·니트릴 장갑·고무 그리퍼 패드를 함께 보는 넓은 범위다.
    젖음은 `by_state.wetness`로 적는다 (상태 속성이 wetness인 객체만).
  - 블록이 비면(재질·촉감·질량·마찰이 모두 없으면) 거부한다.
- 교차 참조 검증은 `Ontology._check_references`가 `_physics_errors`로 모은다 (복잡도 상한 안).
- **재질·촉감 이름은 `display_names`(라벨 어휘 이름표)에 넣지 않는다.** 이 이름표는 번역 용어집이고
  용어집 해시가 번역 버전(`translate-…+g<해시>`)에 들어가, 넣으면 검수 전 영어 번역을 모두 다시 만든다.
  재질·촉감 ID는 라벨에 들어가지 않으므로 용어집 이득이 없다. 이름은 `Ontology.physics_names()`로 따로
  읽는다. AI 1차 검수 프롬프트에도 넣지 않는다 (판정 대상 라벨과 무관).
- 사람(`person`)과 액체·분말은 값을 적지 않는다 (v1에는 액체 객체가 없다). 고정 설비·표면(바닥·벽·세면대·
  변기·거울·욕조)은 질량을 적지 않는다.

### 2. v1 값 (모두 `prior_estimate`, 사람 검토 필요 F-SCH-16)

일반 지식으로 넓고 보수적으로 잡았다. 질량 kg, 마찰은 무차원 정지 마찰 계수.

| 객체 | 재질 | 촉감 | 질량 (상태별) | 마찰 (젖음) |
| --- | --- | --- | --- | --- |
| mop | metal (mop_head fabric) | hard, smooth | 0.5–2.5 (젖음 0.8–3.5) | 0.3–0.8 (0.15–0.6) |
| rag | fabric | soft, absorbent, fibrous | 0.02–0.3 (젖음 0.05–0.9) | 0.4–1.0 |
| sponge | polyurethane_foam | soft, elastic, absorbent | 0.005–0.06 (젖음 0.02–0.4) | 0.5–1.2 |
| brush / toilet_brush | plastic | hard, smooth | 0.05–0.5 / 0.08–0.4 | 0.3–0.8 (0.15–0.6) |
| broom | plastic | hard, smooth | 0.3–1.5 | 0.3–0.8 |
| dustpan | plastic (pan_edge rubber) | hard, smooth | 0.1–3.0 (가득 0.3–3.0, 일부 0.15–2.0, 빔 0.1–0.8) | 0.3–0.8 |
| spray_bottle | plastic | hard, smooth | 0.04–1.2 (가득 0.3–1.2, 일부 0.08–1.0, 빔 0.04–0.15) | 0.3–0.7 |
| bucket | plastic | hard, smooth | 0.3–16 (가득 5–16, 일부 1–10, 빔 0.3–1.5) | 0.3–0.7 |
| trash_bag | plastic | soft, smooth, slippery | 0.005–15 (가득 1–15, 일부 0.1–8, 빔 0.005–0.05) | 0.2–0.6 |
| trash_bin | plastic | hard, smooth | 0.5–25 (가득 2–25, 일부 0.8–15, 빔 0.5–5) | 0.3–0.7 |
| bedsheet / pillow / clothing | fabric | soft (+fibrous·elastic) | 0.3–1.5 / 0.3–2.0 / 0.1–1.5 | 0.4–0.9 / 0.4–0.9 / 0.4–1.0 |
| towel | fabric | soft, absorbent, fibrous | 0.05–1.0 (젖음 0.15–3.0) | 0.4–1.0 |
| cup | ceramic | hard, smooth | 0.02–0.85 (가득 0.2–0.85, 일부 0.08–0.7, 빔 0.02–0.45) | 0.3–0.8 |
| door / drawer / table | wood | hard, smooth | 10–60 / 1–15 / 5–40 | 0.3–0.8 / 0.3–0.8 / 0.3–0.7 |
| bed_rail | metal | hard, smooth, cold_to_touch | 2–12 | 0.2–0.7 |
| diaper | nonwoven | soft, absorbent | 0.03–1.0 (깨끗 0.03–0.15, 더러움 0.05–1.0) | 0.4–0.9 |
| glove | rubber | soft, elastic | 0.003–0.1 | 0.6–1.5 |
| floor | (다양, 비움) | hard, smooth, slippery_when_wet | — | 0.4–0.9 (0.1–0.5) |
| sink / toilet / mirror | ceramic / ceramic / glass | hard, smooth, cold_to_touch (+slippery_when_wet) | — | 0.3–0.7 (sink 0.1–0.4) / 0.3–0.7 / 0.3–0.9 |
| wall | painted_plaster | hard | — | 0.4–0.9 |
| bathtub | plastic (아크릴 가정) | hard, smooth, slippery_when_wet | — | 0.3–0.8 (0.05–0.4) |
| bed | metal (질량은 프레임 포함, 촉감·마찰은 매트리스 덮개) | soft | 30–200 | 0.3–0.9 |

### 3. 버전·등록

- **버전을 올리지 않는다.** v1은 초안(draft)이고 이번 변경은 새 최상위 키(`materials`·`tactile_properties`)와
  객체마다 새 키(`physical_priors`)를 더하기만 한다. `register_ontology`의 덧붙이기 판정(ADR 0028)에 맞아
  `make db-upgrade`(`dlp ontology register 1.0.0`)가 등록 내용을 바꾼다
  (`packages/schema/tests/test_db.py::test_adding_physical_priors_to_registered_draft_is_additive`).
- 새 코드로 등록하면 사전값 없는 객체(person)는 `physical_priors: null`로 저장된다. 나중에 그런 객체(또는
  사전값 없이 덧붙인 새 객체)에 사전값을 채우는 것도 덧붙이기로 받도록 `register_ontology`의 판정에
  좁은 예외를 둔다: `physical_priors` 키만 null → 사전을 허용한다 (`db.repository._FILLABLE_NULL_KEYS`,
  라벨의 해석·유효성과 무관한 정보라서). 이미 있는 사전값의 값 변경은 여전히 거부한다
  (`test_db.py::test_filling_null_priors_is_additive_but_changing_values_is_not`).
- 예전 등록 내용(사전값 없음)도 새 코드로 읽힌다 (필드가 선택·기본 빈 사전). `LegacyOntologyContentError`
  (en 없는 등록)와 무관하다. 재등록 전에는 내보내기의 사전값 표가 비어 있다 (내보내기는 DB 등록본을 쓴다).
- **등록 뒤 값 변경은 덧붙이기가 아니다.** 같은 초안 버전이라도 이미 등록된 숫자·재질을 바꾸면
  `register_ontology`가 거부한다. 전문가 검토(F-SCH-16)는 운영 DB 등록 전에 반영하거나, 등록 뒤라면 새
  온톨로지 버전 + 이관 표로 낸다. 새 버전에서 상태 값을 바꾸면 `by_state`의 키도 함께 바꿔야 사전이
  검증을 통과한다.
- 내용 해시(`ontology_digest`)가 바뀐다. 결과:
  - AI 1차 검수 단위 키(F-AIR-32)가 달라져, 기존 판정 단위는 지금 온톨로지와 맞지 않아 검수 화면에 보이지
    않고 다음 `dlp aireview run`(`--pending`)이 다시 판정한다 (장비 시간 필요). 프롬프트 내용은 같다.
  - 검수 화면 프로젝트 이름의 사전 해시(`dlp_review.ontology_projects`)가 바뀌어 다음 작업 생성부터 새
    CVAT·Label Studio 프로젝트가 생긴다 (초안 덧붙이기의 기존 동작, F-COE-6의 웹훅 등록 주의 포함).
  - 번역 버전(용어집 해시)은 그대로다 (1절, 재질·촉감 이름을 용어집에 넣지 않음).

### 4. 내보내기 (측정값이 아님을 명시)

같은 표 `PhysicalPriorTable`(`measured: false` 고정, `conventions`(단위·상대면·상태별 범위 뜻, 영어),
재질·촉감 ID → 한·영 이름, 객체 ID → `PhysicalPriors`(각자 `source: prior_*`))을 싣는다. 기존 필드는
바꾸지 않고 덧붙인다.

- 구간 JSON: `physical_priors` (형식 1.2, 하위 호환).
- COCO: 사전값이 있는 객체 범주마다 `physical_priors`(`measured: false` + 사전값), 최상위
  `physical_priors`(객체별 값을 뺀 단위·가정·이름).
- LeRobot `meta/dlp_vocab.json`: `physical_priors`(표 전체).

## 결과

- 구매자는 물체 클래스마다 재질·촉감·질량·마찰의 넓은 사전 범위를 받고, 측정값이 아님을 `measured: false`와
  `source`로 기계적으로 확인할 수 있다.
- 라벨 계약·검수 도구 왕복·번역은 바뀌지 않는다.
- 숫자는 아직 추정이다 (F-SCH-16).

## 남은 결정·한계

- 장면별 촉감 라벨(이 영상의 이 물체가 실제로 미끄러운지)과 사내 VLM 제안 흐름은 없다 (F-SCH-14).
- 장갑 압력·손 3D로 파지력·질량을 추정하지 않는다 (F-SCH-15).
- 부분별 촉감·마찰은 없다 (재질만 부분별 덮어쓰기).
- 액체(밀도·점성)는 v1 객체에 없어 다루지 않았다.

## 검증

- 로더 검증: `packages/schema/tests/test_ontology.py`의 `test_v1_objects_carry_physical_priors`,
  `test_physical_prior_validation`(min>max, 0 이하, 무한대, 측정값 출처, 빈 블록, 없는 재질·촉감, 중복 촉감,
  도구 부분이 아닌 재질 부분, 객체에 없는 상태 속성·값), `test_physics_vocabulary_requires_both_names`,
  `test_valid_state_dependent_priors_are_accepted`.
- 등록: `packages/schema/tests/test_db.py::test_adding_physical_priors_to_registered_draft_is_additive`.
- 내보내기: `packages/export/tests/test_formats.py`(구간 JSON 1.2 + 공개 스키마 검증, COCO 범주·최상위,
  LeRobot 어휘).

## 배포

1. 코드를 올린다 (DB 마이그레이션 없음).
2. 숫자를 그대로 운영에 등록해도 되는지 정한다 (등록 뒤 값 변경은 새 버전, 3절). 되면 `make db-upgrade`
   (`dlp ontology register 1.0.0`)로 다시 등록한다 (초안 덧붙이기).
3. AI 1차 검수를 쓰면 `dlp aireview run --pending`으로 열린 세션을 다시 판정한다 (단위 키 변경, 장비 시간).
4. 다음 검수 작업 생성은 새 CVAT·Label Studio 프로젝트로 간다. 웹훅을 쓰면 `dlp review register-webhooks`.
5. 구매자 문서에 구간 JSON 형식 1.2(`physical_priors`)와 "사전값, 측정값 아님"을 적는다.
