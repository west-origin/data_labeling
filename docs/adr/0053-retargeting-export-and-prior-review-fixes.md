# ADR 0053: 리타기팅·내보내기·물리 사전값 전수 검수 정정 (묶음 C)

- 상태: 채택
- 날짜: 2026-10-19
- 관련: ADR 0045(3D 표본 방식·신뢰도), ADR 0047(로봇 리타기팅), ADR 0048·0050(3D 검수 물려받기),
  ADR 0049(물리 사전값), ADR 0027(내보내기 가명), ADR 0028(초안 온톨로지 덧붙이기). 전수 검수
  `docs/handover/full-review-2026-10-17.md` 묶음 C, 문제 레지스트리 F-COE-40~56·F-SCH-20~25.
- 계약 변경:
  - LeRobot 특징 추가 `annotation.hand_points_3d_method`·`annotation.hand_points_3d_intrinsics`·
    `annotation.contact_object_class`, `meta/dlp_vocab.json`의 `object_class`·`hand_point_3d_method`·
    `hand_point_3d_intrinsics`, `meta/dlp_episodes.json`의 `hand_points_3d_coverage`, 로봇 모드
    `meta/dlp_retarget.json`의 `ik`·`hand_parts`·`reachable_definition`·`orientation_error_definition`.
    로봇 모드 `retarget.orientation_error_rad`는 orientation_mode none이면 -1(재지 않음),
    `retarget.velocity_limited`는 그리퍼와 재시작 프레임도 센다.
  - `dlp_schema.export.PhysicalPriorTable.content_sha256`과 규약 `base_range`·`reference_surface`·
    `content_sha256` (구간 JSON 형식은 배포 전이던 1.2 그대로, `make schemas`).
  - 온톨로지 검증 규칙(`dlp_schema.ontology`): `StatePriorRange` 상태 속성 하나, 기본 범위의 뜻
    `BASE_CONDITION_STATES`, `NO_PRIOR_PHYSICAL_TYPES`, prior_literature note 필수, 대표와 같은 부분 재질
    거부. v1 값은 그대로 통과한다 (`ontology_digest` 불변).
  - `db.repository.register_ontology(..., accept_physical_priors=False) -> RegisterAction`,
    `PhysicalPriorsChangeError`. DB 스키마 변경 없음 (Alembic 없음).
  - 정책: `export.yaml lerobot.interp_max_gap_3d_ms`·`min_hand_points_3d_coverage`, SO-101
    `ik.orientation_mode: full`. 리타기팅 판의 풀이기 판 `dls1 → dls2`, `mink1 → mink2`.

## 배경

2026-10-17 전수 검수가 리타기팅·LeRobot 내보내기·물리 사전값에서 24건을 냈다(두 검증 실행). 핵심은 세
가지였다. (1) 3D 표본 간격(깊이 200 ms 뒤 첫 프레임)과 LeRobot 보간 한도(200 ms)가 같아 29.97·24 fps·VFR
영상에서 손 3D와 로봇 목표가 대부분 빠졌다. (2) SO-101을 approach 모드로 풀어 wrist_roll이 사람 손과
무관한 값으로 팔렸다. (3) 사전값 묶음이 덧붙이기 판정의 키 추가 규칙에 묶여 같은 버전 안에서 뜻이 조용히
바뀔 수 있었고, 반대로 정정 경로는 검수를 모두 버리게 했다.

## 결정

### 1. 리타기팅 (`dlp_retarget`, ADR 0047 보완)

- **orientation_mode는 관절 수가 아니라 구조로 고른다.** 접근 축 둘레 롤을 바꾸는 관절(손목 끝 wrist
  roll)이 있으면 full, 없을 때만 approach. SO-101(pan + pitch x 3 + roll)은 full이다: 접근 축의 요는
  shoulder_pan에 묶여 맞추지 못하고(가중 최소제곱이 위치와 나눠 지고 오차로 보고), 롤은 맞춘다.
  합성 정답에서 닫힘 축 오차 최대 22도 → 1도 안, 사람 손 요가 ±20도 어긋난 입력에서 롤 오차 평균 53도 →
  3도 안 (도달 비율 그대로).
- **다시 시작하는 프레임도 연속·속도 한계.** 목표가 끊긴 뒤·첫 목표 프레임의 재시작은 직전 해를 첫
  시작점에 넣고, 도달 해 가운데 직전 해에 가장 가까운 것을 쓴다(없으면 비용 최소). 첫 프레임을 뺀 모든
  프레임을 직전 프레임(내보낸 값, 목표 없음이면 유지 값) 기준 속도 한계로 자르고 `velocity_limited`로
  표시한다 (`ik.limit_step`). 그리퍼도 URDF 그리퍼 velocity x velocity_scale로 같은 규칙을 쓴다. mink는
  끊긴 뒤 마지막 해에서 시작한다. 잘린 프레임은 도달·오차로 드러난다 (지어내지 않는다).
- **목표가 하나도 없는 에피소드는 정책과 관계없이 drop** (관절이 모두 시작 자세라 로봇 데이터로 뜻이
  없다). 첫 목표 앞 프레임은 시작 자세(관절 범위 가운데)·닫힌 그리퍼이고 `target_present=0`이다.
- **orientation_mode none의 회전 오차는 NaN**(내보내기 -1). 0이면 방향이 완벽히 맞은 것과 구분되지 않는다.
- **설정 검증**: 접근·닫힘 축은 단위 벡터이고 직교(허용 오차 1e-6, 조용히 정규화하지 않는다), 그리퍼
  관절은 움직이는 독립 관절이고 팔 관절이 아니며 closed·open이 URDF 한계 안, robot_id는 Identifier
  형식(경로 탈출 금지), 비공개 설정이 저장소 안 모델 파일을 가리키면 거부, `config/robots/models` 아래
  파일은 공개 설정의 모델과 LICENSE뿐(저장소 시험).
- **URDF**: 연쇄 안 `<mimic>` 관절은 거부(독립 자유도로 풀면 결합 조건을 어긴다), revolute·prismatic은
  `<limit lower upper>` 필수(urdfdom과 같다), continuous만 ±inf.
- **mink approach 비용**: 끝 링크에 도구 중심점 site(z = 접근 축, x = 닫힘 축)를 달고 site 좌표 비용
  [w, w, 0]을 쓴다. 끝 링크 좌표 대각 비용 `w(1 - |a|)`는 접근 축이 좌표축이 아니면 롤에도 비용이 붙었다.
- 내보내기 meta에 IK 기준(`ik`), 손 점 역할(`hand_parts`), reachable·회전 오차의 뜻을 싣는다.
- 판: `dls2`, `mink2` (리타기팅 판이 바뀌어 다시 내보내면 결과가 다르다).

### 2. LeRobot 내보내기 (`dlp_export`)

- **3D 보간 한도 분리**: `lerobot.interp_max_gap_3d_ms`(300)를 손 3D·도구 작용부 3D·로봇 목표(월드 손
  점)에 쓴다. 2D 키포인트는 `interp_max_gap_ms`(200) 그대로. 정책 대조 시험이 `3D 한도 ≥ depth.frame_stride_ms
  + 1000/23.976`을 지킨다. 깊이 표본을 고정 격자로 고르는 방안은 반올림으로 201 ms 간격이 여전히 생겨
  한도 분리 없이 해결되지 않으므로 하지 않았다.
- **채움 경고**: 에피소드마다 궤적 구간 안 손 3D 점 채움 비율을 `dlp_episodes.json`에 싣고,
  `min_hand_points_3d_coverage`(0.9)보다 낮으면 manifest `details.warnings`와 CLI `경고:`로 알린다.
- **3D 방식·내부 파라미터 출처 특징**: 신뢰도 범위가 방식·출처끼리 겹치므로(관절별 깊이 대체 0.4가 hand_fit
  0.277보다 높을 수 있다) 손 3D 점마다 방식(hand_fit 0, point_depth 1, 보간 양끝이 다르면 mixed 2)과
  내부 파라미터 출처(calibrated 0, hfov_approx 1)를 싣는다. ADR 0045 1절의 "신뢰도가 이미 방식을
  반영한다"는 이 ADR로 대체한다. 신뢰도 식 자체는 바꾸지 않았다(3D 단계 담당).
- **접촉 객체 클래스 특징**: 손 상태의 접촉 대상(object·tool·fixed_surface)이 가리키는 개체의 박스·마스크
  트랙 클래스를 `annotation.contact_object_class`(어휘 `object_class` = 객체 ID 이름순)로 싣는다. LeRobot
  만 받은 구매자도 `dlp_vocab.json`의 객체별 사전값을 프레임에 이을 수 있다. 트랙이 없으면 -1.
- **손 부위 검증**: `hand_points_3d`는 1개 이상·중복 없음(정책), 내보내기 시작 때 hand_joints와 대조
  (로봇 모드는 `retarget.yaml hand_parts`도), 정책 대조 시험이 프리라벨 `depth.hand_points`와 대조한다.
- **크기 기준**: 리타기팅 정책이 버린 에피소드는 데이터셋 크기·화면비 기준에 넣지 않는다.
- **가명**: 페이로드의 라벨 ID 필드(`source_label_id`, 3D `derived_from`)는 `pseudonym.LABEL_REF_FIELDS`로
  통째로 라벨 가명이 된다. 시험이 계약의 모든 Identifier 필드가 `_id`·`_ids` 규칙이거나 이 목록에 있는지
  대조한다. 지금 3D는 구간 JSON에 나가지 않지만(kinds), 정책 한 줄로 내부 세션 ID가 나가던 틈을 막는다.

### 3. 물리 사전값 (ADR 0049 보완)

- **사전값 묶음은 덧붙이기 판정과 따로 본다.** 객체 `physical_priors`와 재질·촉감 어휘를 뺀 나머지만
  키 덧붙이기 규칙으로 보고, 사전값 묶음이 조금이라도 다르면(null 채우기, 값 변경, by_state·part_materials
  추가) `dlp ontology register <버전> --accept-physical-priors`일 때만 초안 내용을 바꾼다. 승인 없으면
  종료 코드 3으로 아무것도 쓰지 않는다. 확정(frozen) 버전은 승인해도 거부한다.
  - 사전값은 라벨의 해석·유효성과 무관해 라벨 검수에 영향이 없다. 그래서 정정 경로로 "새 버전 + 이관"
    (모든 검수를 미검수로 되돌리고, 세션 버전 전환 CLI도 없다)을 쓰지 않고, 초안에서는 승인 옵션으로
    교체한다. 감사용 별도 표는 두지 않았다(DB 변경 없음): 교체 여부와 사전값 객체 수를 출력하고, 내보내는
    표에 `content_sha256`을 실어 납품 사이 변경을 구매자가 가린다.
  - 내용 해시(`ontology_digest`)가 바뀌는 결과(AI 판정 단위 재판정, 새 검수 프로젝트)는 ADR 0049 3절과 같다.
- **`make db-migrate`**(마이그레이션만)를 두고 운영 배포의 마이그레이션 단계는 이것을 쓴다. `make
  db-upgrade`(마이그레이션 + 등록)는 승인 옵션을 주지 않으므로 사전값을 조용히 등록·변경하지 않는다(처음
  등록하는 빈 DB는 config 그대로 넣는다).
- **기본 범위의 뜻**: 젖음(wetness)은 기본 = 건조 조건(by_state.wetness.wet만, dry 별도 금지), 그 밖의
  상태 속성은 기본 범위가 모든 상태를 덮고 by_state 범위가 그 안이다. by_state는 상태 속성 하나만. 상태를
  모르면 기본과 by_state 전체의 포락선을 쓴다. 도구의 마찰·촉감은 파지부 기준. 구매자 규약(영어)에 같은
  뜻을 싣는다.
- **추가 검증**: prior_literature는 note(출처) 필수, person·fluid_powder 유형은 사전값 거부, 대표 재질과
  같은 part_materials 거부, 빈 상태 값 사전 거부. 수치 타당성 상한(마찰 ≤ 3, 클래스별 질량 상한)은 요구가
  없어 두지 않았다 (F-SCH-16 검토 때 단위 확인).
- **구간 JSON 하위 호환의 뜻**: 새 스키마·리더가 옛 파일을 받는다. 옛 스키마는 최상위
  additionalProperties false라 새 파일을 거부한다(내부 필드 유출을 잡으려고 엄격하게 둔다). 받는 쪽은
  `format_version`에 맞는 스키마를 쓴다. 시험이 1.2 스키마·리더로 1.0·1.1 모양 파일을 검증한다.

## 결과

- 29.97·24 fps·VFR 세션에서도 손 3D·로봇 목표가 궤적 구간 안에서 빠지지 않고, 빠지면 경고가 난다.
- 로봇 관절 열(wrist_roll·그리퍼)이 사람 손을 따르고, action에 로봇이 낼 수 없는 한 걸음이 들어가지 않는다.
- 같은 온톨로지 버전 안의 사전값 변경은 운영자가 승인해야 하고 내보낸 표의 해시로 드러난다.
- 비용: 리타기팅 판·LeRobot 특징이 바뀌어 기존 납품과 열 구성이 다르다(구매자 문서 갱신). 로봇 모드에서
  재시작 직후 몇 프레임은 속도 한계로 미도달이 될 수 있다.

## 남은 일

- 확정(frozen) 온톨로지 버전의 사전값 정정 경로(라벨 이관 없이 사전값만 새 판으로)는 없다. 지금 v1은 초안이다.
- 재시작 프레임을 따로 표시하는 특징(`retarget.restart`)은 두지 않았다 (`target_present` 0 → 1 전이로 안다).
- 3D 표본 신뢰도 식의 방식별 구간 분리는 3D 단계(ADR 0045) 몫이다.

## 검증

- `packages/retarget/tests/test_retarget.py`: 닫힘 축·wrist_roll 정답, 요 어긋남 입력의 롤, 1·7·앞 10프레임
  결손의 재시작 연속·속도 한계, 그리퍼 속도 한계, 목표 없는 에피소드 drop, none 모드 회전 오차, 축·그리퍼·
  robot_id·저장소 모델 검사, URDF mimic·한계. `test_mink.py`(격리 환경): 접근 축이 좌표축이 아닌 로봇.
- `packages/export/tests/test_formats.py`: 29.97·24 fps·VFR 3D 채움, 방식·출처·접촉 객체 클래스, 규약·해시,
  구간 JSON 하위 호환, derived_from 가명과 계약 ID 필드 대조, 손 부위 검증. `test_robot.py`: meta 기준,
  24 fps 월드 표본 목표, 버린 에피소드 크기. `test_source_inheritance.py`: 물려받은 3D의 DB·스냅샷 경로.
- `packages/schema/tests/test_db.py`: 사전값 승인 등록. `test_ontology.py`: 추가 검증.
- `tests/test_policy_consistency.py`: 3D 보간 한도와 깊이 간격, 손 부위와 프리라벨 산출.

## 배포

1. 코드를 올린다 (DB 마이그레이션 없음). 다른 ADR의 마이그레이션은 `make db-migrate`로 한다.
2. ADR 0049 배포 2단계의 결정 뒤 `uv run dlp ontology register 1.0.0 --accept-physical-priors`.
3. 구매자 문서: LeRobot 새 특징 3개와 어휘, 로봇 모드 meta의 판정 기준, SO-101 full 모드, 구간 JSON
   하위 호환의 뜻, 사전값 표 `content_sha256`·`base_range`.
4. 로봇 모드 납품은 다시 내보낸다 (리타기팅 판 dls2).
