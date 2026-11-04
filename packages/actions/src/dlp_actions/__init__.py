"""행동 구간 2단 구조: 운동·접촉 신호로 경계 후보 → VLM이 후보 구간 분류·설명 (WP10).

`dlp actions run <세션> --vlm-url`(`dlp_cli.actions_cmds`)이 `runner.run_actions`를 부른다.
ADR 0012(2단 구조), 0015(멱등·검수 보호), 0019(시각 규약), 0024(블러본만), 0026(4차 검수 정정).

왜 2단인가 (ADR 0012 1절): VLM은 긴 클립에서 시간 경계를 잘 잡지 못한다. 그래서 경계 시각은 손목
속도·접촉 같은 측정 신호로 정하고, VLM은 이미 자른 구간이 무슨 동작인지만 온톨로지 목록 안에서
고른다. VLM 서버가 없으면 행동 구간을 만들지 않는다 (전부 미상으로 채우면 검수 부담만 는다, 5절).

흐름 (손마다)
1. `boundaries`: 손목 속도(손바닥 길이/초)와 접촉 구간에서 경계 후보를 만든다. 경계는 VLM이
   정하지 않는다.
2. `boundaries.segments`: 후보로 타임라인을 빈틈없이 자른다.
3. `vlm.classify`: 후보 구간마다 VLM이 온톨로지 목록 안에서 행동(원시 동작·대상·도구) 또는 사이 구간
   종류를 고르고 한 문장 설명을 쓴다 (JSON Schema 강제, 위반이면 재시도 후 미상).
4. `assemble`: 같은 분류의 인접 구간을 병합하고 action·gap·description 라벨을 만든다.
5. `runner`: 멱등 실행, 이전 버전 정리, 검수된 라벨 보호, DB 쓰기.

모델 버전 (라벨 `provenance.model_version`, 라벨 ID에는 `version_tag`로 들어간다)
- 행동: `actions-<stamp>+<VLM 버전>+o<온톨로지 부분 해시>+i<입력 해시>` (`runner`).
  `stamp`는 `actions.yaml` 해시(백오프·사내 대역·fingerprint 제외)이고 프롬프트 틀 버전이 1이
  아니면 `+pv<틀 버전>`이 붙는다. VLM 버전은 `vlm-<vlm.model>@<vlm.fingerprint>`(엔진·양자화·가중치
  revision, F-PAR-90. stub은 `oracle-vlm-1`). 온톨로지 부분 해시는
  프롬프트가 읽는 범주(`verbs` 전체·`gap_types`)의 내용 해시다. 프롬프트에는 원시 동작만 들어가지만
  해시는 범주 단위라 기술 수준 동사를 덧붙여도 버전이 바뀐다. 입력 해시는 고른 손 트랙·손 상태·
  객체 트랙 라벨 ID 집합의 해시다. 이 넷 중 하나라도 바뀌면 버전이 바뀌어 다음 실행이 다시 만든다.
- 설명 재생성: `describe-<stamp>+<VLM 버전>+o<온톨로지 부분 해시>` (`describe.describe_version`).
- 번역: `translate-<translate.yaml 해시>+<번역기 버전>+g<용어집 해시>`
  (`translate.translation_version`).
- 버전이 바뀌면 검수 전인 이 단계 레코드만 `retractions()`로 지운다. 검수된 것과 사람 레코드는
  남긴다 (ADR 0015, `runner` 모듈 docstring).

모듈
- `policy`: `config/policies/actions.yaml` (해시 = 모델 버전의 일부)
- `boundaries`, `vlm`, `assemble`, `pipeline`(한 손), `runner`(세션), `clients`(VLM 클라이언트)
- `write_check`: 쓰기 직전 세션 확인 (사용 중지·승인 보류·블러본 렌더 해시)
- `describe`: 고친 행동의 설명 재생성 (`dlp actions describe`). 설명은 행동 검수를 따른다
  (ADR 0043).
- `translate`, `translate_policy`, `translators`: 설명 영어 번역 (`dlp translate run|pending`,
  ADR 0043).

CPU·CI에서는 정답 라벨로 답하는 `clients.OracleVlm` stub을 쓴다. 실제 VLM은 OpenAI 호환 서버
(`clients.OpenAICompatibleVlm`)이며 아직 실제 서버로 검증하지 않았다 (TODO(real-model)).
시간: 행동·사이 구간·설명 라벨 시각은 마스터 타임라인 정수 ms다 (ADR 0019).
"""
