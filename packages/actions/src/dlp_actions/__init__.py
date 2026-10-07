"""행동 구간 2단 구조: 운동·접촉 신호로 경계 후보 → VLM이 후보 구간 분류·설명 (WP10).

`dlp actions run <세션> --vlm-url`(`dlp_cli.actions_cmds`)이 `runner.run_actions`를 부른다.
ADR 0012(2단 구조), 0015(멱등·검수 보호), 0019(시각 규약), 0024(블러본만), 0026(4차 검수 정정).

흐름 (손마다)
1. `boundaries`: 손목 속도(손바닥 길이/초)와 접촉 구간에서 경계 후보를 만든다. 경계는 VLM이
   정하지 않는다.
2. `boundaries.segments`: 후보로 타임라인을 빈틈없이 자른다.
3. `vlm.classify`: 후보 구간마다 VLM이 온톨로지 목록 안에서 행동(원시 동작·대상·도구) 또는 사이 구간
   종류를 고르고 한 문장 설명을 쓴다 (JSON Schema 강제, 위반이면 재시도 후 미상).
4. `assemble`: 같은 분류의 인접 구간을 병합하고 action·gap·description 라벨을 만든다.
5. `runner`: 멱등 실행, 이전 버전 정리, 검수된 라벨 보호, DB 쓰기.

모듈
- `policy`: `config/policies/actions.yaml` (해시 = 모델 버전의 일부)
- `boundaries`, `vlm`, `assemble`, `pipeline`(한 손), `runner`(세션), `clients`(VLM 클라이언트)

CPU·CI에서는 정답 라벨로 답하는 `clients.OracleVlm` stub을 쓴다. 실제 VLM은 OpenAI 호환 서버
(`clients.OpenAICompatibleVlm`)이며 아직 실제 서버로 검증하지 않았다 (TODO(real-model)).
시간: 행동·사이 구간·설명 라벨 시각은 마스터 타임라인 정수 ms다 (ADR 0019).
"""
