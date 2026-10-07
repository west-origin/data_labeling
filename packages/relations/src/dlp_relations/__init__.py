"""관계 도출(YAML 규칙 엔진)과 도구-표면 접촉·표면 커버리지 (WP9).

진입점: `dlp relations run <세션>` (`dlp_cli.relations_cmds` → `runner.run_relations`). 멱등이고,
규칙(`config/policies/relations.yaml`)이 바뀌면 차이만 반영한다. 관계는 사람이 그리지 않고 이
규칙으로만 만든다 (관계의 `derived_by` = 규칙 ID). 관련: ADR 0011(관계·커버리지), 0015, 0026.

흐름: 현재 라벨(손 상태, 3D 궤적, 박스·마스크 트랙) → `derive`(순수 계산) → 관계·커버리지 초안 →
`runner`가 이전 결과와 비교해 새 것만 넣고 사라진 것은 삭제 레코드로 표시한다.

모듈:
- `policy`: 정책 로더와 정책 해시(`RelationsPolicy.digest`, 모델 버전에 들어간다).
- `rules`: 규칙 엔진 (입력 구간 → 관계 초안, 짧은 끊김 병합).
- `geometry`: 표면 평면·표면 좌표, 궤적 선형 보간.
- `contact`: 도구 작용부-표면 접촉 구간 (평면 거리 히스테리시스).
- `coverage`: 접촉 경로가 덮은 표면 면적 비율.
- `derive`: 라벨에서 위 계산을 엮는다 (DB 없음).
- `runner`: DB에서 읽고 쓰는 멱등 실행.

시간: 관계·커버리지는 마스터 타임라인 구간이다(stream_id 없음). 입력 3D 궤적 시각은 바디캠 PTS
ms이고 바디캠이 기준 스트림이라 같은 축으로 본다 (ADR 0019).
"""
