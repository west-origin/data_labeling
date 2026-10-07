# config

- `defaults.yaml`: 미결정 사항의 기본값. 결정이 나면 값만 바꾼다.
- `ontology/<버전>/`: 온톨로지 사전 YAML (WP1에서 추가).
- `policies/`: 작업 패키지별 정책. `sync.yaml`(WP4 동기화 방법 순서·임계값), `privacy.yaml`(WP5 블러 대상별 탐지기·여유·추적·검수 우선순위). 이후 블러, 큐, 샘플링, 내보내기 정책을 추가한다.
