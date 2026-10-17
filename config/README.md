# config

- `defaults.yaml`: 미결정 사항의 기본값. 결정이 나면 값만 바꾼다.
- `ontology/<버전>/`: 온톨로지 사전 YAML (WP1에서 추가).
- `policies/`: 작업 패키지별 정책. `sync.yaml`(WP4 동기화 방법 순서·임계값), `privacy.yaml`(WP5 블러 대상별 탐지기·여유·추적·검수 우선순위), `dataset.yaml`(WP7 검증 비율·골든셋 크기·lakeFS), `prelabel.yaml`(WP8 모델·접촉·착용자 매칭). 이후 블러, 큐, 샘플링, 내보내기 정책을 추가한다.
- `policies/retarget.yaml`: 로봇 리타기팅 운영 규칙(풀이기, 도달하지 못한 프레임 처리, 손 점 이름, mink 격리 환경, ADR 0047). `prelabel.yaml camera_pose`·`world_frame`은 바디캠 자세·월드 좌표 3D(ADR 0046).
- `robots/<ID>.yaml`: 리타기팅 대상 로봇 (모델 경로·sha256·라이선스, 팔 관절·끝 링크·축·그리퍼·배치, IK 설정). 저장소에는 공개 시험 모델(`so101`, `robots/models/so101/`, Apache-2.0)만 둔다. 고객 로봇은 저장소 밖 `DLP_ROBOTS_DIR` (절차 `docs/handover/robot-retargeting.md`).
- `review_ui.yaml`: 검수 화면 테마 (CVAT·Label Studio의 다크/화이트 색, 라벨 색, 원시 동작 단축키, 로고, ADR 0038). 바꾼 뒤 `make ui-theme`, CVAT는 `make cvat-up`.
