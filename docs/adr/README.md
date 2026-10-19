# ADR 색인

[인수인계 첫 화면](../handover/README.md)

ADR은 결정 당시의 이유와 이관 조건을 기록한다. 후속 ADR이 기존 결정을 보완·대체할 수 있다.
현재 기능 상태와 미해결 과제는 [인수인계](../handover/README.md)를 기준으로 확인한다.

| 번호 | 결정 기록 |
| --- | --- |
| 0001 | [ADR 0001: 개발 도구 체인과 개발 서비스 구성](0001-dev-tooling-and-services.md) |
| 0002 | [ADR 0002: 계약 타입과 라벨 저장 방식](0002-contracts-and-label-storage.md) |
| 0003 | [ADR 0003: 세션 수집과 미디어 처리](0003-ingest-and-media.md) |
| 0004 | [ADR 0004: 멀티스트림 동기화](0004-stream-sync.md) |
| 0005 | [ADR 0005: 프라이버시 게이트](0005-privacy-gate.md) |
| 0006 | [ADR 0006: 검수 도구 연동](0006-review-tools.md) |
| 0007 | [ADR 0007: 데이터셋 버전과 계보](0007-datasets-and-lineage.md) |
| 0008 | [ADR 0008: 자동 프리라벨 어댑터](0008-prelabel-adapters.md) |
| 0009 | [ADR 0009: 모델 레지스트리와 실제 모델 연동 (RTMPose, OWLv2, 메트릭 깊이)](0009-real-model-registry.md) |
| 0010 | [ADR 0010: 모델의 상업 사용 분류](0010-model-commercial-use.md) |
| 0011 | [ADR 0011: 관계 도출과 표면 커버리지](0011-relations-and-coverage.md) |
| 0012 | [ADR 0012: 행동 구간 2단 구조](0012-two-stage-action-segmentation.md) |
| 0013 | [ADR 0013: 지표와 평가 하네스, 배포 게이트](0013-evaluation-harness.md) |
| 0014 | [ADR 0014: 검수 운영 로직 (우선순위 큐, 표본 검증, 블라인드·오류 삽입·이중·QA)](0014-review-operations.md) |
| 0015 | [ADR 0015: 감사 뒤 정정 (보안·멱등성·평가 정확성)](0015-audit-fixes.md) |
| 0016 | [ADR 0016: 재학습 루프와 모델 레지스트리](0016-retraining-loop.md) |
| 0017 | [ADR 0017: 액티브 러닝 (세션 선택과 FiftyOne)](0017-active-learning.md) |
| 0018 | [ADR 0018: 내보내기 (COCO, 구간 JSON, LeRobot)](0018-exports.md) |
| 0019 | [ADR 0019: 공간 라벨 시각 규약과 2차 전체 검수 정정](0019-time-convention-and-audit-2.md) |
| 0020 | [ADR 0020: 운영 지표, 원본 접근 감사, 원본 보관 만료](0020-ops-dashboard-and-security.md) |
| 0021 | [ADR 0021: 내보내기·운영 지표·원본 접근 감사 정정 (3차 검수)](0021-export-and-ops-audit-fixes.md) |
| 0022 | [ADR 0022: 3차 검수 정정 — 계약·DB·수집·동기화](0022-audit-3-contracts-db-sync.md) |
| 0023 | [ADR 0023: 프라이버시 게이트·검수 연동 감사 정정](0023-privacy-gate-and-review-audit-fixes.md) |
| 0024 | [ADR 0024: 프라이버시 게이트·검수 연동 4차 감사 정정](0024-privacy-review-audit-4.md) |
| 0025 | [ADR 0025: 평가 하네스·재학습 루프 감사 정정 (4차)](0025-evaluation-and-retraining-audit-4.md) |
| 0026 | [ADR 0026: 감사 4차 — 프리라벨·행동·관계 단계 정정](0026-audit-4-prelabel-actions-relations.md) |
| 0027 | [ADR 0027: 내보내기 ID 가명·파일 목록, COCO 이미지, 검증 완료 시각 (4차 검수)](0027-export-ids-and-verified-at.md) |
| 0028 | [ADR 0028: 4차 검수 정정 — 긴 녹화 동기화, 초안 온톨로지 갱신, 생애주기 기록](0028-audit-4-sync-drift-ontology-lifecycle.md) |
| 0029 | [ADR 0029: 검수 완료 판정과 생애주기 전이를 명령으로 남긴다](0029-verify-and-lifecycle-transitions.md) |
| 0030 | [ADR 0030: 5차 검수 — 프라이버시·수집·검수 운영·운영 지표 수정](0030-audit-5-privacy-media-review-ops.md) |
| 0031 | [ADR 0031: 5차 검수 정정 — 저장소·데이터셋·동기화·프리라벨·관계·평가·가명](0031-audit-5-schema-sync-prelabel-eval-datasets.md) |
| 0032 | [0032 — 탐지 완료·검수 세대·배포와 내보내기 게시의 일관성](0032-review-integrity-and-publication.md) |
| 0033 | [ADR 0033: 완료 탐지 세대와 검수 작업 연결·검수 계획 집계](0033-privacy-generation-and-review-query.md) |
| 0034 | [ADR 0034: 웹훅 실패 응답과 영상 표시 회전 검사](0034-webhook-boundary-and-video-orientation.md) |
| 0035 | [ADR 0035: 실제 영상 파일럿 준비 검사와 입력 무결성](0035-real-video-pilot-readiness.md) |
| 0036 | [ADR 0036: 모델 해시 캐시와 블러 렌더 복구](0036-model-cache-and-render-recovery.md) |
| 0037 | [ADR 0037: 온톨로지 로딩·저장·계층·이관 검증 경계](0037-ontology-validation-boundaries.md) |
| 0038 | [ADR 0038: 검수 화면 테마를 도구 밖에서 덧씌운다](0038-review-ui-theme-overlay.md) |
| 0039 | [ADR 0039: 블러가 바뀌면 이미 배포한 라벨링 작업 매체를 회수한다](0039-revoke-labeling-media-on-blur-change.md) |
| 0040 | [ADR 0040: AI 1차 검수(로컬 VLM)는 참고용 판정만 남기고 사람 검수를 줄이지 않는다](0040-ai-first-pass-review.md) |
| 0041 | [ADR 0041: 골든셋 버전은 만들 때 정답 라벨 ID를 고정한다](0041-golden-truth-freeze.md) |
| 0042 | [ADR 0042: 사고 대응 중 프라이버시 승인을 보류한다](0042-privacy-approval-hold.md) |
| 0043 | [ADR 0043: 라벨을 한국어와 영어로 함께 남긴다 (온톨로지 영어 이름, 설명 자동 번역)](0043-bilingual-labels.md) |
| 0044 | [ADR 0044: 단일 카메라 영상에서 손가락 끝 3D를 손 모델 3D 점으로 맞춘다](0044-fingertip-3d-from-hand-world-landmarks.md) |
| 0045 | [ADR 0045: 3D 표본의 방식·신뢰도, 카메라 내부 파라미터 등록, 작업자 손 크기 사전값](0045-3d-sample-quality-calibration-and-hand-size.md) |
| 0046 | [ADR 0046: 바디캠 6자유도 자세(IMU 회전 + 메트릭 깊이 이동)로 손 3D를 세션 월드 좌표에 둔다](0046-camera-pose-and-world-frame.md) |
| 0047 | [ADR 0047: 사람 1인칭 손 동작을 고객 로봇 관절 궤적으로 옮겨 LeRobot 에피소드로 낸다](0047-robot-retargeting-export.md) |
| 0048 | [ADR 0048: 3D 손 궤적은 그것을 계산한 2D 손 트랙의 검수를 물려받는다](0048-3d-trajectory-review-inherited-from-2d.md) |
| 0049 | [ADR 0049: 온톨로지 객체에 물리 사전값(재질·촉감·질량·마찰)을 둔다](0049-object-physical-priors.md) |
| 0050 | [ADR 0050: 끊긴 물려받은 3D는 궤적 따르기가 지우고, 박스 중심 3D도 물려받으며, 카메라 자세는 자동 라벨이다](0050-inherited-3d-retraction-box-center-and-automatic-pose.md) |
| 0053 | [ADR 0053: 리타기팅·내보내기·물리 사전값 전수 검수 정정 (묶음 C)](0053-retargeting-export-and-prior-review-fixes.md) |
