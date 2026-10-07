"""내보내기 (WP15): 데이터셋 버전 → COCO, 구간 JSON, LeRobot 에피소드.

파이프라인의 마지막 단계다. `dlp export coco|intervals|lerobot <데이터셋 버전>`
(`dlp_cli.export_cmds`)이 `dlp_export.runner.run_export`를 부른다.
관련 문서: WP15(docs/ai-implementation-plan.md), ADR 0018(내보내기), 0021(3차 검수 정정:
이력 선커밋·가명·LeRobot 격리 환경 잠금), 0024(승인 취소 뒤 이전 블러본 금지),
0027(세션·라벨 ID 가명), 0029(내보낸 세션의 exported 전이).

핵심 규칙:
- 라벨은 데이터셋 버전 스냅샷(lakeFS 커밋, `dlp_datasets.snapshot`)에서 읽는다
  (같은 버전이면 같은 내보내기). DB의 지금 라벨을 읽지 않는다.
- 검증 정책: 기본은 사람이 만든 라벨과 사람 승인·수정·표본 검증 라벨만. 미검수는 명시적 옵션
  (`--include-unreviewed`)으로만. 라벨마다 검증 상태를 그대로 표시한다.
- 사용 중지 세션은 내보내는 시점 기준으로 뺀다 (스냅샷에 있어도). 블러 라벨은 어떤 형식에도
  내보내지 않는다.
- 영상은 라벨링 버킷의 블러본만 쓴다. 원본 버킷 URI가 결과에 있으면 실패한다.
- 작업자·장소·세션·라벨 ID는 내보내기마다 다른 가명(HMAC)으로 바꾼다 (`dlp_export.pseudonym`).
- 내보내기 이력(DB `exports`)은 버킷에 올리기 **전에** 따로 커밋한다 (`dlp_export.runner`).

모듈:
- `policy`: 정책(config/policies/export.yaml) 로더와 검증.
- `source`: 내보낼 세션·라벨 고르기, 블러본 받기, 원본 위치 검사 (모든 형식 공용).
- `pseudonym`: 내보내기별 가명.
- `frames`: 마스터 시각 ↔ 스트림 시각, PTS로 프레임 고르기·디코딩.
- `coco` / `intervals` / `lerobot`: 형식별 쓰기.
- `runner`: 전체 실행 순서 (고르기 → 쓰기 → manifest → 이력 커밋 → 올리기 → 생애주기 전이).
"""
