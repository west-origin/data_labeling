"""청소 전후 다각도 사진의 3D 복원과 물체 이동 비교 (ADR 0066).

흐름 (명령은 `dlp scene3d …`, `dlp_cli.scene3d_cmds`):
1. `ingest <세션> <before|after> <디렉터리>` → `photos.ingest_capture`: 원본 사진을 감사 원본
   저장소에 불변으로 둔다 (photo_id = 원본 sha256).
2. `privacy detect|review|approve|render` → `privacy_gate`: `dlp_privacy` 탐지기로 사진마다 블러
   상자를 찾고, 원본 접근 권한자가 모든 사진의 상자를 확정해 승인한 뒤에만 블러 PNG를 라벨링
   버킷에 쓴다. 이후 단계는 `open_blurred`로만 사진을 받는다.
3. `reconstruct <세션> <단계>` → `pipeline.reconstruct`: SfM 카메라 자세·희소 점(`backends`)과
   3D 가우시안 스플랫 PLY.
4. `compare <세션>` → `pipeline.compare`: after → before 정렬(`alignment`, 확신이 낮으면 거부),
   사진별 물체 탐지의 다시점 삼각측량(`objects`), 전후 대응과 moved·unchanged·removed·added 분류
   (`change`). 위치·이동 벡터는 before 장면 좌표(미터).
5. `report <세션>` → 지금 비교 결과 JSON·CSV.

순수 계산(`geometry`, `objects`, `alignment`, `change`)은 저장소·모델 없이 돈다. 계약 타입은
`dlp_schema.scene3d`, 정책은 `policy`
(`config/policies/scene3d.yaml`).
"""
