"""청소 전후 다각도 사진의 3D 복원·물체 이동 비교와 장면 안 바디캠 위치 추정 (ADR 0066, 0067).

흐름 (명령은 `dlp scene3d …`, `dlp_cli.scene3d_cmds`):
1. `ingest <세션> <before|after> <디렉터리>` → `photos.ingest_capture`: 원본 사진을 감사 원본
   저장소에 불변으로 둔다 (photo_id = 원본 sha256).
2. `privacy detect|review|approve|render` → `privacy_gate`: `dlp_privacy` 탐지기로 사진마다 블러
   상자를 찾고, 원본 접근 권한자가 모든 사진의 상자를 확정해 승인한 뒤에만 블러 PNG를 라벨링
   버킷에 쓴다. 이후 단계는 `open_blurred`로만 사진을 받는다.
3. `reconstruct <세션> <단계>` → `pipeline.reconstruct`: SfM 카메라 자세·희소 점(`backends`),
   크기를 아는 마커로 정한 미터 척도(`markers`, 없으면 metric=False), 3D 가우시안 스플랫 PLY.
4. `compare <세션>` → `pipeline.compare`: after → before 정렬(`alignment`, 확신이 낮으면 거부),
   사진별 물체 탐지의 다시점 삼각측량(`objects`), 전후 대응과 moved·unchanged·removed·added 분류
   (`change`). 위치·이동 벡터는 before 장면 좌표(미터).
5. `report <세션>` → 지금 비교 결과 JSON·CSV.
6. `localize <세션>` → `localize.localize`: 그 세션 바디캠의 지금 승인된 블러본(`ego_video`)에서
   프레임마다 카메라 자세를 before 장면 좌표로 구한다 (기술자 대응 + PnP `pnp`). 전후 복원과
   받아들여진 정렬이 모두 있어야 한다. 읽기는 `localize.current_localization`·`PoseTrack`.
7. `status <세션>` → `status.scene_status`: 전후가 다 갖춰졌는지 (읽기 전용).

순수 계산(`geometry`, `objects`, `alignment`, `change`, `pnp`)은 저장소·모델 없이 돈다. 계약 타입은
`dlp_schema.scene3d`, 정책은 `policy`(`config/policies/scene3d.yaml`).
"""
