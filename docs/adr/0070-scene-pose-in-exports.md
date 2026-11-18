# ADR 0070: 내보내기 프레임마다 장면 좌표의 바디캠 자세를 싣기 (`dlp export lerobot|gr00t`)

- 상태: 채택 (구현됨. 합성 정답으로만 확인. LeRobot 공식 쓰기·읽기 API와 GR00T 자체 로더로 읽는 확인은 아직)
- 날짜: 2026-11-18
- 관련: ADR 0067(장면 안 바디캠 위치 추정, 이 결정의 입력), 0069(촉각장갑 채널, 같은 방식으로 싣는다),
  0066(전후 3D 복원), 0065(GR00T 형식), 0047(로봇 모드), 0027(내보내기마다 다른 가명), 0019(시각 규약)
- 계약 변경: 없음 (`dlp_schema` 그대로, `make schemas` 대상 없음). 정책 `config/policies/export.yaml scene` 절과
  `dlp_export.policy.ScenePolicy`를 더했다. 구매자에게 주는 새 파일 `meta/dlp_scene.json`(형식 `dlp-scene`, 판 1.0)이
  생겼다. 패키지 의존 `export → scene3d`를 더했다 (`docs/handover/architecture.md` 의존 표).

## 배경

소유자 요구 둘째는 "전후 사진으로 만든 공간 정보 안에서 1인칭 작업 영상과 촉각 센서를 동기화한다"이다. ADR 0067이
바디캠 프레임별 카메라 자세를 before 장면 좌표로 구했고(`dlp scene3d localize`), ADR 0069가 촉각장갑 값을 에피소드
프레임 시각에 맞춰 내보내기에 실었다. 두 결과는 같은 마스터 시각을 쓰지만 서로 다른 곳에 있었다. 납품하는 에피소드의
한 행에 영상 프레임·손 3D·촉각 값과 함께 "그 순간 카메라가 공간의 어디에서 어디를 보는가"가 있어야 공간 안에서
동기화된 자료가 된다.

## 결정

1. **싣는 곳.** `dlp export lerobot`과 `dlp export gr00t` 에피소드에, 세션의 위치 기록에서 읽은 프레임 시각의 카메라
   자세를 싣는다 (`dlp_export.scene`). coco·intervals에는 싣지 않는다.
2. **값.** 카메라 중심 x·y·z(장면 좌표)와 카메라 → 장면 회전의 단위 사원수 qw·qx·qy·qz(w ≥ 0). 위치 기록의
   `x_cam = R x + t`에서 회전은 Rᵀ, 중심은 -Rᵀ t다. 카메라 좌표의 점 p(OpenCV 축, observation.state의 손 3D와 같은
   좌표)는 장면 좌표 `R(q) p + (x, y, z)`다. 행렬 9칸이 아니라 사원수로 둔 것은 상태 벡터가 짧고 로봇 데이터셋에서
   흔한 표현이기 때문이다.
3. **시간 맞춤.** 에피소드 프레임 시각(마스터)을 위치 기록 스트림의 영상 시각으로 옮기고(`frames.stream_ms`), 그
   시각의 자세를 `dlp_scene3d.localize.PoseTrack.at`이 이웃한 두 자세 사이에서 보간한다 (중심 선형, 회전 구면). 간격
   한도는 위치 기록을 읽는 규칙인 `scene3d.yaml interpolation.max_gap_ms`(500)를 그대로 쓴다. 내보내기 정책에 같은
   값을 두 번 적지 않는다.
4. **지어내지 않는다.** 위치 기록 밖(외삽 없음), 이웃 간격이 한도를 넘는 곳, 쓸 위치 기록이 없는 세션은 값 0과 표시
   0이다 (손 3D 점·촉각과 같은 표현).
5. **쓸 수 있는 위치 기록.** 둘 다 맞을 때만 싣는다. 아니면 싣지 않고 세션 상태와 경고로 남긴다.
   - 지금 전후 복원으로 만든 기록이다 (`current_localization`, 상태 `stale`이 아님). before 복원이 바뀌면 장면 좌표계가
     달라진다.
   - 그 기록을 만든 블러본이 이 내보내기가 쓰는 블러본이다 (기록의 `render_hash` = `ExportSession.render_hashes`,
     아니면 상태 `other_video`). 영상 프레임과 자세가 같은 영상에서 나왔음을 보장한다.
6. **세션마다 다른 좌표계.** 장면 좌표계는 그 세션의 before 복원이다. 세션끼리 공통 좌표가 아니고 중력 방향을
   모른다. 길이 단위는 그 복원이 마커로 미터가 됐을 때(`metric`)만 미터다. 미터가 아니면 손 3D(미터)를 장면 좌표로
   옮기면 안 된다고 meta에 적는다.
7. **LeRobot.** 특징 `observation.scene_camera` (float32[8]: 값 7개 + 끝의 `scene_camera.present`). `write_package`가
   사람 모드·로봇 모드 특징 위에 덧붙인다 (촉각과 같다).
8. **GR00T.** `observation.state` 끝(촉각 구간 뒤)에 `scene_camera` 구간 7칸, 표시 열
   `observation.scene_camera_valid`(float32[1]). action에는 넣지 않는다 (관측이지 행동이 아니다). 확인기
   (`gr00t_check` 8번)가 촉각과 같은 조건을 본다: 구간이 있고 길이·이름이 meta와 같으며 action에 없고, 표시는 0·1,
   표시 0인 행의 값은 0.
9. **기록.** `meta/dlp_scene.json`과 manifest `details.scene`이 같은 내용이다: 형식과 판, 값의 뜻(사원수 순서, 좌표
   변환식, 축), 좌표계, 단위, 시간 맞춤, 없음 표현, 출처(사람 검수 없는 모델 출력), 보간 한도, 값 이름, 세션마다(가명)
   상태(`exported`·`no_localization`·`stale`·`other_video`)와 실었으면 미터 여부, 위치 추정 모델 버전, 프레임 수, 자세가
   있는 프레임 수, 위치 기록의 프레임 수.
10. **넣지 않는 것.** 복원·위치 기록 태그, 렌더 해시, 예외 문장(내부 세션 ID가 든다). 태그와 해시는 세션마다 고정된
    값이라 내보내기마다 다른 가명(ADR 0027)을 넘어 같은 세션을 잇게 된다. 경고는 가명으로 따로 쓴다.
11. **켜고 끄기와 불변.** `export.yaml scene.enabled`(기본 true). 끄면 위치 기록을 찾지도 않는다. 위치 기록이 있는
    세션이 없으면 결과물은 이 기능 전과 같다 (특징·열·파일·manifest 키를 더하지 않는다). 기록은 있는데 실은 세션이
    없으면(모두 낡음) 열은 그대로 두고 기록 파일만 더한다.
12. **읽는 곳.** 위치 기록은 라벨링 버킷에 있다 (`dlp_scene3d.localize.stored_localization`·`current_localization`).
    원본 버킷을 읽지 않는다. 그래서 `export → scene3d` 의존이 생겼다 (`scene3d`는 `export`를 쓰지 않아 순환이 없다).

## 대응표

| 위치 | 값 | 정하는 곳 |
| --- | --- | --- |
| LeRobot `observation.scene_camera` float32[8] | x, y, z, qw, qx, qy, qz, present (없으면 값 0·present 0) | `SceneData.lerobot_features`·`lerobot_arrays` |
| GR00T `observation.state`의 `scene_camera` 구간 (촉각 구간 뒤) | 값 7개 (없으면 0) | `SceneData.gr00t_modality`·`gr00t_arrays` |
| GR00T `observation.scene_camera_valid` float32[1] | 1 있음, 0 없음 | 같은 곳 |
| `meta/dlp_scene.json`, manifest `details.scene` | 뜻·좌표계·단위·세션별 상태 | `SceneData.meta` |
| manifest `details.warnings` | 낡은 기록, 다른 블러본의 기록, 자세가 하나도 없음 | `SceneCollector` |
| `export.yaml scene` | `enabled` | `ScenePolicy` |
| 보간 한도 | `scene3d.yaml interpolation.max_gap_ms` | `runner.scene_source` |

## 지금 되는 것과 아닌 것

합성 정답으로 확인한 것 (`packages/export/tests/test_scene.py`): 카메라 중심이 선형으로 움직이고 회전이 한 축 둘레로
일정하게 도는 움직임을 위치 기록에 200 ms마다 적고 한 구간을 비웠을 때, 내보낸 값이 그 프레임 시각의 정답 자세와 1e-5
안에서 같고, 그 값으로 카메라 좌표 점을 옮기면 처음의 장면 점이다. 기록 앞·뒤와 비운 구간은 값 0·표시 0이다. 낡은 기록,
다른 블러본의 기록, 기록 없음은 싣지 않고 상태·경고로 남으며 결과에 내부 ID·태그·렌더 해시가 없다. 위치 기록이 없으면
장면 기록이 없다. LeRobot 묶음과 GR00T 데이터셋(촉각 뒤 구간, 확인기 통과, 깨뜨린 데이터셋 거부).

아닌 것과 모르는 것 (F-COE-100, F-COE-101):

- LeRobot 공식 쓰기·읽기 API와 GR00T 자체 로더로 이 특징을 읽어 보지 않았다 (촉각과 같은 사정, F-COE-97).
- 위치 기록을 데이터셋 버전에 고정하지 않는다. 내보낼 때 라벨링 버킷의 지금 기록을 읽고, 쓴 기록의 태그는 결과에
  남기지 않는다 (가명 규칙). 다시 추정하면 같은 데이터셋 버전에서 다른 값이 나올 수 있다.
- 장면 자체(희소 점, 가우시안 스플랫)는 내보내지 않는다. 받는 쪽은 세션마다의 장면 좌표계 자세만 받는다. 집 안 3D
  복원을 납품물에 넣을지는 개인정보·제품 결정이라 소유자가 정한다.
- 손 3D를 장면 좌표로 옮긴 값은 싣지 않는다 (받는 쪽이 위 식으로 옮긴다). 장면이 미터가 아닌 세션은 옮길 수 없다.
- 위치 기록 자체의 한계(실영상 미검증, 프레임마다 따로 풂, 세션당 공간 하나)는 ADR 0067·F-PAR-93~99 그대로다.

## 결과

- 운영: `dlp export lerobot|gr00t <버전> --target <받는 곳>`이 쓸 수 있는 위치 기록이 있는 세션의 자세를 함께 싣고,
  끝에 세션 상태별 수와 자세가 있는 프레임 비율을 한 줄로 알린다. 끄려면 `export.yaml scene.enabled: false`.
- 기존 형식의 결과: 위치 기록이 없는 세션만 내보내면 바뀌지 않는다. 있으면 LeRobot 특징 하나, GR00T state 7칸과
  표시 열 하나가 늘어난다.
- 시험: `packages/export/tests/test_scene.py`, `packages/cli/tests/test_export_cmds.py`(요약 줄).
