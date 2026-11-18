# gsplat-env: SfM·프레임 특징·3D 가우시안 스플래팅 격리 환경 (ADR 0066, 0067)

`dlp scene3d reconstruct`·`localize`의 실제 백엔드(`dlp_scene3d.backends.IsolatedSfm`, `GsplatTrainer`,
`IsolatedFrameFeatures`)가 `uv run --project scripts/gsplat-env python <스크립트>`로 부른다. 주고받는 것은
파일뿐이다.

- `run_sfm.py --images DIR --out DIR`: pycolmap(BSD-3) 특징 추출·전수 대응·증분 SfM.
  `cameras.json`(`dlp_schema.scene3d.CameraPose` 목록)과 `points.npz`(points, descriptors = 트랙 SIFT
  기술자 평균, colors)를 쓴다. 척도는 모른다(metric=False). 미터 척도는 복원 단계가 이 결과에 마커 척도를
  곱해 정한다 (`dlp_scene3d.markers`, ADR 0067).
- `extract_frame_features.py --images DIR --out NPZ --max-features N`: 바디캠 블러본 프레임
  (`<프레임 시각 ms>.png`)의 pycolmap SIFT 특징(위치, 128차원 기술자). SfM과 같은 추출기라 복원 점 기술자와
  대응된다 (ADR 0067).
- `train_splat.py --images DIR --sfm DIR --iterations N --out PLY`: gsplat(Apache-2.0) 래스터라이저로
  SfM 점에서 시작해 가우시안을 학습하고 표준 3DGS PLY를 쓴다 (CUDA 필요).

입력은 언제나 승인된 블러본이다 (사진은 `privacy_gate.open_blurred`, 영상 프레임은
`ego_video.open_blurred_video`). 원본을 넣지 않는다.

화소 좌표 규약: 스크립트가 쓰고 읽는 주점·특징 위치는 COLMAP·gsplat 규약(왼쪽 위 화소의 중심이
(0.5, 0.5))이다. 작업공간 안은 OpenCV 규약(중심이 (0, 0))이라 어댑터가 경계에서 0.5를 빼고 더한다.

## 상태 (2026-11-18)

- 잠금 파일(`uv.lock`)이 없다. 개발 컨테이너에 PyTorch CUDA판을 받지 않았다. GPU 서버에서 이 디렉터리의
  `uv lock` 뒤 `uv sync --locked`로 만든다.
- `run_sfm.py`와 `extract_frame_features.py`는 **CPU판 pycolmap 4.2.1만 깐 임시 환경**에서 돌려 봤다
  (이 환경 정의 그대로는 아니다: torch·gsplat 없이, 잠금 없이). 입력은 무늬가 있는 합성 방을 광선 추적으로
  그린 그림이다 (실제 사진·영상 아님). 확인한 것:
  - 16장 SfM: 16/16 등록, 점 약 7,300개, 카메라 중심이 정답과 0.15 cm(닮음 변환 뒤 RMS).
  - 마커 척도: 정답 척도와 0.1% 안. 전·후를 따로 복원해 기술자 대응으로 정렬: 내점 5,391/5,460, 정답과
    점 RMS 0.55 cm.
  - 프레임 특징 → 대응 → PnP: 24/24 프레임, 위치 중앙값 0.18 cm, 회전 0.03도 이내.
  - 화소 규약: 중심을 아는 원반의 특징 위치가 (중심 + 0.5), 어댑터를 거친 주점이 정답과 같음.
  - 이때 `run_sfm.py`의 pycolmap 4.x API 오류 둘을 고쳤다 (`Database`는 `open()`으로 열고,
    `read_descriptors`의 배열은 `.data`).
- `train_splat.py`는 실행해 보지 않았다 (TODO(real-model), CUDA 필요).
- 알려진 빈 곳:
  - 렌즈 왜곡: PINHOLE 카메라로 고정했다. 휴대폰 사진은 SIMPLE_RADIAL로 풀고 왜곡을 편 뒤 넘기는 편이 낫다.
    왜곡이 남으면 마커 모서리 삼각측량과 척도에도 오차가 실린다.
  - 사진마다 카메라를 따로 둔다 (`camera_mode` 기본값). 같은 기기로 찍은 사진은 한 카메라로 묶는 편이 낫다.
  - 밀도 조절(densification)·SH 고차 계수가 없는 최소 학습 루프다. gsplat의 `DefaultStrategy`로 바꾼다.
  - GPU SIFT와 CPU SIFT의 기술자가 얼마나 다른지 재지 않았다. SfM과 프레임 특징은 같은 환경에서 뽑는다.
