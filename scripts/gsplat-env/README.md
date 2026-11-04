# gsplat-env: SfM·3D 가우시안 스플래팅 격리 환경 (ADR 0066)

`dlp scene3d reconstruct`의 실제 백엔드(`dlp_scene3d.backends.IsolatedSfm`, `GsplatTrainer`)가
`uv run --project scripts/gsplat-env python <스크립트>`로 부른다. 주고받는 것은 파일뿐이다.

- `run_sfm.py --images DIR --out DIR`: pycolmap(BSD-3) 특징 추출·전수 대응·증분 SfM.
  `cameras.json`(`dlp_schema.scene3d.CameraPose` 목록)과 `points.npz`(points, descriptors = 트랙 SIFT
  기술자 평균, colors)를 쓴다. 척도는 모른다(metric=False).
- `train_splat.py --images DIR --sfm DIR --iterations N --out PLY`: gsplat(Apache-2.0) 래스터라이저로
  SfM 점에서 시작해 가우시안을 학습하고 표준 3DGS PLY를 쓴다 (CUDA 필요).

입력 사진은 언제나 승인된 블러 PNG다 (`privacy_gate.open_blurred`). 원본을 넣지 않는다.

## 상태 (2026-11-04)

- 잠금 파일(`uv.lock`)이 없다. 개발 컨테이너 디스크가 약 3.5 GB 남아 PyTorch CUDA판을 받을 수 없었다.
  GPU 서버에서 이 디렉터리의 `uv lock` 뒤 `uv sync --locked`로 만든다.
- 두 스크립트는 실행해 보지 않았다 (TODO(real-model)). 알려진 빈 곳:
  - 렌즈 왜곡: PINHOLE 카메라로 고정했다. 휴대폰 사진은 SIMPLE_RADIAL로 풀고 왜곡을 편 뒤 넘기는 편이 낫다.
  - 밀도 조절(densification)·SH 고차 계수가 없는 최소 학습 루프다. gsplat의 `DefaultStrategy`로 바꾼다.
  - 척도 기준(크기를 아는 마커)이 없어 before 복원이 미터가 아니면 `compare`가 거부한다
    (`scene3d.yaml change.require_metric`).
