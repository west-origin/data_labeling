# ADR 0066: 청소 전후 다각도 사진의 3D 복원과 물체 이동 비교

- 상태: 채택 (CPU stub으로 종단 검증, 실제 SfM·스플랫은 GPU 실측 전)
- 날짜: 2026-11-04
- 관련: ADR 0005·0024(프라이버시 게이트), 0010·0058(상업 사용 분류), 0015(라벨 불변), 0020·0021(원본 접근 감사), 0062(격리 환경)

## 배경

현장에서 청소 전과 후에 방을 여러 각도에서 사진으로 찍는다. 이 사진으로 방의 3D 공간 정보(카메라 자세,
희소 점, 3D 가우시안 스플랫)를 얻고, 전후를 비교해 **어떤 물체가 어디로 얼마나 움직였는지**(치워진 것,
새로 생긴 것 포함)를 알고 싶다. 사진에는 얼굴·문서·화면이 찍힌다. 영상과 같은 개인정보·라이선스·이력
규칙을 지켜야 한다.

## 결정

### 1. 새 패키지와 명령

`packages/scene3d` (`dlp_scene3d`), 명령 `dlp scene3d …`(`dlp_cli.scene3d_cmds`), 정책
`config/policies/scene3d.yaml`. 의존: media, models, privacy, schema. 흐름:

```
ingest(before, after) → privacy detect → privacy review → privacy approve → privacy render
→ reconstruct(before, after) → compare → report
```

세션 하나가 청소 작업 하나이고, 단계(phase)는 `before`·`after` 둘이다. 사진만 찍은 현장도 있어
`sessions` 표 등록을 요구하지 않는다.

### 2. 수집: 원본은 감사 원본 저장소에 불변으로

- 원본 사진 바이트를 그대로 원본 버킷 `sessions/<세션>/scene3d/<단계>/photos/<photo_id><확장자>`에 둔다
  (`dlp_media.storage.put_immutable`). photo_id = `p` + 원본 sha256 앞 16자라 재수집이 멱등이다.
- 원본 버킷은 `dlp_cli.raw_access.raw_store`(DB 감사) 또는 `raw_store_offline`(로컬 JSON Lines)로만 연다.
  서비스 목적 `scene3d.ingest`·`scene3d.privacy-detect`·`scene3d.privacy-render`를 `ops.yaml`에 넣었다.
- EXIF 방향은 원본을 고치지 않고 읽을 때 적용한다 (`photos.decode_photo`, PIL `exif_transpose`). 모든
  픽셀 좌표는 방향 적용 뒤 이미지 기준이다.
- 이미 수집한 단계에 다른 사진 묶음을 넣으면 거부한다 (승인·복원이 어느 사진에 대한 것인지 흐려진다).

### 3. 프라이버시: 사진 블러 게이트 (사람 승인 뒤 블러본만)

- 탐지는 `dlp_privacy`의 탐지기(`build_detectors`, `FrameDetector`)를 사진마다 한 번 부른다
  (`privacy_gate.FramePhotoDetector`). 대상·문턱·여유는 `privacy.yaml` 그대로다. 탐지를 거르거나 지우지
  않는다. 탐지기가 없는 대상은 `no_detector`로 남겨 검수자가 사진 전체를 보게 한다.
- 탐지 세대 버전 = `photo-privacy-1+<탐지기:버전>+p<탐지 정책 해시>`. 같은 세대면 건너뛴다. 새 세대를 쓰면
  라벨링 버킷의 렌더 표시를 무효로 바꾼다 (이전 승인으로 만든 블러본을 하류가 못 쓰게).
- 승인은 원본 접근 권한자(`review.yaml reviewers.privacy`, `dlp_review.roles.check_privacy_reviewers`)만
  한다. 검수 묶음(`privacy review`)은 원본 사진과 `boxes.json`을 검수자에게 주고(감사 기록의 행위자 =
  검수자), 검수자는 상자를 지우거나 그린 뒤 `privacy approve --boxes`로 **모든 사진**의 확정 상자를 낸다.
  표본 승인은 없다. 승인은 본 탐지 세대가 최신일 때만 받는다. 탐지 기록은 덮어쓰지 않고 승인은 추가만 한다.
- 렌더는 지금 승인의 상자로만 `dlp_privacy.render.apply_blur`(정책의 렌더 방식·블록 크기)를 입혀 PNG로
  라벨링 버킷에 쓴다. PNG로 다시 쓰므로 EXIF(위치·기기 정보)가 빠진다.
- **하류 단계(SfM, 스플랫, 물체 탐지, 정렬, 비교, 뷰어 산출물)는 `privacy_gate.open_blurred`로만 사진을
  받는다.** 렌더 표시가 유효하고 받은 파일 sha256이 렌더 기록과 같아야 하며, 아니면
  `RenderNotCurrentError`다. 복원·비교는 원본 버킷을 열지 않는다. 복원 기록은 쓴 렌더 태그를 담고,
  비교는 지금 렌더와 다르면 거부한다.
- 오프라인 경로의 승인·탐지 "최신" 표시는 원본 버킷의 표시 파일이고, 영상 게이트처럼 DB의 완료 세대로
  옮기는 것은 후속이다. 검수 화면은 지금 JSON 묶음이다. CVAT 이미지 작업 연동은 후속이다.

### 4. 복원: SfM(pycolmap)과 3D 가우시안 스플래팅(gsplat), 격리 환경

- 카메라 자세·희소 점은 COLMAP(pycolmap)으로, 스플랫 학습은 gsplat으로 한다. 둘 다 본 작업공간에 넣지 않고
  `scripts/gsplat-env`(PyTorch CUDA, pycolmap, gsplat)에서 하위 프로세스로 돌린다
  (`backends.IsolatedSfm`, `GsplatTrainer`). 주고받는 것은 파일(cameras.json, points.npz, PLY)뿐이다.
- CPU stub: `OracleSfm`(합성 정답 카메라·점), `InitPointsSplat`(SfM 점마다 등방 가우시안 하나를 둔 표준
  3DGS PLY, 학습 없음). 정책으로는 stub을 고를 수 없고 테스트와 `--oracle`로만 끼운다.
- 산출물은 라벨링 버킷 `…/recon/<태그>/{reconstruction.json,cameras.json,points.npz,splat.ply}`.
  태그 = (SfM·스플랫 버전, `reconstruction` 절 해시, 렌더 태그)의 해시. 같은 태그가 있으면 다시 계산하지
  않는다. 등록 사진 비율이 `min_registered_ratio` 미만이면 실패한다.

### 5. 정렬: 고정 구조의 닮음 변환, 확신이 낮으면 거부

- after 복원은 before와 다른 좌표계·척도다. 두 복원의 희소 점을 기술자 상호 최근접 + 비율 검사로 대응하고,
  RANSAC Umeyama로 척도 포함 닮음 변환 `x_b = s R x_a + t`를 찾는다 (`alignment.align_scenes`). 움직인 물체
  위의 점은 이상점으로 빠진다.
- 내점 수 `min_inliers`, 내점 비율 `min_inlier_ratio`, 잔차 RMS `max_rms_m` 중 하나라도 못 미치면 정렬을
  거부하고 비교하지 않는다 (종료 코드 3, 정렬 기록은 남긴다). 잘못된 정렬은 모든 물체를 "움직였다"로
  만들기 때문이다.
- 대안: before·after 블러 사진을 한 SfM으로 함께 복원하면 같은 좌표계에 놓인다. 그때도 같은 검사를 한다.
  실제 경로에서 어느 쪽을 기본으로 할지는 GPU 실측 뒤 정한다.

### 6. 물체와 변화

- 사진마다 물체를 탐지한다 (`ObjectDetector`: 실제 `OwlObjectDetector` = 기존 OWLv2 ONNX 가중치
  `owlv2_onnx`, 질의는 `scene3d.yaml objects.queries`의 온톨로지 objects 키 → 영어 질의. stub
  `OracleObjectDetector`).
- 같은 라벨 탐지 두 개(다른 사진)로 3D 가설을 세우고, 다른 사진에 다시 투영한 중심이 상자 대각선의
  `center_tolerance` 안에 드는 탐지를 받침으로 모은다. 받침이 많은 가설부터 받침 전체로 다시 삼각측량(DLT)해
  물체로 확정하고, 쓴 탐지는 다시 쓰지 않는다. 받침이 `min_views` 미만이면 버린다 (`objects.localize_objects`).
- 외형은 승인된 블러 사진의 상자 가운데 색 히스토그램(HSV H x S), 크기는 상자 대각선 x 깊이 / 초점거리.
- after 물체를 정렬 변환으로 before 좌표로 옮기고(크기에도 척도를 곱한다), 같은 라벨끼리 헝가리안으로
  대응한다. 외형 거리(Hellinger)·크기 비가 관문을 넘으면 대응하지 않는다. 대응 쌍은 이동 거리 ≤
  `unchanged_tolerance_m`이면 unchanged, 넘으면 moved. 남은 before는 removed, after는 added (`change`).
- 결과 `ChangeReport`(물체·변화·정렬)는 모델 출처, 미검수(`review_state = unreviewed`)다. 라벨링 버킷
  `…/changes/<태그>.json`에 한 번만 쓰고 `changes/current.json` 표시만 옮긴다. `dlp scene3d report`가
  JSON·CSV로 낸다. 검수 화면·사람 수정 연동은 후속이며, 그때도 수정은 새 기록으로 남긴다.
- 가려져 after에서 안 보인 물체도 removed로 나온다. 검수가 필요하다.

### 7. 단위·좌표계

- 길이 미터. 카메라는 COLMAP 규약 (`x_cam = R x_world + t`, x 오른쪽, y 아래, z 앞).
- 위치·이동 벡터는 **before 장면 좌표계**다. 중력 방향은 모른다 (축에 의미가 없다, "바닥 위 이동"으로
  나누려면 바닥 평면 추정이 필요하다, 후속).
- SfM만으로는 척도를 모른다. `Reconstruction.metric`이 False인 before로는 `change.require_metric`이면
  비교를 거부한다. 실제 경로는 크기를 아는 마커(ArUco 등)로 척도를 정해야 한다 (후속, TODO(real-model)).
  after는 정렬의 척도 s로 before 단위가 된다.

### 8. 계약과 DB

- 계약 `dlp_schema.scene3d`: `PhotoEntry`, `PhotoCapture`, `PhotoBox`, `PhotoPrivacyDetection`,
  `PhotoPrivacyApproval`, `PhotoRender`, `CameraPose`, `Reconstruction`, `Alignment`, `ObjectInstance`,
  `ObjectChange`, `ChangeReport`, `Scene3dRecord`. JSON Schema `schemas/scene3d_*.schema.json`.
- 추가 전용 표 `scene3d_records`(Alembic 0027, 수정·삭제·TRUNCATE 트리거): 산출물마다 (세션, 단계, 종류,
  모델 버전, 태그, 버킷, 키, 행위자, 시각) 한 줄. 같은 (종류, 태그, 키)는 다시 넣지 않는다. 원본 버킷 키는
  이 표에만 있고 라벨링 쪽 기록에는 없다. `--no-db`면 기록하지 않는다.

### 9. 라이선스 (`config/models.yaml external`, 2026-11-04 확인)

| 이름 | 라이선스 | 분류 | 근거 |
| --- | --- | --- | --- |
| colmap (pycolmap) | BSD-3-Clause | review | COLMAP README "new BSD license", PyPI pycolmap 4.2.1 BSD-3-Clause. 서드파티 의존성(GPU SIFT 등)은 별도 라이선스라 빌드 확인 전 review |
| gsplat | Apache-2.0 | allowed | nerfstudio-project/gsplat LICENSE |
| graphdeco-inria/gaussian-splatting | Gaussian-Splatting License (비상업) | forbidden | LICENSE.md "THE USER CANNOT USE ... FOR COMMERCIAL PURPOSES". 쓰지 않는다 |
| owlv2_onnx | Apache-2.0 | review | 기존 항목 (학습 데이터 Objects365 비상업) |

SfM·스플랫은 장면마다 최적화하는 도구라 직접 학습 데이터가 없다. `dlp models licenses`가 정책의 도구 이름을
검사하므로 정책이 Inria 코드를 고르면 `make check`가 실패한다.

## 지금 되는 것과 아닌 것

- CPU stub으로 종단 검증: 합성 방(`dlp_fixtures.scene3d`, 사진 14장씩, after는 알려진 닮음 변환 좌표계)에서
  이동 3개·그대로 2개·치움 1개·새로 둠 1개를 맞게 나누고 이동 벡터가 정답과 2 cm 안이다
  (`packages/scene3d/tests/test_pipeline.py`). CLI 오프라인 경로(`--store local: --no-db --oracle`)도 같다.
- GPU·실측 전: `scripts/gsplat-env`(잠금 파일 없음), `run_sfm.py`, `train_splat.py`(밀도 조절 없는 최소
  루프), 척도 마커, 렌즈 왜곡, 바닥 평면, 검수 화면 연동, DB 기반 탐지 세대.

## 결과

- 사진 기반 3D 비교가 영상과 같은 개인정보 약속(감사 원본 접근, 사람 승인 뒤 블러본만)을 따른다.
- 실제 모델 없이도 하류 로직(정렬, 삼각측량, 대응, 분류)을 정답으로 시험할 수 있다.
- 다음 담당자는 GPU 서버에서 격리 환경을 잠그고 실측해, stub과 같은 계약으로 실제 결과를 내게 한다.
