# 카메라 내부 파라미터·작업자 손 크기 등록 절차 (운영자용)

기준: ADR 0045. 손가락 끝 3D(ADR 0044)는 바디캠 내부 파라미터(초점 거리·주점)와 손 크기 사전값으로
거리를 정한다. 둘 다 없으면 가로 화각 90도 근사와 성인 평균 손바닥 0.09 m를 쓰고, 3D 표본 신뢰도가
낮게(`intrinsics=hfov_approx`) 기록된다. 이 문서는 기기마다 한 번, 작업자마다 한 번 하는 등록 절차다.

## 1. 기기 ID 정하기

- 바디캠 한 대에 ID 하나 (예: 일련번호 `gopro-c3471325`). 영숫자로 시작, 영숫자와 `_ . : -`, 128자 이하.
- 같은 기기라도 해상도·화각 모드(예: Wide/Linear)·렌즈 모듈을 바꾸면 내부 파라미터가 달라진다. 촬영
  모드는 세션 내내 고정하고, 모드가 다르면 다른 ID로 관리한다 (예: `gopro-c3471325-linear-4k`).
- 수집 매니페스트에 기기 ID를 적는다. 내부 파라미터를 직접 적었으면(`calibration.intrinsics`) 그 값이
  등록보다 앞선다.

```yaml
calibration:
  camera_id: gopro-c3471325-linear-4k
```

## 2. 체커보드 영상 찍기

준비물: 평평한 판에 붙인 체커보드 인쇄물. 안쪽 코너 수(예: 9x6 칸이면 안쪽 코너 8x5)와 칸 한 변을
자로 잰다 (mm, 인쇄 배율 오차가 그대로 들어간다).

1. 세션과 같은 촬영 모드·해상도로 찍는다. 손떨림 보정(전자식 안정화)은 끈다 (프레임마다 내부
   파라미터가 바뀐다). 세션도 같은 설정이어야 한다.
2. 보드만 찍는다. 사람 얼굴·서류·화면이 들어오지 않게 한다 (이 영상은 원본 버킷에 올리지 않고 로컬
   파일로 다룬다).
3. 보드를 화면 가운데·네 모서리·가장자리로 옮기고, 앞뒤·좌우로 20~40도 기울인 자세를 15개 이상
   찍는다. 자세마다 0.5초 이상 멈춘다 (움직임 흐림 방지). 보드가 화면의 1/4 이상을 채우게 가깝게.
4. 조명은 고르게, 반사(광택 인쇄)를 피한다.

## 3. 등록

```bash
# 결과만 보기 (DB에 쓰지 않음)
dlp calib intrinsics gopro-c3471325-linear-4k board.mp4 --board 8x5 --square-mm 30 --dry-run
# 등록
dlp calib intrinsics gopro-c3471325-linear-4k board.mp4 --board 8x5 --square-mm 30
```

- 종료 코드 2: 보드 장면 부족(`calibration.yaml intrinsics.min_views`) 또는 재투영 RMS 초과
  (`max_rms_px`). 코너 수를 다시 세고, 자세를 늘려 다시 찍는다.
- 출력의 fx·fy가 서로 1~2% 안이고 cx·cy가 화면 가운데 근처인지 본다. 재투영 RMS는 보통 0.2~0.5 px다.
- 같은 결과를 다시 등록하면 쓰지 않는다 (멱등).
- 제조사·다른 도구로 이미 아는 값은 파일로 가져온다: `dlp calib import <기기> cam.yaml`
  (이 저장소 형식 `width, height, fx, fy, cx, cy, distortion`, OpenCV FileStorage, ROS camera_info).
- 렌즈를 바꾸거나 다시 재면 `--valid-from 2026-11-01T00:00:00+09:00`으로 새로 등록한다. 그 시각
  이후 녹화한 세션만 새 값을 쓰고, 예전 세션은 예전 값을 쓴다. 등록은 지우거나 고치지 않는다 (추가
  전용 DB 표 `camera_calibrations`).
- 확인: `dlp calib show --camera gopro-c3471325-linear-4k` (`*`가 지금 시각 기준으로 쓰이는 값).

## 4. 작업자 손 크기

손목 주름 가운데에서 가운데 손가락 뿌리 관절(손등 쪽 너클)까지를 자로 잰다 (m, 보통 0.075~0.10).

```bash
dlp calib palm w01 0.084
dlp calib show --worker w01
```

잰 값이 없으면 캘리브레이션된 세션에서 추정할 수 있다 (깊이 모델 가중치 필요, `make export-models`).
원본 바디캠을 읽으므로 감사 기록 `calib.palm_estimate`이 남는다.

```bash
dlp calib palm-estimate w01            # 보고만 (표본 수·세션 수·중앙값)
dlp calib palm-estimate w01 --apply    # estimated로 등록
```

- 표본이 `prelabel.yaml palm_estimate.min_samples`보다 적으면 종료 코드 3 (등록하지 않음).
- 추정값은 깊이 모델의 배율 오차를 그대로 담는다. 잰 값이 있으면 잰 값을 쓴다 (나중 등록이 앞선다).
- 작업자 ID는 DB에만 남고 내보내기에는 들어가지 않는다. 3D 모델 버전에는 사전값 출처
  글자(`+hpw` 작업자, `+hpp` 정책)만 남는다.

## 5. 반영

등록 뒤 세션마다 프리라벨을 다시 돌린다. 3D 단계 입력 해시가 바뀌어 검수 전 3D만 새 값으로 다시
만든다. 승인·표본 검증한 3D는 그대로 남고, 같은 (개체, 부위)의 새 궤적은 만들지 않는다.

```bash
dlp prelabel run <세션>
```

새 3D 라벨의 페이로드 `intrinsics`가 `calibrated`, 표본의 `method`가 `hand_fit`이면 반영된 것이다.

## 한계

- 핀홀 + OpenCV 왜곡 모델만 다룬다. 어안 렌즈(초광각 모드)는 `cv2.fisheye`가 필요하다. 왜곡 계수는
  등록만 하고 3D 단계는 아직 쓰지 않는다 (F-PAR-28). 광각 모드에서는 화면 가장자리 손의 3D가 틀릴 수
  있다.
- 카메라-IMU 외부 파라미터(SLAM용)는 이 절차에 없다 (F-PAR-26).
