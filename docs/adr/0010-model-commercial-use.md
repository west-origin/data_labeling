# ADR 0010: 모델의 상업 사용 분류

- 상태: 채택
- 날짜: 2026-10-07
- 관련: ADR 0009. 라벨링한 데이터를 판매한다 (프로그램은 판매·배포하지 않는다).

## 배경

라벨링한 데이터를 파는 것도 상업 사용이다. 소프트웨어 라이선스(GPL 등)의 의무는 주로 프로그램을
배포할 때 생기므로 사내에서만 돌리면 거의 상관없다 (AGPL은 외부 사용자에게 네트워크로 제공하면
예외). 반면 모델 가중치·학습 데이터의 "비상업" 조건은 사용 목적을 제한하므로 프리라벨에 쓰는 것
자체가 해당된다.

공개 모델 대부분은 가중치를 Apache·MIT로 배포하지만, 학습 데이터에 비상업·연구용 데이터셋이
섞여 있다 (WIDER FACE, Objects365, CrowdPose 등). 이 경우의 법적 해석은 정리되지 않았다.

## 결정

1. **`config/models.yaml`의 모든 모델에 가중치 라이선스, 직접 학습·미세조정 데이터와 그 라이선스,
   상업 사용 분류를 적는다.**
   - `allowed`: 가중치가 상업 사용을 허용하고 직접 학습 데이터에 비상업 조건이 없다.
   - `review`: 가중치는 허용하지만 직접 학습 데이터에 비상업·연구용 조건이 있다. 판매 전 법무 확인 대상.
   - `forbidden`: 가중치 자체가 비상업이다 (예: Depth Anything V2 Base·Large).
   교사 모델 등 간접 계보는 적기만 하고 분류에는 넣지 않는다.
2. **정책이 쓰는 모델은 `accept`에 든 분류여야 한다.** `make check`(contracts)가 `dlp models licenses`로
   검사한다. 지금은 `[allowed, review]`다. 법무 확인 결과에 따라 `review`를 빼면 해당 모델을 쓰는
   정책이 검사에서 걸려 대체 모델로 바꿔야 한다.
3. **대체 비용이 작은 `review` 모델은 바로 바꾼다.** RTMPose 앞단 사람 탐지기를 YOLOX-m Human-Art
   (Human-Art: CC BY-NC-SA 4.0)에서 Megvii 공식 YOLOX-m COCO로 바꿨다. 실사 사진 4장(사람 16명)에서
   15명 대 16명을 찾았고 박스 위치도 거의 같았다 (놓친 1명은 20 px 크기의 배경 인물).
4. **`dlp models licenses`(`make licenses`)의 표를 판매 실사 자료로 쓴다.** 라벨마다 `model_version`이
   남으므로 어떤 데이터가 어떤 모델의 프리라벨을 거쳤는지 계보로 답할 수 있다.

## 현재 분류 (정책이 쓰는 모델)

| 모델 | 용도 | 분류 | 이유 |
| --- | --- | --- | --- |
| YuNet | 얼굴 블러 | review | WIDER FACE (CC BY-NC-ND 4.0) |
| MediaPipe 손 | 손 21관절 | allowed | Google 자체 데이터 |
| EfficientDet-Lite0 | COCO 객체 | review | COCO 이미지 (Flickr 개별 CC, CC BY-NC 계열 포함) |
| YOLOX-m COCO | 사람 박스 | review | COCO 이미지 (Flickr 개별 CC, CC BY-NC 계열 포함) |
| RTMPose-m Body7 | 전신 17점 | review | CrowdPose 등 연구용 데이터 포함 |
| OWLv2 | 도구·블러 대상 | review | Objects365 (비상업) 미세조정 |
| Depth Anything V2 Metric Small | 3D | review | 인코더 의사 라벨 학습 이미지 (ImageNet-21K, SA-1B 등 연구용) |

COCO 학습 모델 주: COCO 주석은 CC BY 4.0이지만 이미지는 Flickr 사진마다 다른 CC 라이선스이고
CC BY-NC 계열이 섞여 있다. 직접 학습 데이터에 비상업 조건이 있으므로 `review`다 (감사 4차, ADR 0026).
COCO가 든 모델의 학습 데이터 표기는 모두 "주석 CC BY 4.0, 이미지는 Flickr 개별 CC 라이선스로 CC BY-NC
계열 포함"으로 통일한다.

## review를 허용하지 않기로 할 때의 대안 (성능 하락 있음)

- 얼굴: MediaPipe BlazeFace (Google 자체 데이터). 근거리(셀카) 모델이라 멀리 있는 얼굴 재현율이
  떨어진다. 블러 누락은 사고이므로 전수 검수 비율을 올려야 한다.
- 전신: MediaPipe Pose (Google 자체 데이터). 여러 사람·가림에서 RTMPose보다 약하다.
- 오픈 보캐뷸러리: Florence-2 (MIT, Microsoft 자체 구축 데이터). 탐지 품질과 속도는 따로 검증해야 한다.
