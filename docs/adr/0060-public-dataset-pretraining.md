# ADR 0060: 공개 데이터셋으로 프리라벨 모델 1차 학습

- 상태: 제안 (라이선스 표 확인 뒤 채택, 구현 전)
- 날짜: 2026-10-26
- 관련: ADR 0010(상업 사용 분류), ADR 0016·0025(재학습 루프), ADR 0057(학습 증강), ADR 0058(사내 모델
  개발 용도), ADR 0020(원본 접근), F-PAR-60(도구 오탐), F-PAR-64(손 끊김), F-PMS-31(블러 오탐).
- 계약 변경: 없음 (라벨·DB·내보내기 그대로). 새 설정 `config/external_datasets.yaml`, `models.yaml` 항목
  필드 `training_datasets`(선택), 격리 학습 환경 `scripts/pretrain-env/`.

## 배경

재학습 루프에는 oracle-stub 학습기만 있고(`dlp_train.trainers`, TODO(real-model)), 우리 데이터는 아직
검수된 양이 적다. 공개 데이터셋으로 먼저 학습한 가중치를 출발점으로 쓰면 적은 우리 데이터로도 다듬을 수
있다. 그러나 공개 데이터셋은 라이선스가 제각각이고(1인칭 손·행동은 대부분 비상업), 우리 데이터와 섞이면
계보·분할·내보내기 규칙이 깨진다. 이 컨테이너와 사내에는 아직 GPU가 없다 (DGX Spark는 ADR 0040에서
VLM 서버로 정했고 실측 전).

## 결정

1. **2단계로 나눈다.** 1단계: 공개 데이터셋만으로 가중치를 학습한다 (파이프라인 밖, 격리 환경). 2단계:
   그 가중치를 출발점으로 `dlp train`이 데이터셋 버전의 학습·검증 분할(우리 검수 데이터)로 다듬는다.
   공개 데이터는 라벨 이력·데이터셋 버전·골든셋·내보내기·증강 입력에 절대 들어가지 않는다
   (`label_records`에 쓰지 않는다, 라벨링·원본 버킷에 두지 않는다).
2. **공개 데이터셋 목록을 설정으로 둔다.** `config/external_datasets.yaml`: 이름, 판(버전), 받는 곳,
   주석·매체 라이선스, 약관 요지, 신청·서명 필요 여부, 상업 사용 분류(`allowed | review | forbidden`,
   ADR 0010과 같은 뜻), 쓰는 과제, 받은 파일 목록의 sha256. `forbidden`(약관이 학습 자체를 막거나
   파생 모델 사용을 막음)은 학습에 쓰지 않는다.
3. **가중치 분류는 데이터에서 계산한다.** 1단계 산출물은 `models.yaml`에 `training_datasets: [이름...]`로
   등록하고, `make licenses`가 `commercial`이 그 데이터셋 분류 중 가장 엄한 것 이상인지 검사한다
   (allowed < review < forbidden). 출발 가중치(예: YOLOX COCO, review)의 분류도 더한다. 2단계 모델은
   출발 가중치의 분류를 물려받는다 (기존 규칙).
4. **재현 기록.** 1단계 실행마다 데이터셋 이름·판·파일 목록 sha256, 고른 클래스·표본 수, 학습 설정,
   코드 커밋, 환경 잠금 파일 해시, 산출물 sha256을 `manifest.json`으로 산출물 옆에 둔다. MLflow 실험
   `pretrain/<과제>`. 가중치는 ONNX로 내보내 기존 Predictor 어댑터가 그대로 읽게 한다.
5. **배포 판단은 우리 골든셋으로만.** 공개 데이터 검증 점수는 참고다. 정책이 새 가중치를 쓰려면
   `dlp eval golden`으로 지금 모델과 비교해 `evaluation.yaml` 게이트를 통과해야 한다.
6. **계산 자원.** 1단계는 공개 데이터만 쓰므로 외부 GPU(Kaggle·Colab 등)를 써도 된다. 조건:
   - 우리 데이터(원본·블러본·라벨·골든셋·작업자 정보)는 올리지 않는다. 2단계는 사내 GPU에서만 한다.
   - 노트북·데이터는 비공개로 둔다 (Kaggle 노트북 기본값 확인). 저장소 비밀값을 넣지 않는다.
   - 환경은 `scripts/pretrain-env/uv.lock`에서 내보낸 고정 목록으로 설치하고, 실행 기록(4절)을 남긴다.
   - 중간 체크포인트는 세션이 끊겨도 이어서 하도록 외부 저장소(Google Drive 등)에 둔다.
   - 서비스 약관(무료 계정의 사용 목적 제한)을 쓰기 전에 확인한다.
7. **학습 코드 라이선스.** 상업 사용이 막히거나 가중치에 조건이 붙는 학습 코드(예: AGPL-3.0인 Ultralytics
   YOLO)는 쓰지 않는다. 객체는 지금 쓰는 YOLOX(Apache-2.0)를 미세조정한다.
8. **우선순위.** (1) 도구·가사 객체 고정 클래스 탐지기: Open Images 등 상업 사용이 무난한 데이터에서
   부엌·청소·돌봄 도구 클래스를 골라 YOLOX를 미세조정한다 (OWLv2 오탐, F-PAR-60). (2) 손: 1인칭 손
   데이터는 대부분 비상업이라 법무 판단 전에는 우리 검수 데이터로 다듬는다. (3) 얼굴 블러: 우리 검수 블러
   라벨로 다듬는다 (F-PMS-31).

## 결과

- 공개 데이터와 우리 데이터의 경계가 설정·검사로 고정된다. 라벨·내보내기 쪽 코드는 바뀌지 않는다.
- 외부 GPU로 1단계를 바로 시작할 수 있다. 2단계와 실제 학습기(`dlp_train.trainers`)는 사내 GPU가 생겨야 한다.
- 남은 일: `external_datasets.yaml`·검사·`scripts/pretrain-env` 구현,
  Open Images 클래스 대응표(온톨로지 ID ↔ Open Images 클래스), 1단계 노트북.

## 라이선스 표

원문 확인 결과와 법무 질문은 [public-datasets.md](../handover/public-datasets.md), 설정은
`config/external_datasets.yaml`(`make licenses` 표). 요약: allowed는 Open Images V7·HoloAssist와 서명 계약인
Ego4D·Ego-Exo4D, forbidden은 ARCTIC, 나머지 1인칭 손·행동 학계 데이터는 review(비상업·학술 한정)다.
