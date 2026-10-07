# ADR 0016: 재학습 루프와 모델 레지스트리

- 상태: 채택
- 날짜: 2026-10-07
- 관련: WP13. 계약 변경: ModelVersion·ModelStatus (Alembic 0007 `model_versions`, `make schemas`),
  평가 과제 privacy 추가 (config/policies/evaluation.yaml)

## 결정

1. **학습 데이터는 데이터셋 버전 스냅샷에서만 뽑는다** (`dlp_train.extract`). 버전의 분할 중
   `config/policies/training.yaml splits`(학습·검증)만 쓰고, 골든·holdout은 정책 검증기가 막는다. 학습 세션과
   골든셋 세션이 겹치면 학습을 시작하지 않는다. 운영 라벨(`current_labels`)만 쓰므로 오류 삽입·측정 레코드와
   그 후손은 들어가지 않는다. 미검수 모델 라벨은 예제가 아니다.
2. **예제마다 자동 원본과의 차이를 붙인다**: accepted(그대로 승인·표본 검증), corrected(사람이 고침,
   origin=처음 모델 레코드), added(모델이 놓침), deleted(사람이 지운 모델 라벨 = 오탐). 모델 버전이 바뀌어
   지운 삭제 레코드(출처 model)는 사람의 삭제가 아니므로 예제가 아니다.
3. **누적 조건**: 예제 수가 `min_examples` 이상이고 직전 학습보다 `min_new_examples` 이상 늘었을 때만
   학습한다 (`--force`로 무시).
4. **평가는 메모리에서 한다.** 후보와 기존 모델(배포 중인 재학습 모델, 없으면 `--baseline-version`의 DB
   예측)을 골든셋 세션 영상에 돌려 같은 하네스(WP11)로 평가하고 같은 게이트로 판정한다. 골든셋 세션의
   라벨 이력에는 아무것도 쓰지 않는다 (후보 예측이 운영 라벨이 되지 않게).
5. **레지스트리의 기준은 DB(`model_versions`)다.** 상태는 candidate → passed(사람 승인 대기) → deployed →
   retired, 또는 rejected. 과제마다 deployed는 하나이고, 배포하면 이전 배포는 retired가 된다. 산출물은 학습
   산출물 버킷(`buckets.mlflow`)에 sha256과 함께 두고, 읽을 때 해시가 다르면 쓰지 않는다. 학습 실행은
   `training_runs`(계보)에 남아 사용 중지 전파가 영향받은 모델을 찾을 수 있다.
6. **MLflow는 기록과 사본이다.** 실행마다 파라미터, 학습 지표, 골든 지표(`golden/*`), 게이트 결과,
   산출물(모델·리포트)을 남기고, 배포한 모델만 MLflow 모델 레지스트리(`dlp-<과제>`, 별칭 `deployed`)에
   올린다. REST API만 써서 mlflow 패키지에 의존하지 않는다.
7. **배포 방식은 과제별 정책이다** (`deploy: auto | approve`). 프라이버시(블러)는 사람 승인이 있어야 배포한다
   (`dlp train approve`). 배포된 모델은 단계 CLI가 읽는다: 프리라벨은 `replaces`의 기본 어댑터를 빼고
   재학습 모델을 넣고, 프라이버시는 탐지기 결과와 합집합으로 쓴다 (블러는 놓치면 안 되므로 빼지 않는다).
   로더가 쓸 수 없거나 해시가 다르면 기본 어댑터를 쓰고 이유를 출력한다.
8. **평가 과제 privacy 추가**: 정답 블러 박스 면적의 `coverage` 이상이 예측 블러로 덮이면 재현,
   예측 박스 면적의 `precision_overlap` 이상이 정답 위면 정밀. 주 지표는 재현이고 하위 집단에서도 하락을
   허용하지 않는다.

## 학습기

- 학습기(Trainer)와 로더(ModelLoader)는 이름으로 짝지어 등록한다. 레지스트리의 `trainer`로 로더를 찾는다.
- CI는 `oracle-stub`만 쓴다: 학습 예제에서 본 클래스만 기억하고 예측 때 정답을 흔들어 돌려준다. 정답이
  필요하므로 운영에서는 로드되지 않는다(기본 어댑터 유지).
- 실제 학습기(객체 YOLOX, 손·전신 RTMPose 미세조정, 접촉 영상 분류기, 블러 탐지기 미세조정)는
  `TODO(real-model)`이며 GPU 학습 서버가 필요하다. 미세조정 출발 가중치의 상업 사용 분류는
  `config/models.yaml`(ADR 0010)을 따른다.

## 결과

- `dlp train run <과제> <데이터셋 버전>`, `dlp train models`, `dlp train approve <모델 버전>`.
- 통합 테스트(`packages/training/tests/test_loop.py`): 누적 부족 시 건너뜀 → 나쁜 후보 미배포 → 새 예제 없으면
  건너뜀 → 좋은 후보 배포 → 배포 모델보다 나쁜 후보 미배포(기존 유지) → 블러는 승인 후 배포 → 프리라벨
  단계에 배포 모델 적용(해시가 다르면 거부).

## 갱신 (ADR 0019)

2차 전체 검수 정정이 이 결정을 보강한다. 바뀐 내용은 ADR 0019를 따른다.
