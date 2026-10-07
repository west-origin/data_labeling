"""재학습 루프 (WP13).

검수 승인 → 데이터셋 버전 → (누적 시) 학습 → 골든셋 평가 → 게이트 통과 시 배포.

관련 문서: docs/ai-implementation-plan.md WP13, ADR 0016(재학습 루프·모델 레지스트리), ADR 0019,
ADR 0025(4차 감사: 동의 철회 제외, 타임라인 과제, 승인 배포 비교 기준), ADR 0010(상업 사용).
진입점: `dlp train run <과제> <데이터셋 버전>`, `dlp train models`, `dlp train approve <모델 버전>`
(`dlp_cli.train_cmds`). 배포된 모델은 `dlp prelabel run`·`dlp privacy detect`가 `deployed`로 붙인다.

하위 모듈:
- `dlp_train.policy` — `config/policies/training.yaml` 로더 (골든·holdout 분할, 미검수 라벨 금지
  검증).
- `dlp_train.extract` — 데이터셋 버전 스냅샷 → 과제별 학습 예제 (자동 원본과 수정본의 차이 포함).
- `dlp_train.trainers` — 학습기(Trainer)·로더(ModelLoader) 등록부. CI·CPU는 `oracle-stub`만 있다.
- `dlp_train.tracking` — MLflow REST 기록기와 테스트용 메모리 기록기.
- `dlp_train.loop` — 한 번의 재학습 실행, 배포·승인 배포, MLflow 등록 정보.
- `dlp_train.deployed` — 배포 모델을 단계의 기본 어댑터 목록에 끼운다.

주의:
- 학습 예제는 데이터셋 버전의 학습·검증 분할에서만 뽑는다. 골든·holdout 세션이 섞이면 실패한다.
- 후보 모델의 골든셋 예측은 메모리에서만 쓰고 DB에 쓰지 않는다.
- 블러(privacy) 모델은 게이트를 통과해도 사람 승인(`dlp train approve`) 후에만 배포한다.
- DB 쓰기: training_runs, model_versions (호출자가 트랜잭션을 연다). MLflow 레지스트리 등록은 DB
  커밋 뒤에 CLI가 한다 (DB 레지스트리가 기준, MLflow는 사본).
- TODO(real-model): 실제 학습기가 없다 (`trainers` 참조).
"""
