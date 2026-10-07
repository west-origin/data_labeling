# ADR 0025: 평가 하네스·재학습 루프 감사 정정 (4차)

- 상태: 채택
- 날짜: 2026-10-07
- 관련: WP11(평가), WP13(재학습), ADR 0013, ADR 0016, ADR 0019. 계약(dlp_schema)·DB 변경 없음.

## 결정

1. **동의 철회는 학습에도 적용한다.** `dlp train run`은 데이터셋 버전의 학습·검증 분할 중
   `withdrawals`에 있거나 생애주기가 withdrawn인 세션을 학습 예제에서 뺀다 (`loop.withdrawn_sessions` →
   `extract.load_training_data(excluded=…)`). 버전을 만든 뒤 철회해도 다음 학습부터 빠진다. 골든셋 평가는
   이미 `golden_sessions`가 뺀다.
2. **예측 보간 간격은 과제별이다** (`evaluation.yaml max_interp_ms: {default, tasks}`). 프리라벨 트래커는
   `max_gap_ms`보다 짧은 끊김을 한 트랙으로 잇되 그 사이 키프레임을 만들지 않는다 (OWLv2 도구는 500 ms마다
   추론). 평가가 그보다 짧게만 보간하면 같은 박스도 놓친 것으로 세어 기존(기본 어댑터) 지표가 0에 가깝고
   약한 후보도 게이트를 통과한다. objects는 `prelabel.yaml open_vocab_objects.max_gap_ms`(1200) 이상,
   body는 `body.max_gap_ms`(300) 이상으로 두고, 테스트가 두 정책 파일의 관계를 검사한다. 객체 클래스별로
   나누지 않은 것은 과제 하나에 기본 어댑터 두 개(COCO·OWLv2)의 예측이 섞여 평가되기 때문이다 (COCO 트랙은
   간격이 300 ms 이하라 1200으로 보간해도 결과가 같다).
3. **타임라인 과제는 세션마다 한 번 예측한다.** 접촉·행동(관계·상태·커버리지 포함, `runner.TIMELINE_TASKS`)
   라벨은 마스터 타임라인 구간이고 stream_id가 없다 (ADR 0019). 재학습 루프의 골든 예측
   (`predict_golden`)은 이 과제를 기준 스트림(바디캠)에만 돌린다 (영상 스트림마다 돌리면 같은 구간이 겹쳐
   오탐이 된다). oracle-stub 정답 조회는 그 스트림의 정답과 세션의 타임라인 정답을 함께 준다.
4. **키포인트 매칭.** 정답 키프레임 시각마다 일대일 헝가리안 할당을 한다. 전신(coco17)은 보이는
   (visibility>0) 관절만으로 만든 박스의 IoU 합이 최대가 되게 (IoU 0인 짝은 맞추지 않음), 손(hand21)은 같은
   스트림·같은 쪽 손끼리 관절 평균 거리 합이 최소가 되게 맞춘다. 예측 트랙은 한 시각에 정답 하나에만 쓴다.
   바디캠에 착용자와 돌봄 대상의 같은 쪽 손이 함께 보이는 경우를 다룬다.
5. **승인 배포의 비교 기준은 리포트에 남긴다.** 재학습 루프는 게이트에서 비교한 배포 모델 버전(없으면
   null)을 평가 리포트 `golden.json`의 `deployed_baseline`과 MLflow 파라미터에 남긴다. `dlp train approve`
   (`deploy(…, artifacts=…)`)는 리포트를 읽어 그 값이 지금 배포 모델과 같을 때만 배포한다. 판정 시각
   (`decided_at`)은 학습 실행 시작 시각이라 그 사이에 배포된 모델과 비교했어도 늦게 배포된 것처럼 보여
   막히던 문제를 없앤다. 리포트에 값이 없으면(이전 리포트) 다시 학습·평가하라고 막는다. DB 필드를 늘리지
   않는다.
6. **`dlp eval golden`의 한 과제에 여러 버전.** `--model`·`--baseline`을 한 과제에 여러 번 주면 예측을
   세션별로 합친다 (`runner.load_golden_merged`, 재학습 루프의 `--baseline-version`과 같은 코드). 기존 모델
   버전 중 골든셋에 예측이 없는 것이 있으면 실패한다 (잘못 적은 버전으로 게이트가 통과하지 않게).
7. **세션마다 다른 모델 버전.** 접촉(`contact-heuristic-1+p<정책>+i<입력>`), 행동(`actions-<정책>+<VLM>+i<입력>`),
   3D 궤적(`depth3d-…+i<입력>`)은 버전에 세션 입력 해시가 들어가 세션마다 다르다. 평가·비교에는
   `contact=contact-heuristic-1*`처럼 버전 앞부분+`*`를 쓴다 (`dlp eval golden --help`, ADR 0013).
8. **구간 지표 참조 대조.** 구간 F1@IoU는 MS-TCN `f_score`, temporal mAP는 ActivityNet
   `compute_average_precision_detection`의 계산을 테스트에 옮겨 무작위 입력에서 결과가 같은지 본다.

## 남은 일

- 프리라벨은 배포된 재학습 모델을 영상 스트림마다 부른다. 배포 접촉 모델은 타임라인 구간을 내므로
  3인칭 영상이 있는 세션에서 같은 구간이 스트림 수만큼 나올 수 있다. 프리라벨이 타임라인 과제 모델을 기준
  스트림에만 돌리기 전까지 contact는 `deploy: approve`로 둔다 (`training.yaml`).
- 행동 단계는 아직 배포된 행동 모델을 읽지 않는다 (`training.yaml`의 TODO(real-model)).
