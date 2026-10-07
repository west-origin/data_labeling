# ADR 0013: 지표와 평가 하네스, 배포 게이트

- 상태: 채택
- 날짜: 2026-10-07
- 관련: WP11

## 결정

1. **지표는 직접 구현하고, 참조 구현과 결과가 같은지 테스트로 묶는다.** 런타임 의존성을 늘리지 않고
   (numpy, scipy만), 개발 의존성의 참조 구현과 대조한다.
   | 지표 | 참조 구현 | 일치 |
   | --- | --- | --- |
   | 검출 mAP(0.50:0.95), AP50, AP75 | pycocotools | 소수점 9자리 |
   | HOTA, DetA, AssA, LocA, IDF1, MOTA, ID 전환 | TrackEval (커밋 고정) | 소수점 12자리 |
   | macro F1, Cohen 카파 | scikit-learn | 소수점 12자리 |
   PCK, 접촉 시점 F1·오차, 구간 F1@IoU(MS-TCN 방식), temporal mAP(ActivityNet 방식), 경계 일치 F1,
   상태 전이 정확도, ECE, 커버리지 오차는 손으로 계산한 예제와 대조한다.
   TrackEval은 numpy 2에서 없어진 `np.float` 별칭을 써서 테스트 안에서만 별칭을 붙여 부른다.
2. **하네스는 라벨 레코드를 과제별 지표로 바꾼다** (`dlp_eval.harness`). 과제: objects, hands, body,
   contact, actions, relations, states, coverage. 장갑 세션은 접촉 허용 오차 70 ms, 맨손은 150 ms다.
3. **골든셋 정답은 사람이 만들었거나 사람이 승인·수정한 현재 라벨이다.** 표본 검증만 된 모델 라벨은
   정답이 아니다. 예측은 지정한 model_version의 원래 레코드다 (나중에 고쳐졌어도).
4. **하위 집단(장갑·맨손, 장소)별로 같은 지표를 따로 낸다.** 클래스별 정답 수가
   `min_samples_per_class`보다 적으면 "표본 부족"으로 표시한다 (게이트는 막지 않고 경고).
5. **배포 게이트** (`config/policies/evaluation.yaml gate`): 과제마다 주 지표가 기존보다 `min_gain`
   이상 좋고, 주 지표와 지키는 지표가 전체와 모든 하위 집단에서 `max_drop`보다 나빠지지 않아야 한다.
   기존 모델이 없으면 주 지표가 `first_deploy` 기준을 넘어야 한다. 오차·ECE는 낮을수록 좋다.
6. `dlp eval golden <골든셋> --model <과제>=<버전> [--baseline …] --out reports/x.json`은 JSON과
   Markdown 리포트를 쓰고, 게이트가 실패하면 종료 코드 1을 낸다 (WP13 재학습 루프가 쓴다).

## 한계

- 검출·추적은 정답 키프레임 시각에서만 비교한다. 정답이 성기게 찍혀 있으면 그 사이 오탐을 세지 못한다.
- 전신 PCK는 시각마다 키포인트 박스 IoU로 사람을 맞춘다 (ID 일관성은 보지 않는다).
- 3D 궤적 오차(3인칭 다시점 대비)는 다시점 결과가 생긴 뒤에 추가한다.
