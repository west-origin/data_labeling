# ADR 0029: 검수 완료 판정과 생애주기 전이를 명령으로 남긴다

- 상태: 채택
- 날짜: 2026-10-07
- 관련: ADR 0007(데이터셋·분할), 0018(내보내기), 0020(운영 지표), 0027(verified_at), 0028(생애주기 기록)

## 배경

4차 검수 병합 중, 세션을 `prelabeled → human_verified`로 옮기는 명령이 없다는 것을 확인했다.
종단 테스트만 `set_lifecycle`을 직접 불렀다. 그러면 운영에서는

- `dlp dataset build`가 human_verified 세션만 split_assigned로 옮기므로 어떤 세션도 분할 배정·내보내기
  단계로 나아가지 못하고 (빌드 후보 자체는 ADR 0007대로 프라이버시 승인된 세션이다. 검증 전 세션도
  분할에 들어가며 생애주기만 그대로다 — ADR 0031에서 바로잡은 서술),
- 운영 지표의 "검증 에피소드"와 원본 보관 기산점이 정해지지 않는다.

`split_assigned → exported` 전이도 어느 단계도 남기지 않았다.

## 결정

1. `dlp review verify <세션>` (`dlp_review.verify.verify_session`)이 검수 완료를 판정하고
   human_verified로 옮긴다. 조건(모두):
   - privacy_state = approved (승인이 풀린 세션은 완료가 아니다)
   - 수거하지 않은 검수 작업(open)과 열린 검수 배정이 없다
   - 블러를 뺀 현재 운영 라벨이 있고, 그중 모델 라벨은 모두 human_approved·human_corrected·
     sample_verified다 (사람이 만든 라벨은 그 자체로 검수됨)
   조건을 못 채우면 남은 일을 이유로 출력하고 종료 코드 1. 이미 검증 이후 단계면 그대로 둔다(멱등).
2. 전이 시각(now)과 실행자(`--actor` 또는 DLP_ACTOR)를 생애주기 기록(session_lifecycle_events)에 남긴다.
   운영 지표의 검증 주는 이 첫 전이 시각이다 (`dlp_ops.metrics.verified_time`). 0011 이전에 검증 단계를
   지난 세션(보충 기록뿐)만 ADR 0027의 라벨 이력 추정을 쓴다. 이로써 ADR 0027의 한계(나중의 수명 주기
   변경·늦은 검수로 지난 주의 수가 바뀜)가 사라진다.
3. 내보내기가 manifest까지 올린 뒤 실제로 쓴 세션 중 split_assigned인 것을 exported로 옮긴다
   (실행자 `export:<내보내기 ID>`). holdout 등 split_assigned가 아닌 세션은 그대로 둔다.

## 결과

- 운영 절차: 검수 수거 → `dlp review verify` → `dlp dataset build` → `dlp export ...`.
- 판정은 자동으로 돌지 않는다 (수거 웹훅에서 부르지 않는다). 리드가 남은 일을 확인하고 실행한다.
