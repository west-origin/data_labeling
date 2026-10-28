# ADR 0063: 의사결정 모델 판정단과 검수 판정 모델 학습

- 상태: 제안 (판정단·불일치 우선순위는 stub으로 구현, 실제 모델 비교는 DGX Spark 실측 전)
- 날짜: 2026-10-28
- 관련: ADR 0010(상업 사용), ADR 0040(AI 1차 검수, 사내 서버만, AI ok는 사람 검수를 줄이지 않음), ADR 0060(공개
  데이터 1차 학습과 2단계 경계), ADR 0062(판정 표시 보정, 실제 학습기).
- 계약 변경: `ReviewReason.AI_DISAGREEMENT`(검수 우선 구간 사유 값 하나 추가, `make schemas`). DB·Alembic은 그대로
  (판정은 이미 버전별로 `ai_review_*`에 쌓인다). 정책 `aireview.yaml backend.panel`, `review.yaml ai_review.disagreement`.
  `models.yaml external`에 후보 두 개(`vllm-sr/d3-turbo`, `mukaddamzaid/amos-1-large`, 둘 다 review).

## 배경

2026년에 이미지를 읽고 선택지와 확률로 답하는 "의사결정 모델"(Jev 계열, d3-turbo, Amos 1 등)이 나왔다. 사용자는 이런
모델이 1차 라벨링·검수를 하고 Qwen이 다시 검수한 뒤 사람이 보는 흐름과, 그 시행착오·사람 판단을 학습에 쓰는 것을
원한다. 지금 AI 1차 검수(ADR 0040)는 판정 모델 하나(Qwen3-VL-8B)가 라벨마다 ok·suspect·wrong·unsure를 내고, 검수 화면은
허용 버전(`review.yaml ai_review.versions`)의 판정 중 가장 나쁜 것을 보인다.

## 결정

1. **1차 라벨링에는 쓰지 않는다.** 우리 라벨의 대부분은 박스·키포인트·마스크·3D 트랙이고 그것은 전용 모델(YOLOX·
   MediaPipe·SAM 등)이 만든다. 의사결정 모델은 "고르기"(행동 분류, 검수 판정)에만 쓴다. 행동 분류에 쓰는 것은 이번
   범위 밖이다 (`actions.yaml vlm`, 같은 모델을 판정과 분류에 함께 쓰면 상관 오류 경고가 이미 있다).
2. **직렬 연결이 아니라 병렬 판정단이다.** 판정단 모델은 같은 블러본 프레임·같은 요청 본문으로 각자 판정하고 서로의
   판정을 보지 않는다. 앞 모델의 판정을 뒤 모델에 보여 주면 그 판정을 따라가 독립된 두 번째 의견이 되지 못한다.
   - 판정단은 `aireview.yaml backend.panel`에 모델(이름·served_name·fingerprint)을 적고 `dlp aireview run --model <이름>`
     으로 고른다. 버전은 지금처럼 `aireview-<digest>+vlm-<model>`이라 모델마다 따로 쌓이고 멱등·재개도 그대로다.
     판정 서버는 사내 OpenAI 호환 서버만이다 (ADR 0040, `check_hosts`).
   - 검수 화면은 지금처럼 허용 버전의 가장 나쁜 판정을 보인다.
3. **불일치는 우선순위 신호다.** 허용 버전 중 둘 이상이 같은 라벨을 판정했고 ok와 ok 아님(suspect·wrong·unsure)이
   갈리면 그 라벨 구간에 `ai_disagreement` 사유를 단다 (`review.yaml ai_review.disagreement`). 사람 검수를 늘리는
   쪽으로만 쓴다: 우선순위 큐에서 앞에 두고, 표본 검수 묶음에서 빼는 목록(`sampling_exclude`)에 넣을 수 있다.
   AI끼리 ok로 같아도 사람 검수·표본은 줄지 않는다 (ADR 0040 I9).
4. **도입은 숫자로 정한다.** 골든셋에서 Qwen 단독·후보 단독·판정단 불일치 표시를 `dlp aireview eval`로 비교하고
   (지적 정밀도·재현율, ok 놓침률 95% 상한, ECE), `dlp aireview bench`로 Spark 한 대의 메모리·처리량을 잰다(27B bf16
   약 54 GB + 8B). 불일치 표시가 대조군 기준 사람의 오류 발견을 늘릴 때만 `ai_review.versions`에 두 버전을 함께 둔다.
5. **후보.** d3-turbo(영상 추적·객체 위치, 우리 트랙 라벨과 가장 맞음)와 Amos 1 Large(선택지 확률, 9B·2B 작은 판).
   둘 다 가중치 Apache-2.0이고 학습 데이터는 공개되지 않아 `commercial: review`다. Amos 1의 기반 모델 라이선스는
   미확인이다. 서버에 올리기 전에 모델 카드 원문·revision을 확인한다 (이 컨테이너는 huggingface.co에 접속하지 못했다).

## 검수 판정 모델 학습 (후속, 구현 전)

AI 판정과 사람의 처리는 이미 쌓인다: `ai_review_*`(판정·이유·제안), `ai_review_outcomes`(사람이 그 판정을 어떻게
처리했나), 라벨 이력(사람의 최종 라벨). 이것을 판정 모델 미세조정에 쓸 때 지킬 것:

1. **정답은 사람의 최종 판단만.** AI 판정·이유는 입력이나 틀린 예로만 쓴다 (AI 판정을 정답으로 삼으면 AI의 습관을
   다시 배운다).
2. **데이터셋 버전의 학습·검증 분할에서만** 뽑는다 (`dlp_train.extract`, 골든셋은 평가만, F-ETA-19).
3. **사내 GPU에서만** 학습한다 (우리 데이터, ADR 0060 2단계. Colab·Kaggle 금지).
4. **평가는 대조군만**(AI 판정을 보지 않고 검수한 몫). AI 표시를 본 검수의 판단은 그 표시에 끌려가 있다.
5. 출발 가중치 라이선스가 미세조정·파생 모델 사용을 허용해야 하고, 학습 모델은 그 분류를 물려받는다 (ADR 0062 A.9).
6. 학습 루프에 "검수 판정" 과제 정의(예제 추출·지표·게이트)가 필요하다. 별도 ADR로 정한다.

## 결과

- 판정단과 불일치 우선순위는 코드·정책이 있고 stub(`OracleJudge`)으로 시험한다. 기본값은 판정단 모델 하나(지금과
  같음), `disagreement` 기본 끔이라 운영 동작은 바뀌지 않는다.
- 남은 일: 후보 모델 카드·revision 확인, Spark에서 bench·골든 eval, 결과에 따라 `ai_review.versions`·`disagreement`
  켜기, 검수 판정 과제 ADR.
