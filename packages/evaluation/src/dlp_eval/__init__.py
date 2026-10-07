"""지표와 평가 하네스 (WP11): 지표 라이브러리, 골든셋 평가, 하위 집단 리포트, 배포 게이트.

관련 문서: docs/ai-implementation-plan.md WP11, ADR 0013(지표·하네스·게이트), ADR 0019(시간 규약),
ADR 0025(4차 감사 정정: 과제별 보간 간격, 타임라인 과제, 키포인트 매칭, 승인 배포 비교 기준),
ADR 0031(5차 정정: 게이트 지표 이름을 정책을 읽을 때 검사).

파이프라인 위치: `dlp eval golden <골든셋> --model <과제>=<버전>`(`dlp_cli.eval_cmds`)과 재학습 루프
(`dlp_train.loop`, `dlp train run`)가 쓴다. 입력은 DB의 골든셋 세션 라벨(정답·예측), 출력은 지표
리포트(JSON·Markdown)와 배포 게이트 판정이다. 이 패키지는 DB에 쓰지 않는다 (읽기만 한다).

하위 모듈:
- `dlp_eval.metrics` — 라벨 계약과 무관한 순수 지표 함수 (배열·튜플 입력). 각 함수 docstring에
  따르는 참조 정의(COCO, TrackEval, MS-TCN, ActivityNet 등)와 차이를 적었다.
- `dlp_eval.harness` — `LabelRecord` → 과제별 지표 (`evaluate`)와 하위 집단 리포트.
- `dlp_eval.gate` — 후보 리포트를 기존 리포트와 비교해 배포 여부를 정한다 (`decide`).
- `dlp_eval.runner` — DB에서 골든셋 정답·예측을 모으고 리포트 파일을 쓴다.
- `dlp_eval.policy` — `config/policies/evaluation.yaml` 로더와 과제 이름(`Task`).

주의: 시간은 모두 정수 ms다. 타임라인 과제(접촉·행동·관계·상태·커버리지)는 마스터 타임라인 시각,
공간 과제(객체·손·전신·블러)의 키프레임은 그 스트림 영상의 PTS 시각이다 (ADR 0019).
"""
