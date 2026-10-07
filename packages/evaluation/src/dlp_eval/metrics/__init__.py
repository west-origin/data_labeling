"""지표 함수. 입력은 단순한 배열·튜플이고 라벨 계약과 무관하다 (하네스가 변환한다).

WP11, ADR 0013. 모듈별 참조 정의:
- `detection` — 박스 IoU, COCO 방식 AP·mAP (pycocotools `COCOeval`과 일치 테스트).
- `tracking` — HOTA, IDF1(Identity), CLEAR(MOTA·MOTP) (TrackEval과 일치 테스트).
- `keypoints` — PCK (기준 길이 = 정답 키포인트 박스 긴 변. PCKh·PCK@bbox와 다르다).
- `temporal` — 구간 IoU, 시점 사건 매칭(접촉 시작·종료), 구간 F1@IoU (MS-TCN `f_score`),
  temporal mAP (ActivityNet `compute_average_precision_detection`), 경계 일치율.
- `states` — 상태 전이 정확도 (이 프로젝트 정의, 시점 매칭은 `temporal.match_events`).
- `classification` — macro F1·Cohen 카파 (scikit-learn과 일치 테스트), ECE (Guo et al. 2017).
- `assign` — scipy 헝가리안 할당의 타입 래퍼.

시간 인자는 모두 정수 ms다. 지표 함수는 부작용이 없고 같은 입력에 같은 결과를 낸다.
참조 일치 테스트: packages/evaluation/tests/test_metrics_reference.py,
test_metrics_temporal_reference.py. 손 계산 대조: test_metrics_hand.py.
"""
