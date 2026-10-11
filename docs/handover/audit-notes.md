# 이전 검토 메모: 재현 전 후보

[인수인계 첫 화면](README.md) · [확인된 현재 과제](known-issues.md)

기존 `docs/developer-guide.md`의 정확도·작은 의심·정리 후보를 보존한 자료다.
**각 항목은 이번 인수인계 정리에서 전부 재현하지 않았다.** 과거의 함수 이름·개수·복잡도·영향도
설명은 현재 코드와 다시 대조한다. 확정 결함이나 현재 완료 상태의 근거로 사용하지 않는다.
수정 전 최소 재현·실영상 정답·의도 확인으로 검증하고, 확인된 문제만 `known-issues.md`에 옮긴다.
각 항목의 `[A-nn]`은 [알려진 문제 레지스트리](issues.yaml)의 ID다. 2026-10-11 전수 검수에서 다시 확인한 항목은
레지스트리의 status가 open(확정)이거나 superseded(다른 ID로 옮김)다.

## 정확도와 작은 의심

- [A-01] 카메라 내부 파라미터: 화각 근사에서 주점을 `width/2`로 두고 렌즈 왜곡을 무시한다
  (`dlp_models.depth`, `dlp_prelabel.lift3d`). 광각 액션캠은 화면 가장자리 3D 오차가 크다 — 보정값을
  세션 메타데이터로 받는 것이 다음 단계.
- [A-02] 평가: 블러 정밀도와 객체 오탐은 정답 키프레임 시각에서만 센다 (ADR 0013), 커버리지 평가는 정답에 없는
  (표면, 도구) 예측을 무시한다, 파지·동사 macro F1은 놓친 정답을 "missing" 클래스로 넣어 벌점을 준다
  (의도된 설계로 보이나 확인 필요). `cohen_kappa`는 기대 일치도 1이면 1.0 (scikit-learn은 NaN).
- [A-03] 접촉: merge_gap으로 이은 두 접촉 구간의 점을 이어 커버리지가 조금 커질 수 있다
  (`dlp_relations.contact`, `coverage`). 장갑 히스테리시스 시작·끝 경계 조건이 `>`/`>=`로 비대칭.
- [A-04] 3인칭 착용자 매칭: 키프레임 2개면 속도를 0으로 본다. `wrist_speed`의 `offset_ms`는 시각에
  더하고, 러너는 반환 시각을 `Stream.to_master_ms`로 바꿔 오프셋·드리프트·수동 보정을 반영한다.
- [A-05] OWLv2 질의가 16토큰을 넘으면 잘리며 EOS가 빠진다 (지금 질의는 모두 짧다).
- [A-06] RTMPose: rtmlib YOLOX 출력이 점수 순이라는 보장을 확인하지 못했다 (사람이 `max_people`보다 많을 때).
- [A-07] 동기화: 끝 슬레이트를 찾을 때 seek 직후 첫 프레임에 슬레이트가 이미 보이면 버릴 수 있다
  (`dlp_sync.slate`). `sync adjust`는 ms를 실수로 받는다 (계약도 실수).
- [A-08] 행동: VFR 구간에서 VLM에 요청한 장수보다 적은 프레임을 보낼 수 있다, VLM 4xx(설정 오류)도 장애로
  처리돼 백오프 후 중단된다, gap 응답의 `tool_id`는 위반으로 잡지 않는다.

### 작은 의심

- [A-09] `dlp_review.reconcile`: 128자를 넘는 ID는 앞을 잘라 접두사가 사라질 수 있다 (지금 판정에는 영향 없음).
- [A-10] `dlp_review.clients.lead_seconds`: 취소·서비스 계정 주석의 lead_time도 더한다 (프리라벨은 예측으로
  올리므로 현재 영향 없음).
- [A-11] `dlp_review.timeseries`: `rate_hz`가 1000의 약수가 아니면 `time_ms`가 정수가 아니다 (기본 50 Hz는 정상).
- [A-12] `dlp_schema.episode.EpisodeGraph`: 공간 라벨 구간(스트림 PTS)을 마스터 구간과 그대로 비교한다.
- [A-13] `dlp_schema.db.repository`: `_is_additive`가 목록 원소 추가를 덧붙이기로 보지 않는다 (의도 확인),
  `set_privacy_state`는 전이 순서를 검사하지 않고 기록도 남기지 않는다.
- [A-14] `dlp_privacy`: 여유를 더한 박스가 화면 밖으로 잘린 관측 프레임도 score가 남아 저신뢰 구간에 들어갈 수
  있다 (검수 구간이 늘어나는 쪽), 렌더 결과의 `applied`가 실제로 칠하지 않은 박스도 센다.
- [A-15] `dlp_train`: 누적 기준(`min_new_examples`)이 rejected 모델 뒤에도 걸린다 (의도 확인), MLflow
  실험 조회가 200이 아니면 오류여도 생성을 시도한다.

## 정리 후보

- [A-16] 기존 안내의 추가 후보: `dlp review create` 직접 경로에서 여러 바디캠의 Label Studio 작업과
  시계열 CSV 키가 중복될 가능성, `config/defaults.yaml golden_set`의 `split_unit`·
  `min_instances_per_class`가 사용되지 않는다는 지적. 현재 호출 경로·설정 소비를 다시 확인한다.
- [A-17] 중복 코드: 마스터→스트림 시각 변환(`dlp_export.frames.stream_ms`, `dlp_active.curation._stream_ms` →
  `Stream` 메서드로), 영상 종류 상수(`VIDEO_KINDS`/`VIDEO`) 10곳, 박스 IoU 4곳(`dlp_models.owlv2._iou`,
  `dlp_prelabel.common.iou`, `dlp_review.ops.priority._iou`, `dlp_privacy.geometry.Box.iou`),
  `version_tag` 2곳(`dlp_actions.assemble`, `dlp_schema.episode`), P/R/F1 계산 3곳(`dlp_eval.harness`),
  오디오 상관 호출(`dlp_sync.pipeline`), 임시 PostgreSQL `pg` 픽스처 여러 벌(공용 conftest로).
- [A-18] 하드코딩: `dataset_cmds` 도메인·온톨로지 기본값, `train_cmds` 과제 선택지, `dlp_media.proxy` 인코더
  rate 30, `dlp_review.tasks` 골격 이름(`kp_hand21` 등), `dlp_fixtures.sessions`의 `rng.choice(3)`.
- [A-19] 경계·타입: `dlp_train.deployed`가 `loop._download`(비공개)를 import, `boxes_to_labels`가
  mediapipe 모듈에 있고 `Any` 타입, `dlp_eval.gate`의 `type: ignore`, `dlp_review.collect`가
  `cvat.http`를 직접 호출(4xx가 KeyError로).
- [A-20] 성능(작은 데이터에선 무해): `dlp_datasets.splitter.assign_splits` O(작업자²×세션),
  `dlp_eval.metrics.detection` O(C·I·N), `dlp_sync.taps.match_taps` O(R²·T²), `dlp_media.pts.nearest`,
  `dlp_privacy.tracker` 보간, `dlp_fixtures` `_wrist_track`.
- [A-21] 기타: `dlp_train.tracking`의 `httpx.Client` 미종료, `dlp_cli.review_cmds.cmd_register` 웹훅 중복 등록,
  `dlp_schema.db.repository` 예외 종류 불일치(`NoResultFound`/`KeyError`), `insert_*`의 datetime 되돌리기
  반복, `dlp_actions.runner.KINDS` 미사용, 이름 혼동(`dlp_review.collect`의 `retracted`,
  `VERIFIED_STATES`/`VERIFIED`, `dlp_media.ingest`의 변수 `ms`).
