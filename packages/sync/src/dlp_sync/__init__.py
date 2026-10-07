"""멀티스트림 동기화 (WP4, ADR 0004·0028).

한 세션의 여러 스트림(3인칭 영상, 좌우 장갑, 외부 IMU, 외부 오디오)의 시계를 기준 스트림인
바디캠 시계(= 마스터 타임라인)에 맞춘다. `dlp sync run <세션>`(`dlp_cli.sync_cmds`)이
`dlp_sync.runner.run_sync`를 통해 이 패키지를 쓴다.

시계 모델 (계약 `dlp_schema.session.Stream`과 같다)
    master_ms = offset_ms + manual_adjustment_ms + stream_ms * clock_scale

- `offset_ms`: 스트림 시각 0이 마스터 타임라인의 몇 ms인지. 양수면 스트림이 바디캠보다 늦게 시작했다
  (스트림 0 ms = 마스터 `offset_ms`).
- `clock_scale`: 스트림 시계 1 ms가 마스터 시계 몇 ms인지. 1이면 드리프트 없음. 드리프트(ppm) =
  (`clock_scale` - 1) * 1e6. 양수 ppm이면 스트림 시계가 바디캠보다 느리게 간다(같은 실제 시간에
  스트림 시각이 덜 증가한다).
- `manual_adjustment_ms`: 사람이 검수 화면에서 정한 미세 조정값. 자동 동기화와 따로 저장하고 다시
  동기화해도 유지한다 (`apply_manual_adjustment`).
- 기준 스트림(바디캠, `SyncMethod.REFERENCE`)은 항상 0/1/0이다. 바디캠과 같은 시계를 쓰는 스트림
  (내장 IMU 등 `SyncMethod.SHARED_CLOCK`)은 동기화하지 않는다.

방법과 대체 순서는 `config/policies/sync.yaml`의 `methods`(스트림 종류별 목록)에 있다. 목록을
앞에서부터 시도해 `fit`이 있고 신뢰도가 `min_confidence` 이상인 첫 결과를 쓴다
(`pipeline.synchronize`).

- `qr_slate` (`slate.py`): 두 영상에 같이 찍힌 QR 슬레이트의 첫 등장 프레임을 앵커로.
- `tap_event` (`taps.py`): 기준 오디오와 대상 신호의 "두 번 두드림" 사건을 짝지어 앵커로.
- `audio_xcorr` (`xcorr.py`): 두 마이크 오디오 상호상관 (거친 탐색 → 창별 정밀 탐색).
- `motion_xcorr` (`xcorr.py`): 바디캠 내장 IMU 가속도 크기와 장갑 압력 합의 상호상관.

공개 API
- `synchronize`: 세션 + 스트림 입력 → 동기화된 세션과 보고서 (DB·저장소 접근 없음, 순수 계산).
- `apply_manual_adjustment`: 사람이 정한 조정값을 세션에 기록.
- `SyncReport`: 스트림별 시도한 방법·신뢰도·앵커 (저장소
  `sessions/<세션>/derived/sync_report.json`).
- `SyncPolicy`, `load_policy`: `config/policies/sync.yaml` 로더.

시간 단위: 모든 시각은 ms(float)다. 동기화 결과는 계약의 float 필드(`offset_ms`, `clock_scale`)에
들어가고, 라벨 시각(정수 ms)으로의 변환은 사용하는 쪽이 한다 (ADR 0019).
"""

from dlp_sync.pipeline import SyncReport, apply_manual_adjustment, synchronize
from dlp_sync.policy import SyncPolicy, load_policy

__all__ = ["SyncPolicy", "SyncReport", "apply_manual_adjustment", "load_policy", "synchronize"]
