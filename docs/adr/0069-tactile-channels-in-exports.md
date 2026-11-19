# ADR 0069: 촉각장갑 채널을 내보내기 에피소드 프레임에 싣기 (`dlp export lerobot|gr00t`)

- 상태: 채택 (구현됨. LeRobot 공식 쓰기·읽기 API와 GR00T 자체 로더로 읽는 확인은 아직, 아래 "지금 되는 것과 아닌 것")
- 날짜: 2026-11-17
- 관련: ADR 0004·0028(동기화), 0018·0021·0027(내보내기·이력·가명), 0019(시각 규약), 0020·0021(원본 접근 감사),
  0047(로봇 모드), 0065(GR00T 형식)
- 계약 변경: 없음 (`dlp_schema` 그대로, `make schemas` 대상 없음). 정책 `config/policies/export.yaml tactile` 절과
  `dlp_export.policy.TactilePolicy`를 더했고, `config/policies/ops.yaml audit.service_purposes`에 용도
  `export.lerobot`·`export.gr00t`를 더했다. 구매자에게 주는 새 파일 `meta/dlp_tactile.json`(형식 `dlp-tactile`, 판 1.0)이 생겼다.

## 배경

소유자 요구는 "1인칭 작업 영상과 촉각 센서를 동기화한다"이다. 납품하는 로봇 학습용 에피소드에서 프레임마다 그 순간의
촉각 값이 함께 있어야 한다.

있던 것:

- 수집(`dlp_media.glove`)이 장갑 스트림(glove_left·glove_right)을 정규화 Parquet(`t_ms` + 채널 열, 시각은 장갑 기기
  시계)으로 원본 버킷 `sessions/<세션>/derived/<스트림>.parquet`에 둔다.
- `dlp sync run`이 스트림마다 마스터 타임라인(바디캠 시계)과의 오프셋·드리프트를 정한다 (ADR 0004·0028,
  `Stream.to_master_ms`).
- 장갑 값을 쓰는 곳은 접촉 프리라벨, 검수 화면 시계열, AI 1차 검수 입력이다. 모두 원본 버킷의 Parquet을 감사 저장소로
  읽고 동기화 결과로 마스터 시각으로 바꾼다.

없던 것: 내보내기(LeRobot·GR00T)에 촉각 값이 없었다. 장갑 압력으로 만든 접촉 구간 라벨(hand_state)만 간접적으로 나갔다.

## 결정

1. **싣는 곳.** `dlp export lerobot`과 `dlp export gr00t`의 에피소드에, 그 세션 장갑 스트림의 채널 값을 프레임마다
   싣는다 (`dlp_export.tactile`). 손마다 따로 싣는다 (왼손 = glove_left, 오른손 = glove_right). coco·intervals에는
   싣지 않는다.
2. **시간 맞춤.** 장갑 샘플 시각(기기 시계)을 그 스트림의 동기화 결과로 마스터 시각으로 옮긴다. 식을 다시 적지 않고
   계약의 `Stream.to_master_ms`(오프셋 + 사람 조정 + 시각 × 배율)를 부른다. 에피소드 프레임 시각은 `build_episode`가
   정한 마스터 시각(`Episode.times_ms`)을 그대로 받는다 (프레임 번호로 새로 계산하지 않는다). 프레임 시각의 값은 앞뒤
   샘플 사이 선형 보간이다. 세션의 동기화 값은 데이터셋 버전 스냅샷에 고정된 것이라, 다시 동기화했으면 데이터셋 버전을
   다시 만들어야 반영된다 (영상 스트림과 같다).
3. **없는 값은 지어내지 않는다.** 다음 프레임은 "없음"이다.
   - 장갑 기록 범위 밖 (첫 샘플 앞, 마지막 샘플 뒤. 외삽하지 않는다).
   - 앞뒤 샘플 간격(마스터 ms)이 `tactile.interp_max_gap_ms`(기본 50)보다 큰 곳.
   - 그 세션·손에 실은 장갑이 없는 에피소드.

   없음은 손 3D 점과 같은 방식으로 적는다: 값 0과 표시 0. 압력 0은 실제 값(누르지 않음)일 수도 있으므로 표시로만
   가른다. 보간 규칙은 `lerobot.interp_series`와 같다 (샘플 시각과 정확히 같은 프레임은 그 값).
4. **동기화되지 않은 장갑은 싣지 않는다.** `sync_method`가 unsynced인 스트림은 시간축을 알 수 없다. 동기화를 돌리지
   않았거나, 모든 방법이 `sync.yaml min_confidence`에 못 미쳐 거부된 스트림이 여기에 든다. 판정 기준은 접촉
   프리라벨·AI 1차 검수 입력과 같다 (`sync_method is not unsynced`). 싣지 않은 사실은 세션 기록(`not_synced`)과
   경고로 남긴다. 같은 손의 장갑 스트림이 둘 이상이면 어느 것이 그 손의 값인지 정하지 않고 싣지 않는다
   (`multiple_streams`).
5. **원본 버킷은 감사 저장소로만 읽는다.** `run_export`가 `raw`(감사 저장소 `AuditedStore`)를 받고, CLI가
   `raw_store(..., "export.lerobot")`·`raw_store(..., "export.gr00t")`로 연다. 용도는 형식마다 다르고 ops.yaml에
   등록했다. 읽기마다 `raw_access_log`에 남는다. 동기화되지 않은 장갑과 장갑 없는 세션은 읽지 않는다. 결과물에는 채널
   값, 채널 이름, 장갑 기종, 동기화 방법·신뢰도, 채움 수만 넣는다. 원본 위치(버킷·키·URI)와 내부 세션 ID는 넣지 않는다
   (`assert_no_raw`가 형식과 관계없이 결과 바이트를 본다). 실어야 할 장갑을 읽을 수 없으면(감사 저장소를 받지 못함,
   객체 없음, 형식 오류) 조용히 빼지 않고 실패한다.
6. **채널.** `tactile.channel_prefixes`가 비면 정규화 Parquet의 채널 전부를, 있으면 그중 하나로 시작하는 채널만 싣는다.
   기본은 전부다. 장갑 기종과 채널 이름이 정해지지 않았고, 접두사 기본값이 맞지 않는 기종의 촉각 채널이 조용히 빠지는
   것보다 이름을 메타에 적고 전부 싣는 쪽이 안전하다. 열 순서는 파일의 열 순서가 아니라 채널 이름의 자연 정렬이다
   (pressure_2가 pressure_10 앞). 같은 손의 채널 구성이 세션마다 다르면(장갑 기종이 다름) 한 데이터셋의 열 모양이
   달라진다. 0으로 채우거나 버려서 맞추지 않고, 이유(두 세션, 채널 수, 한쪽에만 있는 채널, 장갑 기종)와 함께 거부한다.
   장갑이 있는 세션과 없는 세션이 섞이는 것은 거부하지 않는다. 없는 세션은 3번의 "없음"이다.
7. **값의 뜻.** 값은 기기가 낸 그대로다. 단위는 모른다고 적는다 (`units: unknown`). 세션 계약의 기종별 보정값
   (`Calibration.glove_calibration`)은 적용하지도 싣지도 않는다. 장갑 기종(`glove_model`)은 세션마다 적고, 기종이
   섞이면 경고한다.
8. **LeRobot.** 손마다 특징 `observation.tactile.<손>` (float32, 길이 = 채널 수 + 1). 이름은 `<손>.tactile.<채널>`과
   끝의 `<손>.tactile.present`다. 값과 표시를 한 벡터에 두는 것은 `observation.state`의 손 3D 점
   (`x3d, y3d, z3d, present`)과 같다. `write_package`가 사람 모드와 로봇 모드의 특징 위에 덧붙인다.
9. **GR00T.** `observation.state` 끝에 손마다 채널 값을 잇고 `meta/modality.json` state에 `<손>_tactile` 구간을
   더한다. 표시는 새 열 `observation.tactile_valid`(float32, 손마다 1/0)다. `observation.hand_valid`와 같은 방식이고
   기존 열의 모양은 바꾸지 않는다. action에는 넣지 않는다. 촉각은 관측이지 행동이 아니다. 그래서 촉각이 실리면 state가
   action보다 길고 modality.json의 state·action 구간이 다르다. NVIDIA 문서(`getting_started/data_preparation.md`,
   2026-11-17 main)의 modality.json 구조는 state 키가 state 배열의, action 키가 action 배열의 구간을 따로 가리킨다.
   확인기(`gr00t_check`)는 `meta/dlp_tactile.json`이 있으면 그 구간이 채널 목록과 맞는지(길이, info.json 이름,
   action에 없음, 표시 열)와 표시가 0인 행의 값이 0인지를 본다.
10. **켜고 끄기와 불변.** `tactile.enabled`(기본 true)로 끈다. 끄면 원본 버킷을 열지 않는다. 내보내는 세션에 장갑
    스트림이 하나도 없으면 결과물은 이 기능이 없던 때와 바이트 단위로 같다 (특징·열·파일·manifest 키를 더하지 않는다).
    장갑 스트림은 있는데 실은 손이 없으면(모두 동기화 전) 열은 그대로 두고 기록 파일만 더한다.
11. **기록.** 데이터셋의 `meta/dlp_tactile.json`과 manifest `details.tactile`이 같은 내용이다. 형식 이름과 판
    (`dlp-tactile` 1.0), 시간 맞춤 방식, 없음 표현, 단위 모름, 보간 한도, 채널 선택, 손마다 채널 이름과 값이 실린 위치,
    장갑 기종 목록을 적는다. 세션마다(가명) 장갑 기종과 손별 상태, 동기화 방법·신뢰도, 샘플레이트, 프레임 수, 장갑 기록
    구간 안 프레임 수, 값이 있는 프레임 수를 적는다. 판 규칙은 구간 JSON과 같다 (필드를 더하면 부 버전, 받는 쪽 리더가
    깨지는 변경은 주 버전). 경고(`details.warnings`)는 넷이다: 동기화 안 됨, 장갑 기록이 에피소드와 겹치지 않음, 기록
    구간 안 채움 비율이 `tactile.min_coverage`(0.9)보다 낮음, 장갑 기종 섞임.

## 대응표

| 위치 | 값 | 정하는 곳 |
| --- | --- | --- |
| LeRobot `observation.tactile.<손>` float32[채널 수 + 1] | 프레임 시각의 채널 값, 끝이 present (없으면 값 0·present 0) | `TactileData.lerobot_features`·`lerobot_arrays` |
| GR00T `observation.state`의 `<손>_tactile` 구간 (손·팔 구간 뒤, 왼손이 앞) | 프레임 시각의 채널 값 (없으면 0) | `TactileData.gr00t_modality`·`gr00t_arrays` |
| GR00T `observation.tactile_valid` float32[실은 손 수] | 손마다 1 있음·0 없음 | 같은 곳 |
| GR00T `action` | 촉각 없음 (손·팔 state의 다음 프레임 그대로) | `gr00t.episode_table` |
| `meta/dlp_tactile.json`, manifest `details.tactile` | 채널·위치·장갑 기종·단위 모름·세션별 상태 | `TactileData.meta` |
| manifest `details.warnings` | 동기화 안 됨, 겹치지 않음, 채움 비율 낮음, 기종 섞임 | `TactileCollector` |
| 세션·손 상태 | `exported`, `not_synced`, `no_glove`, `multiple_streams` | `TactileCollector` |
| `export.yaml tactile` | `enabled`, `channel_prefixes`, `interp_max_gap_ms`, `min_coverage` | `TactilePolicy` |
| `ops.yaml audit.service_purposes` | `export.lerobot`, `export.gr00t` | `dlp_cli.export_cmds.tactile_store` |

## 지금 되는 것과 아닌 것

합성 정답(`dlp_fixtures.tactile`: 채널 값이 마스터 시각의 식이고 장갑 시계를 안다)으로 확인한 것:

- 내보낸 프레임 값이 정답 신호를 그 프레임의 마스터 시각에서 읽은 값과 같다. 두 장갑의 시계가 다르다 (오프셋
  +412.5 ms·배율 1.02, 오프셋 −318 ms 중 37.5 ms는 사람 조정·배율 0.985). 허용 오차는 앞뒤 샘플 간격의 선형 보간 오차
  상한이다. 오프셋, 드리프트, 사람 조정 중 하나라도 빼먹게 구현을 바꾸면 시험이 실패하는 것을 확인했다.
- 장갑 기록 전·후와 112 ms 빈 구간의 프레임은 없음이고, 41 ms 빈 구간은 보간해 싣는다.
- 동기화되지 않은 장갑은 싣지 않고 기록에 남으며 원본을 읽지 않는다. 채널 구성이 다른 세션이 섞이면 거부하고, 접두사로
  공통 채널만 고르면 내보낸다.
- 장갑 없는 내보내기는 변경 전 코드와 결과가 같다 (GR00T 결과 파일 9개의 sha256, LeRobot 묶음의 특징·배열, details).
- GR00T 결과가 확인기를 통과하고, 촉각 구간을 깨뜨린 데이터셋 넷을 확인기가 거부한다.
- PostgreSQL 종단(`run_export`): gr00t와 lerobot(쓰기 스크립트는 대역)에서 manifest·`raw_access_log`가 맞고 결과
  바이트에 원본 버킷 이름·장갑 키·내부 세션 ID가 없다.

아닌 것과 모르는 것:

- (2026-11-19 바로잡음) LeRobot 공식 쓰기·읽기 API로 촉각 특징을 쓰고 읽는 시험
  (`test_tactile.py::test_official_lerobot_loader_reads_tactile_features`, isolated_env)은 구현 환경에서는 PyTorch CPU판을
  받지 못해 돌리지 못했지만, CI의 서비스 작업(`make test-isolated`)에서 2026-11-18 푸시부터 돌아 통과한다.
- **GR00T 자체 로더·통계 생성기로 읽어 보지 않았다** (ADR 0065와 같다). state와 action의 키가 다른 데이터셋을 받는지는
  문서로만 판단했다. (F-COE-97)
- **로봇 모드(`--robot`)는 묶음까지만.** 같은 촉각 특징(사람 장갑 값, 두 손)을 로봇 특징 위에 덧붙인다
  (`dlp_export.robot`은 고치지 않았다). 로봇 팔 쪽 손만 고르거나 그리퍼 값으로 바꾸지 않는다. 격리 환경 종단은 돌리지
  못했다. (F-COE-97)
- **실제 장갑 데이터로 돌려 보지 않았다.** 채널 이름, 샘플레이트, 끊김 양상을 모르고, 보간 한도 50 ms가 맞는지도
  모른다. 값은 보정하지 않은 기기 값이고 단위를 모른다. 압력이 아닌 채널(IMU·온도 등)도 기본으로 실린다 (이름으로
  가른다). 기종이 섞이면 경고만 한다. (F-COE-98)
- **프레임 값의 시간 정확도는 동기화 정확도를 넘지 못한다.** 합성에서 장갑 동기화 오차는 약 5~7 ms다 (ADR 0004·0028).
  실제 장갑은 재지 않았다. 프레임별 시간 불확실성은 싣지 않고 세션 단위 `sync_method`·`sync_confidence`만 적는다.
  (F-COE-99)
- **장갑 Parquet을 데이터셋 버전에 고정하지 않는다.** 수집은 파생 파일을 없을 때만 만들어 보통 바뀌지 않지만,
  내보내기는 해시를 확인하지 않는다. 원본 보관 만료로 지워진 세션은 내보내기가 실패한다. (F-COE-99)
- **값 없음 = 0.** GR00T 통계(평균·분산)는 0을 실제 값으로 본다. `observation.tactile_valid`로 걸러야 한다
  (ADR 0065의 손 점과 같다).

## 결과

- 운영: `dlp export lerobot|gr00t <버전> --target <받는 곳>`이 동기화된 장갑이 있는 세션의 촉각 값을 함께 싣는다.
  끝에 실은 손의 채널 수와 세션·손 상태별 수를 한 줄로 알리고 경고를 낸다. 끄려면 `export.yaml tactile.enabled: false`.
- 원본 접근: lerobot·gr00t 내보내기가 원본 버킷의 장갑 Parquet을 읽게 됐다. 월간 감사 리포트에는 용도
  `export.lerobot`·`export.gr00t`의 서비스 계정 읽기로 나온다. 영상은 전과 같이 라벨링 버킷의 블러본만 읽는다.
- 기존 형식의 결과: 장갑 스트림이 없는 세션만 내보내면 바뀌지 않는다. 동기화된 장갑이 있는 세션을 lerobot·gr00t로
  내보내면 특징·열이 늘어난다 (GR00T는 `observation.state` 길이가 늘고 modality.json state에 키가 늘어난다).
- 시험: `packages/export/tests/test_tactile.py`(DB 없이), `test_runner.py`의 촉각 종단 둘(services),
  `packages/cli/tests/test_export_cmds.py`(원본 저장소를 여는 조건과 용도).
- 남은 일은 F-COE-97~99에 등록했다.
