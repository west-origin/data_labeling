# ADR 0022: 3차 검수 정정 — 계약·DB·수집·동기화

- 상태: 채택
- 날짜: 2026-10-07
- 관련: ADR 0002(계약과 라벨 저장), 0003(수집), 0004(동기화), 0019(시각 규약), 0020(추가만 하는 기록).
  Alembic 0010, 온톨로지 v1 보강, `make schemas`

## 1. DB (Alembic 0010)

- `label_records.model_version`을 `VARCHAR(128)`에서 `TEXT`로 넓혔다. 모델 버전에 정책 절 해시가 붙어
  128자를 넘는 경우가 있다(프라이버시 파이프라인 약 165자). PostgreSQL에서는 삽입이 실패했다.
- `dataset_versions.golden_set_version`을 `VARCHAR(64)`에서 `VARCHAR(128)`로 넓혔다. 골든셋 버전 ID
  (`golden_sets.version`, 계약 `Identifier`)와 길이를 맞춘다.
- 추가만 하는 표(`label_records`, `raw_access_log`, `review_work`, `privacy_audits`, `retention_decisions`)에
  `BEFORE TRUNCATE ... FOR EACH STATEMENT` 트리거를 걸었다. 행 단위 수정·삭제 트리거는 TRUNCATE를 거치지
  않는다. `TRUNCATE sessions CASCADE`처럼 딸려 비우는 경우도 막힌다.
- `set_model_status`는 없는 모델 버전이면 `KeyError`를 낸다(조용히 0행 갱신하지 않는다).
- `update_stream_sync`는 기준 스트림(바디캠)의 시계(reference, 오프셋 0, 배율 1, 조정 0)를 바꾸거나
  다른 스트림을 reference로 바꾸는 갱신을 거부한다.

## 2. 계약

- `Session`: 기준 스트림은 오프셋 0, 배율 1, 사람 조정 0이어야 하고, reference 방법은 바디캠만 쓴다
  (ADR 0019의 "두 시각이 같다"를 계약으로 강제).
- `SPATIAL_KINDS`에 `trajectory3d`를 넣었다. 3D 궤적도 그 영상의 PTS 시각이므로 `stream_id`가 필수다.
- `ExportedLabel.payload`는 블러 라벨(`blur_track`)을 거부한다. 어떤 내보내기에도 블러 라벨을 넣지 않는다.
- 온톨로지 v1에 `surface_parts`(corner_0~3, 표면 평면 꼭짓점)와 `hand_joints`(wrist, 손가락 끝 5개)를
  더했다. `Ontology.known_parts()` = 모든 도구 작용부·파지부 ∪ 표면 부분 ∪ 손 관절.
  `check_label`은 `trajectory3d.part`와 `relation.subject_part`·`object_part`가 이 안에 있는지 본다
  (이 페이로드는 객체 클래스를 싣지 않으므로 합집합으로 본다). 기존 합성 픽스처(corner_0~3, cloth_face)가
  사전과 맞게 되었다. 새 필드는 기본값이 빈 사전이라 예전 온톨로지 내용도 읽힌다.
- 온톨로지 이관(`migrate_labels`)
  - 이력 전체가 아니라 `current_labels(operational=False)`만 이관한다. 예전에는 수정된 레코드, 삭제된 레코드,
    삭제 레코드까지 parent=자기 자신으로 새 버전에 복사해 지운 라벨이 되살아났다. 이미 새 버전인 라벨은
    건너뛰므로 다시 돌려도 아무것도 만들지 않는다. 오류 삽입·측정 레코드는 표시를 지닌 채 이관된다.
  - 이관 범주에 `parts`를 더해 `mask_track.part`, `trajectory3d.part`, `relation.subject_part`·`object_part`를
    바꾼다.
- `PlatformConfig`에 `media.proxy`(max_height, crf, keyframe_ms)를 더했다 (`config/defaults.yaml`).

## 3. 수집

- 재수집 충돌은 매니페스트·미디어에서 나온 필드만 비교한다. 동기화 결과, 프라이버시·생애주기 상태,
  온톨로지 버전처럼 수집 뒤 단계가 바꾸는 필드가 달라도 `unchanged`다.
- GPMF에 가속도(ACCL) 샘플이 2개 미만이면 내장 IMU 스트림을 만들지 않는다(샘플레이트 0인 스트림이
  계약 검증에서 실패했다). 자이로가 없으면 자이로 열은 NaN이다. 사이드카 IMU가 2개 미만이면 오류다.
- 장갑 정규화는 시각 열 후보(`t_ms`, `timestamp_ms`)를 모두 채널에서 뺀다.
- 프록시 인코딩 값은 `config/defaults.yaml media.proxy`에서 읽는다 (`dlp ingest`가 넘긴다).

## 4. 동기화

- 자동으로 맞추지 못한(unsynced) 스트림에 `dlp sync adjust`로 값을 넣으면 `manual`이 된다(오프셋 0, 배율 1,
  조정값 = 사람이 정한 오프셋). 예전에는 unsynced로 남아 다음 단계가 그 스트림을 버렸다.
- `dlp sync run`은 manual 스트림을 건드리지 않고, 다시 돌렸는데 맞추지 못하면 이전의 자동 결과를 유지한다
  (unsynced로 내리지 않는다). 새로 맞추면 자동 결과는 바뀌고 사람 조정값은 유지된다.
- 장갑 동기화 신호는 `sync.yaml glove.pressure_prefixes` 접두사로 고른 압력 채널의 합이다(시각·IMU·온도 제외).
- 상호상관 신뢰도의 잔차 척도(예전 하드코딩 5 ms)를 `audio_xcorr`·`motion_xcorr.residual_scale_ms`로 옮겼다.
- `SyncPolicy.methods`의 키는 `StreamKind`만 받고, 바디캠(기준)은 받지 않는다.
- 오디오 두드림으로 120초 녹화의 드리프트를 10 ppm 안으로 추정하는 회귀 테스트를 더했다. 장갑(100 Hz)은
  샘플 간격이 앵커 오차라 드리프트 정밀도는 이보다 낮다(시각 오차는 1프레임 안).

## 남은 일

- `dlp_prelabel.lift3d`는 hand21 이름이 없는 점을 `joint_<번호>`로 쓴다. 정책 `hand_points`를 이름 붙은 점
  (손목·손가락 끝) 밖으로 넓히면 온톨로지 `hand_joints`에도 더해야 한다.
