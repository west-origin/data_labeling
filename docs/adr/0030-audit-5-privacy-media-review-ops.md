# ADR 0030: 5차 검수 — 프라이버시·수집·검수 운영·운영 지표 수정

- 상태: 채택
- 날짜: 2026-10-07
- 관련: ADR 0003(수집), 0005·0023·0024(프라이버시 게이트), 0006(검수 도구 변환), 0014(검수 운영),
  0015·0019(재실행·라벨 이력), 0020·0021(운영·감사 기록), 0029(검수 완료 판정)

## 배경

5차 검수에서 프라이버시·수집·검수 운영·운영 지표 범위의 버그 의심 11건을 코드로 확인했다. 모두 실제
버그였고, 계약(`dlp_schema`)과 DB 스키마는 바꾸지 않고 고쳤다.

## 결정

1. **탐지 0개 스트림의 재탐지** (`dlp_privacy.runner`): 블러가 하나도 나오지 않은 스트림은 DB 이력에
   모델 버전이 남지 않아 `dlp privacy detect`마다 영상 전체를 다시 탐지했다(CPU OWLv2 프레임당 수 초).
   결과 0개로 끝난 탐지·모델 버전을 원본 버킷의 파생 객체 `sessions/<세션>/derived/privacy_detect/
   <스트림>.json`(감사 저장소로 읽고 쓴다)에 남기고, 이력의 버전과 합쳐 건너뛸지 정한다.
   - 블러를 낸 버전은 표시에 넣지 않는다. 표시는 저장소 쓰기라 DB 트랜잭션이 되돌아가도 남으므로,
     넣으면 라벨 없이 "끝났다"고 남아 영영 다시 돌지 않는다.
   - 표시 덕분에 끝난 경우에는 지워지지 못한 이전 버전 블러(stale)가 없을 때만 건너뛴다. 있으면 영상은
     다시 탐지하지 않고 삭제 레코드만 다시 쓴다.
   - 스키마 변경을 피하려고 DB 대신 객체 저장소를 골랐다. 원본 영상에서 나온 판단이라 원본 버킷에 둔다.
2. **GPMF 패킷 길이 없음** (`dlp_media.imu.fill_durations`): packet.duration이 없으면 샘플이 패킷
   시작에 몰려 `ImuData` 시각 순증가 검증이 실패하고 수집 전체가 멈췄다. 길이를 다음 패킷 시작까지로,
   마지막 패킷은 길이 중앙값으로 채운다. 채울 수 없으면(패킷 하나, 시작 시각 역순) 경고를 남기고 내장
   IMU 스트림만 만들지 않는다.
3. **감사 길이** (`dlp_privacy.audit.stream_duration_ms`): 잔여 누락 감사 후보와
   `dlp ops privacy-audit`가 3인칭 스트림에도 세션(바디캠) 길이를 써 시간당 누락률이 틀렸다. 바디캠은
   세션 길이(같은 PTS 인덱스에서 나온 값), 그 밖의 영상은 그 스트림의 PTS 인덱스를 원본 버킷에서 읽어
   계산한다. PTS 인덱스가 없으면 세션 길이로 대신하지 않고 실패한다. 두 명령에 `--store`를 더했다
   (바디캠이 아닐 때만 감사 저장소를 만든다).
4. **수기 검수 시간 출처** (`dlp_ops.metrics.work_source`): `dlp ops log-work`가 작업 키가 있으면
   무조건 `cvat`으로 기록했다. 작업 키 접두사(`cvat:` / `label_studio:`)로 정하고, 키가 없으면
   `manual`, 모르는 접두사는 거부한다.
5. **표본 판정** (`dlp_review.ops.sampling.judge`): 검수 전에 표본이 모두 지워지면(판정할 표본 0개)
   결함 0 → 합격으로 봐 사람이 보지 않은 나머지가 `sample_verified`가 됐다. 이제 불합격이며,
   `finish_assignment`가 보류 라벨 전수 재검수 배정을 만든다.
6. **블라인드 편향 기준** (`dlp_review.ops.runner.shown_prelabels`): 프리라벨 편향을 이력 전체의 모델
   레코드(지워진 이전 버전·삭제 레코드 포함)로 쟀다. 짝 표준 배정 작업의 `sent_label_ids`(실제로 보낸
   라벨) 중 모델 라벨을 쓰고, 그 열이 없던 예전 작업은 작업 생성 시점의 운영 현재 라벨로 대신한다.
7. **작업 없는 배정 마무리** (`finish_assignment`): 작업이 하나도 없으면 `any([])`가 거짓이라 done이
   됐다. 작업이 있고 모두 수집됐을 때만 마무리한다.
8. **CVAT 키포인트 가시성** (`dlp_review.cvat`): `dlp_visibility` 값 수가 점 수와 다르거나 정수가
   아니면 `IndexError`·`ValueError` 대신 `CvatFormatError`(수집하지 않음)로 멈춘다. 모자란 값을 지어내거나
   남는 값을 버리지 않는다.
9. **QA 배정 상태** (`dlp_review.ops.assign.plan_qa`): `model_copy`가 검증하지 않아 status가 문자열
   `"open"`으로 남았다. `AssignmentStatus.OPEN`을 넣는다.
10. **검수 완료 판정자** (`dlp review verify`): `--actor`·`DLP_ACTOR`가 없으면 `"unknown"`으로 기록했다.
    다른 명령처럼 `current_actor()`(OS 사용자 대체)를 쓴다 (ADR 0029 결정 2의 실행자).
11. **CI CVAT 헬스체크**: `services-cvat` 작업이 CVAT를 보지 않는 `make health`를 돌렸다.
    `dlp services check --include-cvat`로 CVAT까지 확인한다. 다른 작업은 그대로다.

## 결과

- 결정마다 회귀 테스트를 더했다 (서비스가 필요한 것은 `@pytest.mark.services`).
- 원본 버킷에 탐지 표시 객체가 하나 늘었다. 원본 보관 만료로 세션 원본을 지울 때 함께 지운다.
- `dlp privacy audit-sample`과 `dlp ops privacy-audit`은 3인칭 후보가 있으면 원본 버킷에서 PTS 인덱스를
  읽어 `raw_access_log`에 읽기 기록이 남는다.
