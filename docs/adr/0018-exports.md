# ADR 0018: 내보내기 (COCO, 구간 JSON, LeRobot)

- 상태: 채택
- 날짜: 2026-10-07
- 관련: WP15. 계약 변경: ExportRecord.label_states (Alembic 0008), 구간 JSON 파일 계약
  `dlp_schema.export.IntervalFile` (`schemas/export_intervals.schema.json`, `make schemas`)

## 결정

1. **내보내기는 데이터셋 버전에서 만든다.** 라벨은 버전 스냅샷(lakeFS 커밋)의 이력에서 읽어 같은 버전이면
   같은 내용이 나온다. 세션은 버전의 정책 분할(기본 학습·검증, 골든셋은 기본에서 뺀다) 중 **내보내는 시점에**
   사용 중지되지 않은 것이다.
2. **검증 정책** (`config/policies/export.yaml label_states`): 사람이 만든 라벨은 늘 넣고, 모델 라벨은 사람
   승인·수정·표본 검증 상태만 넣는다. 미검수는 `--include-unreviewed`(또는 defaults
   `export.include_unreviewed`)로만 넣는다. 라벨마다 검증 상태를 그대로 표시하고, 적용한 정책을 manifest와
   내보내기 이력(`exports.label_states`)에 남긴다. 운영 라벨만 쓰므로 오류 삽입·측정 레코드는 빠진다.
3. **블러 라벨과 원본은 내보내지 않는다.** 블러 라벨은 정책 검증기가 제외를 강제한다. 영상은 라벨링 버킷의
   블러본만 읽고, 결과의 텍스트 파일에 원본 버킷 위치가 있으면 실패한다. 검수자 ID는 넣지 않는다
   (작업자·장소 ID는 가명).
4. **COCO**: 정답 키프레임 시각마다 이미지 하나(블러본의 그 시각 프레임 JPEG). 키프레임이 블러본 프레임
   시각에 없으면 그 주석은 버리고 센다 (보간한 값을 정답처럼 내보내지 않는다). 범주는 온톨로지 객체 +
   키포인트 범주(hand: hand21, person: coco17). 주석에 track_id·label_id·검증 상태·출처를 붙인다.
5. **구간 JSON** (`dlp-intervals` 1.0): 세션마다 파일 하나, 행동·상위 구간·공백·손 상태·객체 상태·이벤트·관계·
   커버리지·설명. JSON Schema를 함께 공개한다.
6. **LeRobot (v3.0)**: 세션의 바디캠 블러본 하나가 에피소드 하나다. 고정 프레임률(정책 fps) 시각마다 그 시각에
   보이던 블러본 프레임을 PTS 인덱스로 고른다 (가변 프레임 영상). 특징은 손 21관절 2D(COCO 보임 값),
   손 3D 점, 쥔 도구 작용부 3D, 손 상태, 도구-표면 접촉, 동사, 묶음별 검증 등급, task(작업 구간, 없으면 도메인).
   키포인트·궤적은 정책 간격 안에서만 보간하고 그 밖은 값 0·표시 0이다. 어휘는 `meta/dlp_vocab.json`,
   에피소드-세션 대응은 `meta/dlp_episodes.json`.
7. **LeRobot은 공식 쓰기·읽기 API로 다루고, 격리된 일회용 환경에서 돌린다.** 형식을 직접 쓰면 버전마다
   어긋날 위험이 크다. 그러나 lerobot은 PyTorch를 끌어오고, 같은 잠금에 넣으면 작업공간 전체가 av 19→15,
   numpy 2.5→2.2, pandas 3→2.3으로 내려간다. 그래서 `uv run --no-project --with lerobot[dataset]==0.6.1 ...`
   (PyTorch CPU판)로 `scripts/lerobot_write.py`를 돌리고, 쓴 뒤 `scripts/lerobot_check.py`가 공식 로더로 다시
   읽어 에피소드 수를 확인한다 (버전은 정책 `lerobot.env`에 고정). 이 테스트는 `isolated_env` 표시로 기본
   검사에서 빼고 `make test-isolated`와 CI 서비스 작업에서 돈다.
   (0021에서 고정 환경으로 바뀜: `uv run --no-project --with ...` 대신 `scripts/lerobot-env/uv.lock`에 전이
   의존성까지 고정하고 `uv run --project scripts/lerobot-env --locked --isolated`로 돈다. ADR 0021 5항.)
8. **내보내기 ID**는 데이터셋 버전·형식·대상·검증 정책·분할·시각의 해시다. 같은 시각이라도 정책이 다르면
   다른 폴더(`exports/<ID>/`)에 쓴다.

## 결과

- `dlp export coco|intervals|lerobot <데이터셋 버전> --target <대상> [--include-unreviewed] [--split]`.
- 완료 기준 테스트: pycocotools로 읽고 정답을 예측으로 넣은 COCOeval AP 1, LeRobot 공식 로더로 읽은 프레임이
  만든 특징과 일치, 기본 정책에서 미검수 0건·사용 중지 세션 0건, 결과에 원본 위치·검수자 ID 없음.

## 갱신 (ADR 0019)

2차 전체 검수 정정이 이 결정을 보강한다. 바뀐 내용은 ADR 0019를 따른다.
