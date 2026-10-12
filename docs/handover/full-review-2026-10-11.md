# 2026-10-11 전수 검수 결과

[인수인계 첫 화면](README.md) · [알려진 문제 레지스트리](issues.yaml) · [남은 과제](known-issues.md)

기준: 커밋 `1ead7b1`(검수 화면 테마 `238f8aa` 이전). 6개 영역(schema / review / privacy·media·sync /
prelabel·actions·relations·models / evaluation·training·active·datasets / cli·export·ops·fixtures·스크립트)의
src·tests 전 파일과 관련 정책 YAML을 처음부터 끝까지 읽었다. 각 발견은 코드를 다시 읽거나 실행해 확인한 것만
남겼다. 서비스(docker) 대상 테스트는 검수 중 돌리지 않았고, 서비스 없이 도는 단위 테스트는 영역별로 모두 통과했다.

- 발견은 레지스트리 [issues.yaml](issues.yaml)의 `F-<영역>-nn`이다. 상태·심각도·관련 파일은 레지스트리가 기준이고,
  이 문서는 설명이다. 이미 기록된 문제(T1~T5, audit-notes)를 다시 확인한 것은 새 ID를 만들지 않고 기존 항목에 적었다.
- `repro`가 있는 항목은 [tests/test_known_defects.py](../../tests/test_known_defects.py)에 올바른 동작을 단언하는
  `xfail(strict=True)` 테스트로 재현했다. 고치면 XPASS가 실패로 보고되므로 표시를 지우고 레지스트리를 `fixed`로 바꾼다.
- 이번 검수는 결함을 고치지 않았다 (발견 기록과 재현만).

## 요약

| 심각도 | 건수 | 대표 항목 |
| --- | --- | --- |
| 상 | 4 | F-REV-01 정책 파일 무시, F-REV-02 블러 변경 후 배포 매체 미회수, F-PAR-01·02 프리라벨 전체 롤백 |
| 중 | 21 | 양손 행동·보호 검사·이관(SCH), LS·CVAT 수집 중단(REV), 재수집 시간대·홀수 크기 렌더(PMS), 수정률 클래스(ETA) |
| 하 | 54 | 주석·테스트 정확성, 작은 입력 경계 |

재확인한 기존 기록: T1(학습 CLI 실패), T2(정답 수정 재학습 누락), T3(철회 세션 모델 배포), T4(deploy 게이트),
T5(골든 정답 변동), A-10(lead_seconds), A-15(MLflow 조회), A-17(중복 코드).

## 상 등급

- **F-REV-01** `review_cmds._setup`이 `ReviewSetup`에 선택된 정책의 `cvat`·`media`를 넘기지 않는다. 라이브러리
  기본 로더가 Git의 `review.yaml`(`cvat.users: {}`)을 읽어, `DLP_REVIEW_POLICY_FILE`에 계정 연결을 둬도 파일럿 안내의
  `dlp review create --stage privacy --assignee pilot-privacy`가 "CVAT 계정이 없습니다"로 실패한다. labeling CVAT
  작업은 미배정으로 만들어지고 웹훅은 담당자 불일치로 거부한다. 수정: `_setup`에서 정책을 읽어 `cvat_config`·`media`로 넘긴다.
- **F-REV-02** 승인 뒤 블러 재검수로 블러가 바뀌면 승인 해제·렌더 기록 무효화만 한다. 이미 만든 라벨링 작업의
  워터마크 사본과 LS 서명 URL은 남아, 일반 라벨러가 새로 찾은 얼굴이 블러 안 된 영상을 계속 본다.
  수정: 그 스트림의 열린 라벨링 작업을 회수·폐기·재생성하고 기록한다.
- **F-PAR-01** 같은 entity_id의 바디캠 손 트랙이 둘(승인된 이전 버전 + 새 버전)이면 `lift_tracks`가 같은 시각 표본을
  두 번 넣어 3D 라벨 검증이 실패하고 프리라벨 트랜잭션 전체가 롤백된다. 다시 돌려도 같다.
  수정: (entity_id, part)마다 입력 트랙 하나를 `pick_hand_tracks`와 같은 우선순위로 고른다.
- **F-PAR-02** 장갑 시각을 마스터로 바꾼 뒤 0 미만·세션 밖 구간을 자르지 않는다. 장갑을 먼저 켜 오프셋이 음수면
  음수 시각 접촉 라벨이 검증에서 실패해 프리라벨 전체가 롤백된다. 수정: [0, 세션 길이]로 자르고 빈 구간은 버린다.

## 중 등급

- **F-SCH-01** 행동 노드를 action_id 하나로 식별해, 양손 행동(두 손 레코드가 같은 action_id, ADR 0002)이 있으면
  상위 구간과 함께 저장할 수 없다. 수정: 같은 action_id·다른 손은 한 노드로 묶고 같은 손 중복만 위반으로 본다.
- **F-SCH-02** 보호 검사가 지우려는 레코드 자체의 상태만 봐, 승인된 원래 트랙의 모델 자식(착용자 사본)을 모델이
  지우면 통과하고 원래 트랙도 운영 라벨에서 사라진다. 수정: 조상 사슬에 보호 상태가 있고 다른 현재 후손이 없으면 거부.
- **F-SCH-03** 이관의 집합 검사를 비운영 레코드까지 섞어 돌려, 오류 삽입 사본(같은 action_id)이 있으면 운영 라벨까지
  재검수 목록으로 간다. 수정: 운영 라벨·배정별 사본·측정 레코드를 따로 검사한다.
- **F-SCH-04** 0014 트리거 테스트가 앞 단계와 같은 label_id를 써서 트리거가 없어도 PK 중복으로 통과한다.
- **F-REV-03** LS에서 그릴 수 있지만 수집할 수 없는 구간(접촉→none 변경, 새 tool·object·person 손 상태, 새 객체 상태)이
  하나라도 있으면 ValidationError·ValueError로 작업 수집 전체가 멈춘다. 수정: 종류 변경 시 호환되지 않는 필드를 비우고,
  대상 ID 입력란을 두고, 변환 오류는 구간 정보를 담은 TaskError로.
- **F-REV-04** CVAT에서 새로 그린 손 골격은 hand가 없어 ValidationError. 수정: `kp_hand21_left/right` 분리 또는 명확한 오류.
- **F-REV-05** `_new_id`의 `[-128:]` 자르기가 `-`·`:` 등으로 시작하는 ID를 만들어 수집이 실패한다(A-09를 확정).
  수정: `derived_id`처럼 앞부분과 해시로 줄인다.
- **F-REV-06** LS 수집이 최신 주석 작성자를, 웹훅이 `completed_by`를 담당자와 대조하지 않는다. 다른 LS 사용자의 제출이
  담당자의 검수로 기록된다. 수정: LS 사용자 연결을 정책에 두고 검증·필터한다.
- **F-REV-07** 담당자 없는 배정에 "unassigned"를 실제 담당자로 넣어 워터마크·경로·검수자 ID가 모두 그 문자열이 된다.
- **F-REV-08** LS 작업의 영상·시계열 서명 URL이 7일 고정이고 갱신 경로·정책 값이 없다.
- **F-PMS-01** 재수집 충돌 판정이 JSON 문자열을 비교해, +09:00으로 적은 recorded_at과 DB에서 UTC로 읽은 같은 순간이
  충돌로 판정된다. 수정: UTC 정규화 또는 객체 비교.
- **F-PMS-02** 홀수 크기 원본은 수집·프록시는 되지만 libx264 yuv420p 블러 렌더에서 실패한다. 수정: 수집에서 거부하거나 렌더에서 패딩.
- **F-PMS-03** 반사면 영역의 폭·높이가 1px 미만이면 빈 crop이 YuNet에 들어가 `privacy detect` 전체가 중단된다.
- **F-PMS-04** `config/defaults.yaml`이 프록시(블러 전 영상)를 라벨링 버킷에 둔다고 설명한다. 믿고 따르면 원본이 노출된다.
- **F-PAR-03** COCO person 박스 접촉을 `contact_target_kind=object`로 기록한다(온톨로지상 person + body_part 필수).
- **F-PAR-04** 행동 단계가 같은 손 hand21 트랙 중 마지막을 써, 프리라벨 접촉 단계(검수 트랙 우선)와 기준이 다르다.
- **F-ETA-06** 액티브 러닝 수정률을 사람이 고친 뒤의 클래스로 세어, 모델이 cup으로 낸 것을 mug로 고쳐도 cup 수정률이 0이다.
  수정: corrected는 모델 원본(`origin`)의 클래스로 센다.

## 하 등급

| ID | 위치 | 요약 |
| --- | --- | --- |
| F-SCH-05 | schema/tests/test_db.py | `match="TRUNCATE"`가 모든 SQL 오류 문자열에 맞음 |
| F-SCH-06 | label_validation.py | 기존 위반이 있으면 무관한 새 라벨까지 거부, 오류 ID도 엉뚱 |
| F-SCH-07~10 | repository·episode·tables·validation·`__init__`·session | 낡거나 빠진 주석·이름 목록 |
| F-SCH-11 | schema/tests/test_contracts.py | contact_held 테스트가 규칙을 검사하지 않음 |
| F-SCH-12 | repository·review_repository | 시간대 없는 datetime 미검사 |
| F-SCH-13 | migrations/versions | 영어 리비전 docstring·트리거 메시지 |
| F-REV-09 | ops/results.py | 측정 레코드를 배정이 아닌 (단위, 담당자)로 모음 |
| F-REV-10 | cvat·reconcile·collect | 깨진 dlp_meta·중복 원래 ID가 TaskError로 바뀌지 않음 |
| F-REV-11 | labelstudio.py | 이벤트 속성 입력란을 모두 비우면 원래 값이 남음 |
| F-REV-12 | verify.py | 잠금 없이 조건을 읽음 (TOCTOU) |
| F-REV-13·14 | ops/`__init__`·test_units | 주석·테스트 범위 |
| F-PMS-05 | privacy/pipeline.py | 반올림 ms가 같은 프레임 → 키프레임 중복 |
| F-PMS-06 | media/imu.py | 잘린 GPMF 패킷 하나로 수집 전체 실패 |
| F-PMS-07 | glove.py·imu.py | NaN 시각이 단조 검사 통과 |
| F-PMS-08·09 | probe.py·privacy policy | 낡은 문서 안내, 기본값 주석 |
| F-PMS-10·11 | sync pipeline·test | 실패 사유 뭉뚱그림, 약한 단언 |
| F-PAR-05 | actions pipeline·assemble | 없어진 절단 후보가 대표 경계가 됨 |
| F-PAR-06 | prelabel runner | 예측기 접촉이 검수된 손 상태와 겹침 검사 없음 |
| F-PAR-07 | actions/policy.py | VLM timeout이 정책 해시에 들어감 |
| F-PAR-08·09 | lift3d·runner | 박스 3D 표본 누락, 전신 트랙이 3D 입력 해시에 |
| F-PAR-10 | contact·boundaries | visibility 0 점 좌표 사용 |
| F-PAR-11~13 | lift3d·prelabel policy·relations test | 주석·일치 검사·테스트 범위 |
| F-ETA-08 | active/curation.py | 마스크·3D 궤적을 시간 구간처럼 변환 |
| F-ETA-09·10 | active/select·datasets/lineage | `--limit 0`, 학습 분할 하드코딩 |
| F-ETA-11~15 | eval tracking·datasets build·training·eval tests | 주석·테스트 이름 |
| F-COE-02~05 | 루트·cli·fixtures 테스트 | 주장과 다른 테스트 |
| F-COE-06 | review_cmds.py | 초안 온톨로지 갱신 뒤 이전 프로젝트 웹훅 미등록 |
| F-COE-07~10 | export_cmds·export source·ops_cmds·schema_cmds | 낡은 주석 |
| F-COE-11·12 | pilot_cmds·ops policy | 프로필 하드코딩, off_hours 범위 검증 없음 |

## 문제가 없었던 부분 (영역별 요약)

- schema: tables.py와 Alembic 0001~0014 일치, 라벨 불변·추가 전용 트리거, current_labels와 SQL 판정 일치, ADR 참조 실재.
- review: 웹훅 서명·크기·시간 제한, 원본 URI 차단, 블러 검수 권한·grant 순서, 수집 잠금 순서, reconcile 판정, 좌표 왕복.
- privacy·media·sync: 프레임 번호×간격 계산 없음(PTS 인덱스만), 감사 저장소 순서, 동기화 앵커·드리프트, 세대·철회.
- prelabel·actions·relations·models: 모델 해시·라이선스(forbidden 미사용), version_tag·정책 해시, 검수 전 라벨만 철회.
- evaluation·training·active·datasets: 지표가 참조 구현(pycocotools·TrackEval·MS-TCN·ActivityNet)과 일치, 분할 누수 차단.
- cli·export·ops: 블러 라벨 3중 차단, 가명 HMAC, manifest sha256, 추가 전용 감사 표, `set -eu` 셸 처리.

## 후속 발견 (상 등급 수정 중, 2026-10-11)

F-REV-02를 고치며(ADR 0039) 찾은 경계와 기존 불안정 테스트다. 위 요약 표 건수에 포함했다.

| ID | 심각도 | 내용 |
| --- | --- | --- |
| F-REV-15 | 중 | DB 롤백으로 기록 없이 남은 고아 CVAT·LS 작업과 라벨링 버킷 사본은 DB 기준 회수가 찾지 못한다 (R4) |
| F-REV-16 | 하 | 회수한 작업의 진행 중 주석은 스냅숏(`sessions/<세션>/revoked/`)으로만 남고 새 작업으로 이어받지 않는다 |
| F-REV-17 | 중 | 사용 중지 세션의 검수 매체 회수가 연결되지 않았다 |
| F-REV-18 | 중 | 승인 보류 명령이 없어, 사고 대응 중에도 블러 검수 수집 전까지 같은 렌더로 새 작업을 만들 수 있다 |
| F-REV-19 | 중 | 이미 내려받은 사본·도구 내보내기·버킷 이전 버전·배포한 라벨러 자격 증명은 DB 회수로 막지 못한다 (운영 점검) |
| F-PMS-12 | 하 | `test_rerender_repairs_damaged_release`가 인코더 출력 바이트 차이로 가끔 실패한다 (변경 전 코드에서도 재현) |
| F-REV-20 | 하 | 웹훅 부분 본문 시간 초과 테스트가 동시 부하에서 가끔 실패한다 (중 등급 수정 중 발견) |
| F-PAR-14 | 하 | 프리라벨 접촉 단계 입력 해시가 고른 손 트랙이 아닌 전체 트랙 ID 기준이다 (F-PAR-04 수정 중 발견) |
