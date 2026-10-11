# 전체 파일 검사와 주석 정정 — 2026-10-10

소스·테스트·설정·실행 스크립트를 파일 목록으로 고정해 자동 검사를 수행하고, 주요 동작 설명을
구현과 대조했다. 이번 검사에서 **35개 파일의 설명을 정정**했다. 실행 코드의 AST, 설정값,
도구 지시문은 바뀌지 않았다. 기능·보안 문제를 모두 해결했다는 판정은 아니다.

## 검사 범위와 증거

- Python **294개**를 포함한 소스·설정 **349개 파일**. 패키지 소스·테스트, 루트 테스트,
  마이그레이션, YAML/TOML, 셸, Makefile, CI, Dockerfile, Mako 템플릿, `.env.example`,
  `.python-version`을 포함한다. 실제 `.env`·가중치·영상·외부 소스·가상환경은 제외했다.
- 최종 상태에서 docstring **2,392개**, Python 주석 토큰 **3,213개**, 기타 파일의 `#` 포함 줄
  **1,010개**를 수집했다. 마지막 수치는 문자열·지시문도 포함하므로 고유한 설명 주석 수가 아니다.
- 모든 수집 지점에 구조 검사를 적용했다. Python 구문·주석 추출, 함수 docstring의 Args/인자
  절에서 인식한 인자명과 시그니처, 구체적인 저장소 경로 참조, 도구 지시문 분류를 검사했다.
  이 검사기가 탐지한 인자명·경로 불일치는 **0건**이다. 일반 문장의 의미를 자동으로 증명하지는 않는다.
- 상태 전이·원본 접근·권한·멱등성·시간 단위·추론 간격·학습·평가·관계 계산의 설명은 관련 함수와
  호출 경로를 추가 대조했다. 전체 테스트를 실행해 기존 동작도 검증했다.
- [파일별 검사 목록](full-comment-review-files.csv)은 349개 파일을 한 행씩 기록한다 (2026-10-10 시점 기록.
  그 뒤 추가된 파일은 빠져 있다. 지금의 전체 범위는 `make review-scope`로 만든다).
  자동 검사와 이번에 정정한 설명을 구분한다. 수정하지 않은 파일을 문장별 수동 검수 완료로
  표시하지 않았다. **모든 docstring의 모든 문장을 수동 검증했다는 보증은 아니다.**

Markdown 문서는 이번 검사 결과를 정리하는 자료이며, 전수 주석 집계 대상에는 포함하지 않았다.
생성 JSON Schema는 `make check`의 재생성 일치 검사로 확인했다. 문서의 계획이나 예시를
구현된 기능의 증거로 취급하지 않았다.

## 추가로 확인해 수정한 설명

| 설명 문제 | 실제 구현에 맞춘 정정 | 확인할 파일 |
| --- | --- | --- |
| CVAT 담당자 지정이 다른 작업의 열람을 막는다는 보장 | 프로젝트·조직·사용자 권한도 열람 범위에 영향을 준다 | [클라이언트](../../packages/review/src/dlp_review/clients.py), [검수 정책](../../config/policies/review.yaml) |
| `mutable=False` 속성을 검수자가 수정할 수 없다는 설명 | 프레임별 값 변화 제한이며 보호된 보안 필드가 아니다 | [CVAT 왕복](../../packages/review/src/dlp_review/cvat.py) |
| 새 객체 상태를 CVAT에서 개체 선택으로 연결한다는 설명 | 새 구간은 개체 참조를 몰라 거부한다. 기존 메타데이터의 개체 참조만 왕복한다 | [Label Studio 왕복](../../packages/review/src/dlp_review/labelstudio.py) |
| 같은 검수 결과는 언제 재수집해도 같은 ID라는 설명 | ID 해시에 시각도 들어간다. 재수집 중복 방지는 저장된 작업 상태가 담당한다 | [수집 결과 비교](../../packages/review/src/dlp_review/reconcile.py) |
| 감사 행위자와 grant 기록을 인증·열람 증거로 설명 | 행위자는 자기 신고값이며 grant는 업로드 전 제공 시도다 | [접근 감사](../../packages/media/src/dlp_media/audit.py), [감사 리포트](../../packages/ops/src/dlp_ops/audit.py) |
| 사용 중지 후 재학습·구매자 삭제 요청을 자동 실행하는 듯한 설명 | 영향 목록을 반환한다. 재학습·외부 삭제 요청은 운영자가 별도로 수행한다 | [계보](../../packages/datasets/src/dlp_datasets/lineage.py), [내보내기](../../packages/export/src/dlp_export/runner.py) |
| 모든 평가 입력과 내보내기 결과가 데이터셋 버전만으로 고정된다는 설명 | 골든 평가 입력은 DB 골든셋이며 내보내기는 현재 철회·렌더 상태와 가명 키도 확인한다 | [데이터셋](../../packages/datasets/src/dlp_datasets/__init__.py) |
| 짧은 해시가 서로 다른 입력의 1:1 대응을 보장한다는 설명 | 결정적이지만 해시 충돌 가능성을 없애지는 않는다 | [공통 계약](../../packages/schema/src/dlp_schema/common.py) |
| 동결 계약이 중첩 컨테이너까지 불변이라는 인상 | 속성 대입을 막으며 중첩 dict/list 변경까지 막지는 않는다 | [공통 계약](../../packages/schema/src/dlp_schema/common.py) |
| 전수 검수 판정이 정책을 자동 전환한다는 설명 | full/sample 판정값을 반환하며 작업 생성·승인 정책을 자동 변경하지 않는다 | [프라이버시 감사](../../packages/privacy/src/dlp_privacy/audit.py), [기본 정책](../../config/defaults.yaml) |
| 모든 프레임에 모델 추론을 실행한다는 인상 | 프레임마다 탐지기를 호출하되 내부 stride·캐시를 사용할 수 있다 | [탐지 흐름](../../packages/privacy/src/dlp_privacy/pipeline.py) |
| QR 첫 검출 간격이 실제 등장 시각의 엄밀한 오차 상한이라는 보장 | 연속 검출·선형 시계 전제의 추정이며 읽기 실패·탐색 경계 오차는 포함하지 않는다 | [슬레이트](../../packages/sync/src/dlp_sync/slate.py), [동기화 정책](../../config/policies/sync.yaml) |
| 접촉 커버리지·카메라 이동 불변성의 전제 누락 | 합친 비접촉 틈도 면적에 포함한다. 같은 시각·같은 좌표계의 점을 비교해야 한다 | [커버리지](../../packages/relations/src/dlp_relations/coverage.py), [기하](../../packages/relations/src/dlp_relations/geometry.py) |
| 원본 참조 검사·가명화가 모든 정보 노출을 막는다는 인상 | 원본 검사는 세 바이트 패턴만 찾는다. 가명은 내부 대응표·영상 내용까지 숨기지 않는다 | [내보내기 입력](../../packages/export/src/dlp_export/source.py), [내보내기 정책](../../config/policies/export.yaml) |
| 변경해도 소비되지 않는 정책값이 동작을 바꾼다는 설명 | 분할 단위는 고정이며 actions 배포 연결과 이벤트 작성자 권한 검사는 미구현이다 | [분할 정책](../../config/policies/dataset.yaml), [학습 정책](../../config/policies/training.yaml), [이벤트](../../config/ontology/v1/events.yaml) |
| 테스트 범위·단위·기반 이미지 설명 오류 | 개인정보 합성 시험은 7종 중 6종, 정규화는 채널 평균·표준편차 기준이다. 오프셋 시험은 독립 Stream이며 Dockerfile은 Python 기반이다 | [합성 시험](../../packages/fixtures/tests/test_video.py), [모델 시험](../../packages/models/tests/test_models.py), [큐레이션 시험](../../packages/active/tests/test_curation.py), [Dockerfile](../../services/mlflow/Dockerfile) |

수정량은 Python 25개 파일의 docstring 27개·주석 지점 16개, YAML 9개 파일의 주석,
Dockerfile 1개 파일의 주석이다. 새 설명을 여러 줄로 나눈 경우가 있어 수정 지점 수와 줄 수는 다르다.

## 실행 검증

| 검사 | 결과 |
| --- | --- |
| `make check` | 성공. 린트·형식·strict 타입·온톨로지·JSON Schema 일치·모델 상업 사용 분류 검사 통과 |
| 기본 테스트 | **489 통과, 1 생략, 115 선택 제외**, 189.04초. 생략은 깊이 ONNX 변환 미실행 |
| `make test-services` | **113 통과, 492 선택 제외**, 125.47초. DB·저장소·검수 도구·종단·동시성 회귀 포함 |
| 최종 설명 보완 후 린트·형식·공백 검사 | 성공 |
| 설정·스크립트 구문 | YAML·TOML·셸 구문 및 생성 JSON 파싱 성공. Compose 특수 태그는 구조만 해석하며 서비스 의미 검증과 구분 |
| 수정 전후 비교 | Python 294개에서 docstring 제외 AST 변경 0, YAML 설정값 변경 0, Dockerfile 명령 변경 0, 도구 지시문 변경 0 |

서비스 테스트는 기본 테스트와 분리된 선택이다. 이번 실행으로 통과한 테스트는 합계 **602개**다.
LeRobot 격리 테스트·깊이 모델·실제 VLM 서버·실제 청소 영상 정확도·다중 고객 부하는 검증하지 않았다.

## 남은 판단과 인수 시 주의점

설명을 신뢰하고 읽기 위한 근거는 늘었지만, 현재 제품 범위는 단일 환경 파일럿이다.
실제 재학습기는 `oracle-stub`뿐이고, SaaS 인증·테넌트 격리·영속 작업 큐·장애 복구·현장 품질은
별도 구현·검증이 필요하다. 기존 의존성 보안 이슈를 이번 주석 정정으로 해결하거나 새 보안 검사로
재확인하지 않았다. 확정된 과제는 [남은 과제](known-issues.md)가 기준이다.

생성 계약 설명에는 기존 표현이 남아 있다. `Ms` description의 마스터 기준 표현은 공간 라벨의
스트림 PTS에도 사용되고, `RawAccessEvent` description의 grant “보여 줌”은 실제 열람 증거가 아니다.
소스 주석에 해석을 명시했다. 생성 Schema 문구 자체의 정리는 계약 문서 변경으로 별도 관리해야 한다.

원본·가명화·슬레이트·커버리지 설명을 정정했다고 관련 알고리즘의 한계가 사라지지는 않는다.
특히 바디캠 파일럿에서는 원본 권한자의 개인정보 전수 검수와 실제 촬영 정답 비교가 필요하다.

자동 목록·지점별 검사·수정 전 스냅샷·AST/설정 비교·실행 로그는 클라우드의
`/workspace/.cloud-onboarding/data-labeling/full-comment-review-2026-10-10/`에 보관했다.
파일별 목록은 이 문서와 함께 저장소에 남긴다. 이전 검사는 [주석 집중 검수](comment-review.md)이며,
그 실행 결과와 이번 결과를 구분한다.
