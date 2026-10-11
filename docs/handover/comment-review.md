# 주석 집중 검수 — 2026-10-10

현재 구현과 어긋난 설명을 수정했다. 주석이 풍부하다는 점은 인수에 도움이 되지만, 탐지 세대 도입
이전 설명과 부작용·보안 보장을 과장한 표현이 남아 있어 일부 주석만 믿고 운영하기는 어려웠다.
이번 수정은 구현을 설명하는 문장을 바로잡는 작업이며, 서비스 기능·보안 한계를 해결한 작업은 아니다.

## 범위와 방법

- 추적 파일과 Git 제외 대상이 아닌 신규 파일 439개를 대상으로 목록을 수집했다.
- Python 294개 파일에서 docstring 2,392개와 주석 토큰 3,206개를 수집했다.
- 설정·스크립트에서 `#`가 포함된 줄 940개도 수집했다. 이는 고유한 주석 개수와 같지 않다.
- 상태·시간 단위·원본 접근·승인·멱등성·예외·부작용·미구현 여부와 관련된 의심 표현을 추려,
  프라이버시·검수·CLI·저장소·내보내기·동기화·학습의 실제 호출 경로와 대조했다.
- 전수 목록 수집과 검색 후 위험한 설명을 집중 대조했다. 모든 docstring을 문장별로 수동 검증한
  결과나 전체 설명에 오류가 없다는 보증은 아니다.

Python 24개 파일의 docstring 31개를 수정했고, 프라이버시 YAML과 Makefile 주석도 고쳤다.
CVAT 담당자 배정이 관리자·프로젝트 역할을 포함한 전체 열람 권한을 제한한다는 인상을 주는
주석도 고쳤다. 공개 계약 클래스의 설명이나 `Field(description=...)`, 실행 코드·설정값은 바꾸지 않았다.

## 확인하고 수정한 설명

| 문제 | 실제 동작에 맞춘 설명 | 주요 구현 |
| --- | --- | --- |
| 탐지 결과가 0개면 DB 이력이 없다는 옛 설명 | 0건도 완료 탐지 세대를 기록한다. 외부 0건 표시만 남으면 복구 후 새 검수를 요구한다 | [프라이버시 흐름](../../packages/privacy/src/dlp_privacy/__init__.py), [탐지 실행](../../packages/privacy/src/dlp_privacy/runner.py) |
| 승인 조건을 라벨 검수나 작업 생성 시각만으로 설명 | 스트림별 최신 완료 탐지 ID에 연결된 운영 검수 수집이 필요하다. 완료 이력 이전 데이터만 시각 규약을 쓴다 | [세대 판정](../../packages/privacy/src/dlp_privacy/generation.py) |
| 모든 탐지기가 모든 프레임에서 추론한다는 인상 | 모든 프레임을 순회하되 느린 탐지기는 간격 추론 후 결과를 유지한다. 새 대상·움직임의 누락 가능성이 있다 | [탐지 파이프라인](../../packages/privacy/src/dlp_privacy/pipeline.py) |
| 모든 검수 화면이 블러본을 사용한다는 설명 | 일반 라벨링 검수는 블러본, 원본 접근 권한자의 프라이버시 검수는 원본 프록시를 사용한다 | [검수 작업](../../packages/review/src/dlp_review/tasks.py) |
| 클라이언트 생성은 부작용이 없다는 설명 | CVAT 생성 중 로그인 API를 호출하며 통신·설정 오류가 전파된다 | [CLI 설정](../../packages/cli/src/dlp_cli/review_cmds.py), [REST 클라이언트](../../packages/review/src/dlp_review/clients.py) |
| 배정 잠금이 외부 작업의 중복까지 항상 방지한다는 설명 | 정상 커밋 경로는 직렬화하지만 DB 롤백으로 외부 작업은 삭제되지 않는다. 재시도 시 잔여 작업 확인이 필요하다 | [배정 작업 생성](../../packages/review/src/dlp_review/ops/runner.py) |
| 가명을 아무도 되짚을 수 없다는 보장 | 임의 키는 재계산을 막지만 내부 세션 대응표로 추적할 수 있다. 영상 내용으로 내보내기를 연결하는 것도 막지 못한다 | [가명 처리](../../packages/export/src/dlp_export/pseudonym.py) |
| DB를 쓰지 않으므로 순수 함수라는 설명 | 동기화는 로컬 영상을 읽고 행동 분류는 전달된 VLM 클라이언트로 외부 통신을 할 수 있다 | [동기화](../../packages/sync/src/dlp_sync/pipeline.py), [행동 분류](../../packages/actions/src/dlp_actions/pipeline.py) |
| 점검 함수는 모든 예외를 삼키며 전체 시간이 제한된다는 설명 | 연결·HTTP 실패를 결과로 바꾸지만 잘못된 대상의 ValueError는 전파된다. 소켓 timeout은 전체 점검 마감이 아니다 | [서비스 점검](../../packages/cli/src/dlp_cli/health.py) |
| `labels`를 실제 DB 테이블명으로 표기 | DB 테이블명은 `label_records`다. 모듈명·FiftyOne 필드명 `labels`는 해당 문맥에서 유효하다 | 프리라벨·관계·행동 실행 모듈 |
| 저장소 계층 밖 직접 SQL은 행 잠금뿐이라는 설명 | 전용 검수 조회·집계 등에도 직접 SQL이 있다. 라벨 저장은 온톨로지 의미 검증을 자동 수행하지 않는다 | [DB 저장소](../../packages/schema/src/dlp_schema/db/repository.py) |
| CLI 검수 정책은 항상 기본 YAML이라는 설명 | CLI는 DLP_REVIEW_POLICY_FILE을 선택할 수 있다. 라이브러리의 기본 로더와 구분했다 | [정책 선택](../../packages/cli/src/dlp_cli/review_config.py) |
| 재수집은 어떤 부작용도 없다는 설명 | 데이터 등록은 중복 변경하지 않지만 파일 확인·저장소 접근·감사 기록은 다시 수행할 수 있다 | [수집](../../packages/media/src/dlp_media/ingest.py) |
| 학습기 존재와 MLflow 실패 종료를 과하게 보장하는 표현 | 등록된 학습기는 oracle-stub뿐이며, 종료 기록 요청도 통신 실패 시 보장되지 않는다 | [학습 패키지](../../packages/training/src/dlp_train/__init__.py), [학습 루프](../../packages/training/src/dlp_train/loop.py) |

반사면 탐지 설명의 “착용자 얼굴이 거울에서만 보인다”는 단정과 Makefile의 오래된 가중치 용량도
고쳤다. 기존 [남은 과제](known-issues.md)의 보안·실제 학습기·실영상 성능 문제는 그대로 남는다.

## 검증

- 수정 전후 Python 294개 파일의 AST를 비교했다. docstring을 제외한 실행 코드 변경은 0건이다.
- 수정한 프라이버시 YAML을 파싱해 비교했다. 설정값 변경은 0건이다.
- `make check` 성공: 린트·형식·타입·온톨로지·생성 JSON Schema·라이선스 분류 검사가 통과했다.
  기본 테스트는 **489개 통과, 1개 생략, 115개 선택 제외**, 181.93초였다. 생략은
  `make export-models`가 필요한 깊이 모델 검사다. 선택 제외는 서비스·격리 환경 검사다.
- 마지막 주석 보완 후 린트·형식과 Git 공백 검사도 확인했다. 보고서의 로컬 링크 19개가 유효하다.
- 서비스 통합 테스트는 이번 주석 변경에서 재실행하지 않았다. 이전 검수의 113개 통과 기록은
  [남은 과제](known-issues.md)에 있으며 이번 실행 결과와 구분한다.

자동 목록·전후 비교·검사 로그는 클라우드의
`/workspace/.cloud-onboarding/data-labeling/comment-review-2026-10-10/`에 보관했다.
인수자는 이 문서의 결론과 코드·계약·테스트를 함께 확인해야 한다.
