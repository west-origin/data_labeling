# 개발자 안내

전문가 인수인계의 시작점은 [docs/handover/README.md](handover/README.md)다.
기존의 긴 안내를 목적별 문서로 나눴으며, 이 경로는 기존 링크를 위한 안내로 유지한다.

| 확인할 것 | 문서 |
| --- | --- |
| 전체 흐름·기능 구현 상태·읽는 순서 | [인수인계 첫 화면](handover/README.md) |
| 코드 위치·실제 import 의존·상태/시간/원본 경계 | [구조 지도](handover/architecture.md) |
| 새 환경 설치·검사 범위·현재 클라우드와의 차이 | [환경 준비](handover/development.md) |
| 변경 영향·검사 위치·실패/재시도·잠금 | [변경 및 복구 안내](handover/change-guide.md) |
| 확인된 미해결 문제·우선순위·완료 기준 | [남은 과제](handover/known-issues.md) |
| 이전 검수의 재현 전 후보 | [이전 검토 메모](handover/audit-notes.md) |
| 실제 청소 영상 첫 실행 | [파일럿 안내](weekend-video-pilot.md) |
| 과거 결정과 변경 이유 | [ADR 색인](adr/README.md) |

현재 상태는 인수인계 문서와 코드로 판단한다. 제품/구현 계획은 요구사항이고 ADR은 당시의 기록이다.
코드 변경 규칙과 필수 검사는 [CLAUDE.md](../CLAUDE.md)를 따른다.
