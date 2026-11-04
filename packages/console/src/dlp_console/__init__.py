"""운영 콘솔 (ADR 0061): 기존 `dlp` 명령을 브라우저에서 작업으로 돌리는 화면.

`dlp console serve`로 띄운다 (`server.serve`). 콘솔은 DB를 읽기만 하고 도메인 로직을 두지 않는다.
모듈 지도:

- `server`: uvicorn 실행·주소 묶기 검사. `app`: FastAPI 경로와 요청 검사(토큰·Host).
- `settings`: `config/console.yaml`과 실행 값. `jobs`: `dlp` 자식 프로세스 작업 실행·기록.
- `status`: 세션 상태 읽기. `steps`·`global_steps`: 단계 활성 판정과 명령 만들기.
- `upload`: 영상 올리기 → 준비·수집 명령. `preview`: 블러본 서명 URL·겹쳐 그릴 트랙.
- `training`: 재학습 회차·리포트. `pretrain`·`imports`: 외부 가중치 검사·가져오기.
- `hints`: 실패 로그 → 다음 할 일 안내. `redact`: 화면 글에서 원본 위치·비밀값 가리기.
- `static/`: 빌드 단계 없는 화면 (HTML·JS·CSS).
"""
