"""프라이버시 게이트 (WP5, ADR 0005·0019·0023·0024).

원본 영상(얼굴·문서·화면·송장 등이 그대로 보이는 영상)을 일반 라벨러에게 보여 주기 전에 반드시
거치는 관문이다. 이 패키지가 만든 **블러본**만 검수 작업·행동 VLM·큐레이션·내보내기에 쓰인다.

게이트 흐름 (명령은 `dlp privacy …`, 구현은 `dlp_privacy.runner`):

1. 탐지 `dlp privacy detect <세션>` → `runner.detect_session`
   - 원본 버킷(`dlp-raw`)에서 영상 스트림을 감사 저장소(`AuditedStore`)로 받는다
     (접근 기록이 남는다).
   - `pipeline.detect_video`가 모든 프레임에 탐지기(`detectors/`)를 돌리고, `tracker`로
     트랙을 만들고(병합·IoU 매칭·보간·유지), 트랙마다 `blur_track` 라벨(모델 출처, 미검수)을
     DB `label_records`에 쓴다.
   - 사람이 먼저 봐야 할 **검수 우선 구간**(`review.ReviewSegment`)을 원본 버킷
     `sessions/<세션>/derived/privacy_review/<스트림>.json`에 쓴다.
   - 결과가 0개인 탐지·모델 버전은 DB에 흔적이 남지 않으므로 원본 버킷 탐지 표시
     `sessions/<세션>/derived/privacy_detect/<스트림>.json`에 남겨 다음 실행이 건너뛰게 한다
     (ADR 0030). 블러가 0개여도 사람 검수(2)는 필요하다.
   - 세션 `privacy_state`: PENDING → AUTO_BLURRED. 승인 뒤 블러가 바뀌면 APPROVED → AUTO_BLURRED.
2. 사람 검수 (CVAT, `dlp review create --stage privacy` → 검수 → `dlp review collect`)
   - 블러 검수는 원본 접근 권한자(`review.yaml reviewers.privacy`)에게만 배정한다. 검수자는 원본
     프록시를 보며 빠진 블러를 그리고 잘못된 블러를 지운다. 수집하면 수정본이 새 레코드
     (`parent_label_id`)로 남고 원 레코드는 `HUMAN_APPROVED` 등으로 검증 상태가 바뀐다.
3. 승인 `dlp privacy approve <세션>` → `runner.approve_session`
   - 영상 스트림마다 "마지막 자동 탐지 뒤에 만들어 수집까지 끝난 운영 블러 검수 작업"이 있고,
     운영 블러 라벨이 모두 사람 검수를 거쳤을 때만 APPROVED로 바꾼다.
4. 렌더 `dlp privacy render <세션>` → `runner.render_session`
   - 승인된 세션만, 운영 블러 라벨로 모자이크(또는 단색) 블러본을 만들어 **라벨링 버킷**
     (`dlp-labeling`) `sessions/<세션>/blurred/<스트림>.mp4`에 쓴다. 오디오는 넣지 않고
     PTS는 원본 그대로.
   - 옆에 렌더 기록 `<스트림>.render.json`(`render_hash` = 블러 라벨 ID 집합 + 렌더 정책의 해시,
     `blurred_sha256` = 블러본 파일 해시)을 둔다. 같은 해시면 다시 렌더하지 않는다 (멱등).
5. 소비자 확인 `runner.assert_render_current` (ADR 0024)
   - 블러본을 읽는 모든 단계가 먼저 부른다. 세션이 **지금** APPROVED이고, 렌더 기록 해시가 지금 운영
     블러 라벨·렌더 정책의 해시와 같고, 저장소의 블러본 파일이 그 기록의 파일이어야 한다.
     아니면 `RenderNotCurrentError`. 받은 뒤에는 `check_fetched`로 파일 해시를 다시 본다.
   - 재탐지·블러 검수 수집으로 블러가 바뀌면 렌더 기록을 무효(`render_hash=None`)로 덮어쓴다
     (`invalidate_render`). 무효화를 거치지 않은 경로가 있어도 해시가 달라 막힌다.

버킷 구분:
- 원본 버킷(`dlp-raw`): 원본 영상, 프록시, PTS 인덱스, 검수 우선 구간, 탐지 표시. 원본 접근
  권한자만 본다.
  저장소는 `dlp_cli.raw_access.raw_store`(감사 저장소)로만 만든다 (ADR 0020, `dlp_media.audit`).
- 라벨링 버킷(`dlp-labeling`): 블러본과 렌더 기록. 일반 라벨러가 (읽기 전용 자격 증명으로) 본다.

추가 감사: `audit` 모듈은 승인된 블러본 표본을 다른 사람이 다시 보는 잔여 누락 감사와
전수 검수 종료 판정을 계산한다 (`dlp privacy audit-sample`, `dlp ops privacy-audit`).

정책: `config/policies/privacy.yaml`(+ `config/defaults.yaml privacy`), 로더는 `policy.load_policy`.
시간: 블러 키프레임 `t_ms`는 그 스트림 영상의 PTS 시각(정수 ms)이다 (ADR 0019). 프레임 번호는
저장하지 않는다.

모듈 목록:
- `policy`: 정책 Pydantic 모델과 로더.
- `detection`: 탐지 결과(`Detection`)와 탐지기 인터페이스(`FrameDetector`).
- `geometry`: 박스(`Box`) 연산.
- `detectors`: 탐지기 구현(YuNet, QR·바코드, OWLv2, 반사면, 오라클 stub)과 공장(`build_detectors`).
- `tracker`: 탐지 → 트랙 → 프레임별 박스.
- `review`: 검수 우선 구간 계산.
- `pipeline`: 영상 한 개의 탐지 파이프라인과 탐지 모델 버전(정책 해시 포함).
- `render`: 블러본 렌더 (CVAT와 같은 프레임 번호 보간).
- `runner`: DB·저장소를 엮은 게이트 단계(탐지·승인·렌더·렌더 확인).
- `audit`: 잔여 누락 감사와 전수 검수 종료 판정.
"""
