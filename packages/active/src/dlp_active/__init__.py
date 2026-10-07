"""액티브 러닝 (WP14): 다음에 검수·라벨링할 세션을 고른다.

세션 선택은 검수 우선순위 큐(세션 안의 검수 순서, dlp_review.ops)와 별개다.
블러는 이 점수와 무관하게 종료 조건 전까지 전수 검수한다.

진입점: `dlp active rank`(`select.rank_sessions`), `dlp active fiftyone`(`curation.build_samples` +
`curation.push_to_fiftyone`). 관련: ADR 0017(액티브 러닝), 0024(승인 취소 뒤 이전 블러본 금지).

흐름:
1. `rates`: 검수가 끝난 라벨 이력에서 클래스별 검수자 수정률(베이즈 평활)을 센다.
2. `terms`: 세션 점수 항목(플러그인). 기본은 예상 수정 수(`correction_rate`) 하나.
3. `select`: 후보 세션(기본 prelabeled, 사용 중지 제외, 정책에 따라 골든 제외)에 점수를 매겨 높은
   순으로 고른다.
4. `curation`: 고른 세션의 블러본과 라벨을 FiftyOne 데이터셋으로 만들어 사람이 살펴본다
   (선택 설치 `make install-curation`). 원본 영상은 쓰지 않는다.

이 패키지는 DB에 쓰지 않는다 (읽기만). 정책은 config/policies/active.yaml.
"""
