# ADR 0059: 객체·손 모양 마스크 (SAM 2.1 tiny, 박스·손 트랙 프롬프트)

- 상태: 채택
- 날짜: 2026-10-25
- 관련: ADR 0006(검수 도구 무손실 왕복), ADR 0010(상업 사용), ADR 0015·0019(재실행·시각),
  ADR 0028·0043(온톨로지 초안 덧붙이기·한영 이름), ADR 0049·0053(물리 사전값 등록 승인),
  ADR 0054(박스 트랙 거르기·보간 손 키프레임), ADR 0055(장면별 도구 질의), ADR 0058(착용자 몸 억제),
  실영상 파일럿 [pilot-real-2026-10-22.md](../handover/pilot-real-2026-10-22.md) 12절, 문제 레지스트리
  F-PAR-68·69·70·71.
- 계약 변경: 있음 (DB 스키마·Alembic 변경 없음, 라벨 JSON 안의 선택 필드만 늘어난다).
  - `MaskKeyframe.score` (선택, 마스크 모델 예측 IoU), `MaskTrackPayload.derived_from` (선택, 프롬프트
    트랙 라벨 ID). 둘 다 None이면 JSON에 싣지 않는다 (예전 라벨과 같은 JSON·해시).
  - 온톨로지 객체 `hand`(손, physical_type person, 물리 사전값 없음, v1 초안 덧붙이기 →
    `make db-upgrade`로 재등록). 온톨로지 해시가 바뀌어 검수 화면 프로젝트·AI 1차 검수 단위가 새로
    갈린다 (덧붙이기의 원래 성질).
  - 사전값 비교가 사전값 없는 객체를 None 항목으로 넣어, 사전값 없는 객체 덧붙이기가 승인
    (`--accept-physical-priors`)을 요구하던 것을 고쳤다 (`repository._priors_of`). 채우기·지우기·값
    변경은 그대로 승인 대상이다.
  - 정책: `prelabel.yaml models.mask_*`·`masks`. 모델: `config/models.yaml sam2_tiny_*` (review).
  - CVAT: 마스크는 모양 모드 마스크로 오간다 (`dlp_review.cvat.to_cvat_shapes`, `CVAT_KINDS`·
    `ops.policy.CVAT_KINDS`에 mask_track). COCO: `segmentation`(압축 RLE)·`mask_score`·`hand`·`part`,
    객체 hand 범주가 손 키포인트 범주를 겸한다 (`categories`, person과 같은 방식).
  - `make schemas` (label_record·episode_graph·export_intervals·ai_review_unit).

## 배경

박스만으로는 1인칭 데모처럼 손·도구·객체의 모양(채움·윤곽)을 보일 수 없고, 사내 모델 개발(분할·
접촉)에 모양 정답이 필요하다. 계약에는 `MaskTrackPayload`(COCO RLE, `dlp_media.rle`)가 이미 있었지만
만드는 단계가 없었다 (`tool_part_masks`: "SAM 2.1 마스크 미연동" TODO). 객체·도구 박스(COCO
EfficientDet·OWLv2, 거르기·장면 질의·착용자 억제를 지난 것)와 손 hand21 트랙이 이미 있으므로, 이를
프롬프트로 쓰는 분할 모델이 가장 작은 걸음이다.

## 결정

### 1. 모델과 라이선스 (ADR 0010)

조사 (2026-10-25, 1차 출처):

- 가중치·코드: facebookresearch/sam2 README "The SAM 2 model checkpoints, SAM 2 demo code (front-end
  and back-end), and SAM 2 training code are licensed under Apache 2.0"
  (https://github.com/facebookresearch/sam2). Hugging Face `facebook/sam2.1-hiera-tiny` 모델 카드
  license apache-2.0 (revision de431c4).
- 학습 데이터: SA-V는 `sav_dataset/README.md` "The videos and annotations in SA-V Dataset are released
  under CC BY 4.0". SAM 2 논문(arXiv 2408.00714)의 학습 혼합은 SA-1B(이미지), SA-V, Meta 사내 허가 영상
  (Internal), 공개 VOS(DAVIS·MOSE·YouTube-VOS)다. SA-1B는 Meta 데이터셋 페이지
  (https://ai.meta.com/datasets/segment-anything/, 검색 결과 요약으로 확인, 이 환경에서 직접 열리지
  않음)가 "research purposes"용 제한 라이선스("Limited; see full license language")로 밝힌다. MOSE는
  CC BY-NC-SA 4.0이다.
- 분류: 가중치는 Apache-2.0으로 상업 사용을 허용하지만 직접 학습 데이터에 연구용·비상업 조건(SA-1B,
  MOSE)이 있어 **review**다 (깊이 모델의 SA-1B 계보와 같은 판단). forbidden이 아니므로 연동한다.
  판매 전 법무 확인 대상(S4)이고, 데이터는 지금 사내 모델 개발용이지만 규칙은 그대로다 (ADR 0058 6절).
- 대안(이번에 쓰지 않음, 기록만): EfficientSAM(Apache-2.0, SA-1B 학습)·MobileSAM(Apache-2.0, SA-1B
  증류)도 SA-1B에 묶여 같은 review이고 영상 기억이 없다. 직접 학습 데이터가 모두 상업 허용인 범용
  분할 모델은 찾지 못했다.

변환본: 공식 PyTorch 가중치를 onnx-community가 영상 인코더와 프롬프트 인코더·마스크 디코더로 나눈
fp32 ONNX(`onnx-community/sam2.1-hiera-tiny-ONNX`, 고정 revision 814a066, 카드에 라이선스 표시 없음 →
원 가중치 라이선스를 잇는 것으로 본다)를 받는다. 네 파일(그래프 둘, 외부 가중치 둘, 약 155 MB)을
각각 sha256으로 등록하고 `make models`로 받는다. PyTorch를 코어 워크스페이스에 넣지 않고 격리 환경도
필요 없다 (기존 `onnxruntime` CPU). 가장 작은 tiny를 쓴다 (CPU 인코더 프레임당 약 1~3초).

### 2. 단계: 이미지 모드, 박스·손 프롬프트 (`dlp_prelabel.masks`, `runner._masks`)

- SAM 2 영상 모드(메모리 뱅크 전파)는 ONNX 공개본이 없고 CPU에서 느려 쓰지 않는다. 대신
  `masks.frame_stride_ms`(기본 500 ms) 간격 프레임마다 그 시각의 프롬프트로 이미지 모드를 돌린다.
  박스 트랙이 이미 시간 연속성(트랙 잇기·거르기)을 맡는다.
- 프롬프트: 바디캠의 현재 박스 트랙(개체마다 `track_rank`로 하나, 모델 트랙은 신뢰도
  `min_track_confidence` 이상)과 hand21 손 트랙. 박스는 그 시각 키프레임 또는 앞뒤 보간
  (`box_max_gap_ms` 이하). 손은 그 시각(허용 오차 안)의 관측 키프레임(보간 키프레임 제외, ADR 0054)의
  보이는 관절 외접 박스를 넓힌 것 + 손 안쪽 양성 점(손목·손가락 뿌리). 착용자 팔 arm3·전신 자세는
  프롬프트가 아니다.
- 착용자 몸: 박스 프롬프트마다 `wearer_suppress.box_frame_is_wearer`(값은 `wearer_suppression`,
  단계 목록과 무관하게 늘)를 지나야 쓴다. 객체 단계 억제를 빠져나온 착용자 팔·몸통 박스(검수 전 옛
  판, 사람이 그린 사람 박스 등)도 객체 마스크가 되지 않는다. 손이 안 보여 착용자 증거가 없는
  시각에도 사람 클래스(`masks.person_classes`) 박스의 아래 변이 화면 아래 `person_bottom_band`(3%)
  띠에 닿으면 착용자 몸통으로 보고 쓰지 않는다 (파일럿 250 s에서 착용자 셔츠가 person 마스크가 됐다.
  화면 아래에 닿은 다른 사람도 빠지는 재현 손실은 F-PAR-68). 손은 손 라벨이다 (ADR 0058과 별개).
- 검사 (정책 값): 예측 IoU < `min_score`, 넓힌 박스 밖 화소 비율 > `max_outside_ratio`(새어 나감),
  넓이 / 박스 넓이 < `min_area_ratio`면 그 키프레임을 버리고 사유를 센다. 가장 큰 연결 성분만 남기고
  (`largest_component`), 객체 마스크에서 같은 프레임 손 마스크를 뺀다(`subtract_hands`: 손이 가린 곳은
  손). 마스크 외접 박스 ↔ 프롬프트 박스 IoU를 요약에 남긴다(객체·손 따로).
- 출력: 프롬프트 트랙마다 `mask_track` 하나. entity·class는 프롬프트 트랙(손은 class `hand`, entity
  `<손>_hand`), `derived_from` = 프롬프트 트랙 라벨 ID, 키프레임마다 COCO RLE와 `score`, 라벨 신뢰도 =
  키프레임 점수 평균. 라벨 ID `<세션>-<스트림>-masks-<version_tag>-<개체>`. 마스크는 키프레임 사이를
  보간하지 않는다 (그 시각에만 유효).
- 재실행 (ADR 0015): 버전 `masks-sam2t-<네 파일 실제 해시 묶음>+mk1+p<masks 절>+w<wearer_suppression
  절>+i<입력 해시>`. 입력 해시는 고른 프롬프트 트랙과 착용자 증거 손 트랙의 라벨 ID, 보호된 개체다.
  `stage_version`으로 마지막 판이 같으면 건너뛰고(되돌아오면 `+r<n>`), 바뀌면 검수 전 이전 마스크만
  지운다. 같은 개체에 검수된·사람 마스크가 있으면 그 개체는 새로 만들지 않는다 (`masks_blocked`).
  입력 트랙이 없으면 원본을 받지 않는다.
- `derived_from`과 검수: 출처 기록일 뿐 검수를 물려받지 않는다. 마스크는 모양이 박스와 다른 정보라
  따로 검수한다 (3D 궤적의 따르기 ADR 0048과 다르다). 박스가 고쳐지면 입력 해시가 바뀌어 검수 전
  마스크만 새로 만든다.
- CPU stub `BoxEllipseSegmenter`(박스 내접 타원)는 시험용이다. CLI는 가중치가 없으면 `[모델 없음]`으로
  실행을 거부하고(다른 모델과 같다) `--skip masks`로만 뺀다. stub으로 조용히 대신하지 않는다.
- 프라이버시: 블러는 박스 그대로이고 마스크와 무관하다. 마스크 단계는 원본 영상에서 돈다 (다른 프리라벨
  단계와 같은 감사 저장소 경로).

### 3. 검수 도구·내보내기

- CVAT: 트랙 모드는 마스크를 보간하지 않으므로 모양 모드 마스크로 보낸다 (키프레임마다 모양 하나, 모양
  속성에 원래 라벨 ID·메타). 수집은 원래 라벨 ID로 다시 한 트랙으로 묶고, ID 없는 새 마스크는 모양
  하나가 새 트랙이다. CVAT 마스크 점(외접 박스 안 행 우선 연속 길이 + [왼, 위, 오른, 아래])과 COCO
  RLE는 같은 화소라 무손실이다. 점수는 보내지 않는다 (`quantize`). 원본 해상도 작업에서만 오가고
  블러 검수에서는 받지 않는다. 새로 그린 손 마스크는 어느 손인지 모른다 (entity `hand_h<n>`).
- COCO: 마스크 키프레임마다 `segmentation`(압축 RLE 그대로)·마스크에서 계산한 bbox·area·`mask_score`.
  손 마스크는 hand 범주(객체 hand가 손 키포인트 범주를 겸함)에 `hand`(left/right).
- LeRobot·구간 JSON에는 싣지 않는다 (프레임 특징으로 넣기에 크다, F-PAR-70). 관계·접촉 단계는 아직
  박스를 쓴다 (F-PAR-70).
- 미리보기(`scripts/preview_session.py`): 반투명 채움 + 윤곽선, 손 초록·주황, 객체는 클래스 색.

## 결과

- 정답 모양 합성 영상(`dlp_fixtures.shapes`)에서 실제 SAM 2.1 tiny 마스크 IoU 컵 1.00·양동이 1.00·
  손 0.98, 착용자 팔(person 박스)은 마스크 없음. stub 타원은 컵 IoU 0.9 이상.
- 파일럿 수치는 파일럿 보고서 12절.
- 한계: 영상 전파가 없어 키프레임 사이 마스크가 없고, 박스가 없는(놓친) 객체는 마스크도 없다. 부분
  마스크(작용부·파지부)는 여전히 없다 (`tool_part_masks` TODO).

## 대안

- SAM 2 영상 모드(PyTorch 격리 환경): 정확한 전파지만 CPU에서 프레임당 수 초 + 기억 뱅크 메모리,
  격리 환경 디스크(PyTorch 약 1 GB)가 지금 장비에 크다. GPU 서버가 생기면 같은 `Segmenter` 자리에
  다시 검토한다 (F-PAR-69).
- 박스 채움만: 모양 정보가 없다.
