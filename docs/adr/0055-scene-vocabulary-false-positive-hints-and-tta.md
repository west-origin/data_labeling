# ADR 0055: 장면별 도구 질의, 오탐 의심 힌트, 테스트 시점 증강(TTA)

- 상태: 채택
- 날짜: 2026-10-23
- 관련: ADR 0005·0023(블러 검수 우선 구간), ADR 0008(프리라벨 어댑터), ADR 0010(상업 사용), ADR 0040(AI 1차
  검수는 라벨·검수 상태가 아님), ADR 0054(시간 일관성), 실영상 파일럿 [pilot-real-2026-10-22.md](../handover/pilot-real-2026-10-22.md)
  10.3절, 문제 레지스트리 F-PAR-60·F-PMS-31.
- 계약 변경: 없음 (라벨 형식 그대로). 정책·저장 위치만 더한다.
  - `prelabel.yaml scene`(장면 분류기·장면 목록), `open_vocab_objects.scene_queries`(장면 → 질의).
    도구 단계 버전 `owlv2-…+p<open_vocab_objects·scene 해시>+sc<장면>`.
  - `privacy.yaml hints`(탐지 해시에 들어가지 않음), `review.yaml object_hints`, `cvat.hint_issue_limit`.
  - `ReviewReason`에 `suspected_false_positive`. 원본 버킷 `derived/privacy_review/<스트림>.hints.json`,
    라벨링 버킷 `sessions/<세션>/hints/<스트림>.objects.json`.
  - 공용 기하 변환 `dlp_media.geometry`(TTA·학습 증강이 같이 쓴다).
  - 명령 `dlp review hints <세션> [--tta]`, `dlp prelabel run --scene <장면>`.

## 배경

첫 실영상(부엌 설거지·택배 개봉)에서 OWLv2가 욕조·변기 솔·기저귀를 분당 수십~백여 트랙 냈고(장면과 무관한
질의), 얼굴 블러 키프레임 45,673개 중 RTMPose 머리 관절 근처는 4.7%뿐이었다(싱크·손·바닥의 얼굴 오탐).
COCO 사람 박스는 착용자 자신의 팔·몸통을 덮었다. 정답이 없어 문턱을 정확도로 맞출 수 없다.

## 결정

1. **장면별 도구 질의.** 도구 단계 앞에서 세션 장면(kitchen·bathroom·bedroom·living·unknown)을 정하고
   `scene_queries`의 질의만 묻는다. 모르면(unknown) 전체 질의다 (재현을 잃지 않게). 장면 출처 순서:
   사람이 준 `--scene`(manual) → 정책 분류기. 분류기는 `FixedScene`(CPU·CI stub, 기본 unknown)과
   `VlmScene`(사내 OpenAI 호환 VLM, `DLP_VLM_URL`. 외부 API 금지)이다. VLM에는 그 시각 운영 블러 트랙
   박스를 모자이크한 프레임만 보내고, 블러 트랙이 없으면(프라이버시 탐지 전) 보내지 않는다. 장면이
   버전에 들어가므로 장면이 바뀌면 도구 단계가 다시 돌고 검수 전 이전 결과만 지운다. 로컬 장면 분류기는
   상업 사용 가능한 가중치를 찾지 못해 `TODO(real-model)`이다 (Places365 계열은 비상업).
2. **오탐 의심 힌트는 표시만 한다.** 라벨·블러·렌더·승인·내보내기를 바꾸지 않는다. 블러 검수 화면에는
   누락 위험 구간 이슈(`privacy_issue_limit`) 다음에 따로 센 힌트 이슈(`hint_issue_limit`)로, 라벨링 화면에는
   객체 힌트 이슈로 단다. 규칙: 얼굴 박스 안에 머리 관절이 없음(`no_head`), 얼굴이 손 위(`on_hand`),
   짧은 깜박임(`short_flicker`), 사람 박스가 착용자 손목을 담고 머리가 없음(`wearer_body`), 변환 화면에서
   다시 찾지 못함(`tta_unstable`). 프라이버시는 재현이 우선이라 힌트가 블러 검수 순서를 바꾸지 않고,
   검수자가 오탐을 빨리 걷어내도록 돕기만 한다.
3. **TTA.** 좌우 뒤집기·배율(정책 `tta.transforms`) 화면에 탐지기를 다시 돌려, 역변환한 탐지가 원래 박스와
   IoU `tta.iou` 이상인 변환의 비율을 지지도로 쓴다. 얼굴은 YuNet(빠름), 도구는 OWLv2(CPU 프레임당 수 초)라
   신뢰도 낮은 트랙부터 `max_tracks`개만 본다. 기하 변환은 `dlp_media.geometry` 한 곳에 두어 학습 증강
   (그룹 5)이 같은 행렬·라벨 변환(박스·점·좌우 손·COCO 좌우 관절)을 쓴다.
4. **AI 판정과 같은 위치.** 힌트는 ADR 0040의 AI 판정처럼 참고 정보다. 사람 검수·표본을 줄이지 않는다.

## 결과

- 장면 질의로 부엌 영상에서 장면 밖 도구 질의(욕조·변기 솔·기저귀·거울)가 빠진다. 장면을 틀리게 정하면
  그 장면에 없는 실제 도구를 놓친다 (unknown이 안전한 기본값).
- 힌트 규칙은 RTMPose·MediaPipe 결과에 기대므로 그 모델이 놓치면 힌트도 틀린다. 1인칭에서 다른 사람의
  얼굴이 머리 관절 없이 보이면 no_head가 잘못 붙을 수 있다 (표시일 뿐 블러는 그대로).
- CVAT 이슈 수 상한 때문에 힌트가 많으면 일부만 화면에 보인다 (비율 큰 것부터).

## 개정 (2026-10-28)

- OWLv2 도구 단계 버전은 `owlv2-…+p<open_vocab_objects 해시>+sc<장면>`이다 (F-PAR-76). 질의 선택은 장면 이름과
  `open_vocab_objects`(queries·scene_queries)만으로 정해지고, 장면 이름은 `+sc<장면>`이 이미 담는다. scene 절
  (분류기·시간 제한·표본 프레임·블러 칸 등)은 분류 결과를 거쳐서만 영향을 주므로 해시에서 뺐다. VLM 시간 제한만
  바꿔도 검수 전 도구 박스를 지우고 다시 만들던 문제를 막는다. 이 판으로 올린 뒤 첫 실행은 버전 문자열이 바뀌어
  한 번 다시 돈다.
