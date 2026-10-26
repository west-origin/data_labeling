# 공개 데이터셋 라이선스 확인 (2026-10-26)

ADR 0060(공개 데이터셋 1차 학습)의 근거 자료다. 공식 페이지·라이선스 원문·논문을 받아 문구를 확인했다.
설정은 `config/external_datasets.yaml`이고 `make licenses`가 표로 낸다. 분류 기준은 ADR 0010을 데이터셋에
옮긴 것이다: 상업 사용 허용 = allowed, 비상업·연구용 조건 = review, 상업용 모델 학습 자체를 막음 = forbidden.

## 결과

| 분류 | 데이터셋 | 비고 |
|---|---|---|
| allowed | Open Images V7 | 주석 CC BY 4.0, 이미지 "CC BY 2.0으로 표기", 배포처가 이미지별 라이선스를 보증하지 않음. `Human face`·`Human hand` 박스 포함 |
| allowed | HoloAssist | CDLA-Permissive-2.0, 학습 결과물(모델) 사용 제한 없음. 169시간 1인칭 작업 영상 |
| allowed (서명 계약) | Ego4D, Ego-Exo4D | 공개 초안 기준: 탐지·분류·활동 이해 모델의 상업 제품 개발용 학습 허용, 학습한 모델의 상업 사용 허용. 데이터 자체는 제품·데이터셋에 넣을 수 없고 제3자 제공 금지, 보안 조치 필요. 기관 서명권자가 서명 |
| review | COCO 2017, LVIS v1 | 주석 CC BY 4.0, 이미지는 Flickr 개별 CC (COCO test-dev 집계로 NC 계열 약 70%) |
| review | Objects365, H2O, HO-3D | 학술 기관·학술 목적 한정. 회사가 쓸 자격 자체가 쟁점 |
| review | HOI4D, AssemblyHands, DexYCB, EPIC-KITCHENS-100, Assembly101 | CC BY-NC 4.0 (EPIC는 유상 상업 라이선스 창구 있음) |
| review | FreiHAND | 연구 목적만, 상업 사용 금지 |
| review | WIDER FACE | CC BY-NC-ND 4.0 (데이터 카드로 확인) |
| forbidden | ARCTIC | "상업용 모델 학습 금지"를 명시. MPI 상업 라이선스 창구 있음 |
| 보류 | EgoHands, 100DOH, EGTEA Gaze+, EgoObjects, CrowdPose | 원문을 열지 못했거나(사이트 장애·DNS) 라이선스 표기가 없다. 목록에 넣지 않았다 |

결론:
- 객체: Open Images V7이 가장 깨끗하다. 청소 도구(밀대·걸레·스펀지·빗자루 등)는 Open Images에 없어 LVIS(review)를
  함께 쓴다 (`config/pretrain/objects.yaml`). 출발 가중치 YOLOX COCO도 review라 결과 탐지기는 review다.
- 1인칭 손 3D 정답이 있는 학계 데이터셋 7종은 모두 비상업·학술 한정이다 (ARCTIC은 금지). 상업 가능한 1인칭
  데이터는 Ego4D·Ego-Exo4D(서명)와 HoloAssist뿐이다. 손 3D는 이들로 사전학습하고 우리 검수 데이터로 다듬는다.
- 얼굴: 상업 사용이 허용된 것으로 확인한 것은 Open Images `Human face`뿐이다.

## 법무 확인 질문

1. Ego4D·Ego-Exo4D: 서명본이 공개 초안과 같은지, 우리 프리라벨 모델이 목적 조항(객체·사건·활동 탐지·이해)에
   드는지, 그 모델이 만든 라벨을 판매할 때 "데이터가 제품·데이터셋에 보이면 안 된다"에 걸리지 않는지, 계약 종료 시
   사본 폐기가 학습한 가중치에도 미치는지.
2. 학술 기관·학술 목적 한정 약관(Objects365, H2O, HO-3D): 영리회사의 사내 연구개발이 허용 사용자에 드는지.
   들지 않으면 forbidden으로 올린다.
3. CC BY-NC·NC-ND 데이터로 학습한 가중치: 판매 전 사내 개발 단계 학습이 이미 상업 목적인지, ND에서 학습이
   변경물 작성인지 (WIDER FACE, EPIC, HOI4D, AssemblyHands, DexYCB, Assembly101).
4. 이미지별 라이선스가 섞인 COCO·LVIS: NC 이미지를 걸러 BY·BY-SA 이미지만 쓰면 allowed로 볼 수 있는지.
5. ARCTIC·EPIC-KITCHENS 상업 라이선스 협상의 비용 대비 가치.
