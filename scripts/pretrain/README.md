# 공개 데이터셋 객체 탐지 1차 학습 (ADR 0060 1단계)

Open Images V7 + LVIS v1에서 가사·청소·돌봄 객체(`config/pretrain/objects.yaml`의 21개 클래스)만 골라
YOLOX-s(Apache-2.0)를 미세조정하고 ONNX로 내보낸다. Kaggle·Colab 무료 GPU에서 돌리는 것을 전제로 한다.

- **우리 데이터는 쓰지 않는다.** 원본·블러본·라벨·골든셋·작업자 정보를 노트북이나 외부 저장소에 올리지 않는다.
  이 디렉터리의 코드는 우리 DB·버킷에 접근하지 않는다.
- 노트북·올린 파일은 **비공개**로 둔다. 비밀값은 필요 없고 넣지 않는다.
- 공개 데이터 검증 점수는 참고값이다. 정책이 이 가중치를 쓰려면 우리 골든셋 평가(`dlp eval golden`)와
  `evaluation.yaml` 게이트를 통과해야 한다 (ADR 0060 5절). 2단계(우리 데이터로 다듬기)는 사내 GPU에서만 한다.

## 파일

| 파일 | 하는 일 |
| --- | --- |
| `build_objects.py` | 공식 주석을 받아 대상 클래스를 고르고 이미지를 받아 COCO 형식 train/val + `data_manifest.json`을 쓴다 |
| `train_yolox.py` | YOLOX 고정 커밋 clone, Exp 생성, 학습(이어 하기), ONNX 내보내기, `model_manifest.json` |
| `yolox_ext.py` | Exp가 import하는 YOLOX 확장 (확인 안 된 클래스의 배경 손실 무시, 아래 "확인된 클래스") |
| `requirements.txt` | 플랫폼 기본 torch 위에 더 설치할 패키지 (정확한 버전) |
| `objects_pretrain.ipynb` | Kaggle·Colab 겸용 노트북 |

## 실행 순서

먼저 저장소 루트에서 올릴 묶음을 만든다 (저장소가 공개가 아니어도 된다. 이 두 디렉터리만 올린다):

```bash
zip -r pretrain_bundle.zip scripts/pretrain config/pretrain -x '*/__pycache__/*'
```

### Kaggle

1. 새 노트북 → *File → Import Notebook*으로 `objects_pretrain.ipynb`를 올린다. 공개 범위 Private 확인.
2. 오른쪽 설정: Accelerator = GPU (T4 x1 등), Internet = On. 휴대폰 인증이 필요할 수 있다.
3. *Add Input → Upload*로 `pretrain_bundle.zip`을 **비공개 데이터셋**으로 올린다 (`/kaggle/input/...`).
4. 셀을 위에서부터 실행한다: 플랫폼 감지 → 묶음 풀기 → `pip install -r requirements.txt` →
   `build_objects.py --dry-run`(개수 확인) → `build_objects.py`(이미지) → `train_yolox.py --resume` → 결과 정리.
5. 오래 걸리는 학습은 *Save Version → Save & Run All (Commit)*으로 돌린다 (세션이 닫혀도 계속된다).
   체크포인트는 `/kaggle/working/ckpt`에 있어 버전 출력으로 남는다. 다음 세션에서 이어 하려면 그 버전 출력을
   입력으로 붙이고 7번 셀 `RESUME_FROM`에 그 `ckpt` 경로를 적는다. 이미지(수 GB)는 출력에 남지 않는
   `/kaggle/tmp` 또는 `/tmp`에 두므로 새 세션에서는 6번 셀(이미지 받기)을 다시 돌린다 (주석 캐시는 없어지므로
   주석도 다시 받는다). 시간 제한에 걸린 커밋의 출력이 남는지는 **확인 필요**다.

### Colab

1. `objects_pretrain.ipynb`를 Colab에서 연다. *런타임 → 런타임 유형 변경 → GPU*.
2. 2번 셀이 Google Drive를 연결한다. 체크포인트는 `MyDrive/dlp_pretrain/ckpt`에 둔다 (끊겨도 남는다).
3. 3번 셀: `MyDrive/dlp_pretrain/pretrain_bundle.zip`이 있으면 그것을, 없으면 업로드 창으로 받는다.
4. 셀을 위에서부터 실행한다. 데이터는 런타임 로컬 디스크 `/content/objects`(Drive는 느리다).
5. 세션이 끊기면 다시 연결해 1~6번(이미지는 다시 받음, 이미 받은 것은 건너뜀 — 단 런타임이 바뀌면 로컬
   디스크가 비므로 처음부터)과 7번(`--resume`이 Drive의 `latest_ckpt.pth`에서 이어 감)을 실행한다.

### 명령줄로

```bash
pip install -r scripts/pretrain/requirements.txt
python scripts/pretrain/build_objects.py --config config/pretrain/objects.yaml --out DATA --dry-run
python scripts/pretrain/build_objects.py --config config/pretrain/objects.yaml --out DATA
python scripts/pretrain/train_yolox.py --config config/pretrain/objects.yaml --data DATA \
    --ckpt-dir CKPT --work WORK --resume
# GPU 없이 경로만 확인 (학습 아님): --cpu-smoke --batch-size 4 --workers 0
```

`build_objects.py` 인자: `--config`, `--out`, `--max-images-per-class`(설정 덮어쓰기), `--dry-run`(주석만 받고
개수만 보고, 이미지 안 받음), `--workers`(동시 내려받기, 기본 16), `--max-side`(저장 이미지 긴 변 상한, 기본
1024, 0이면 원본), `--verify-rule all|any`(아래), 시험용 `--oi-max-rows N`·`--lvis-splits val`.
다시 실행하면 `_cache/`의 걸러 낸 주석과 받은 이미지 기록을 써서 이어 간다.

`train_yolox.py` 인자: `--resume`(latest_ckpt.pth가 있으면 이어서, 없으면 COCO 가중치에서), `--epochs`·
`--batch-size`·`--fp16/--no-fp16`(설정 덮어쓰기), `--warmup-epochs 1`(YOLOX 공식 사용자 데이터 예시
`yolox_voc_s.py`와 같음), `--no-aug-epochs 5`, `--eval-interval 5`, `--ignore-score 0.3`, `--no-federated`,
`--skip-train`(내보내기만), `--cpu-smoke`.

## 이미지를 받는 방법 (공식 공개 버킷 https, FiftyOne 아님)

- Open Images: `https://open-images-dataset.s3.amazonaws.com/{train|validation}/{ImageID}.jpg`
  (공식 안내의 `s3://open-images-dataset` 공개 버킷). 표준 라이브러리 `urllib`만 쓴다.
- LVIS: 각 이미지의 `coco_url`(`http://images.cocodataset.org/...`). 실패하면 같은 공식 버킷의 https 경로
  (`https://s3.amazonaws.com/images.cocodataset.org/...`)로 한 번 더 시도한다 (이 개발 컨테이너는 http가 막혀
  있어 이 경로로 받았다).
- FiftyOne zoo(`open-images-v7`)를 쓰지 않은 이유: FiftyOne은 약 1 GB 설치 + MongoDB 프로세스가 필요하고,
  이미지 단위 음성 라벨을 우리 규칙(아래)대로 쓰려면 어차피 공식 CSV를 직접 읽어야 한다. LVIS는 zoo에 없다.
  직접 받으면 의존성이 적고, 받은 주석 파일의 sha256을 manifest에 남길 수 있다.

큰 CSV(OI train 박스 2.26 GB, 이미지 단위 라벨 2.74 GB, 이미지 목록 0.64 GB)는 디스크에 저장하지 않고
스트리밍으로 읽으면서 sha256을 계산한다 (끊기면 Range로 이어 받음). LVIS zip(train 350 MB)은 받아서 ijson으로
스트리밍 파싱하고 바로 지운다. 회전 정보(Rotation)가 0이 아닌 OI 이미지와 EXIF 회전 태그가 있는 이미지는
박스 좌표계를 확신할 수 없어 뺀다.

## 확인된 클래스와 손실 처리

두 데이터셋 모두 이미지마다 모든 클래스를 다 달지 않았다 (LVIS는 연합 주석, OI는 확인한 라벨만).
`build_objects.py`는 이미지마다 `images[].verified_class_ids`(그 이미지에서 빠짐없이 확인된 우리 클래스 ID)를 쓴다.

- 원래 클래스 하나가 확인됨: 박스가 있음(LVIS), 또는 박스가 있음·사람이 없다고 확인함(OI Confidence 0 라벨),
  LVIS `neg_category_ids`. 확인 안 됨: LVIS `not_exhaustive_category_ids`, OI 묶음 박스(IsGroupOf, 박스 자체도
  버림), `min_box_px` 미만으로 버린 박스가 있는 클래스, 박스 없는 OI 양성 라벨.
- 우리 클래스(여러 원래 클래스를 합침)가 확인됨: `--verify-rule all`(기본) = 대응한 원래 클래스 전부가
  확인됨. `any` = 하나라도 확인되고 확인 안 됨 표시가 없음 (나머지 원래 클래스 물체가 있을 위험을 받아들임).
  그 데이터셋에 대응이 없는 클래스는 늘 확인 안 됨 (예: LVIS 이미지의 door, OI 이미지의 mop).

YOLOX에는 클래스별 손실 무시가 없다. 클래스 손실은 원래 박스가 배정된 앵커에만 걸리고, 문제는 클래스와
무관한 objectness가 "배정 안 된 앵커 = 배경"으로 학습되는 것이다. `yolox_ext.py`가 하는 근사:

- 모자이크·mixup으로 섞인 이미지들의 "확인 안 된 클래스" 합집합을 라벨 텐서 뒤 열로 붙여 넘긴다.
- 배경 앵커 가운데 모델이 확인 안 된 클래스를 `ignore_score`(objectness * 클래스 확률, 기본 0.3) 넘게 예측한
  앵커는 objectness 배경 손실에서 뺀다. 박스 배정(SimOTA)·박스·클래스 손실은 그대로다.
- 이 개발 컨테이너에서 같은 배치로 확인: 무시를 켜면 objectness 손실만 줄고 박스·클래스 손실은 같다
  (마스크 열 정렬, 모자이크 합집합도 확인).

**한계 (지어내지 않음)**: 위치를 모르는 누락 물체를 모델 예측으로 짐작하는 휴리스틱이다. 학습 초기처럼 예측이
약하면 거의 작동하지 않고, 모델이 못 맞히는 누락 물체는 여전히 배경으로 학습된다. `ignore_score` 0.3은 이
도구가 정한 값이고 조정 근거(실험)는 아직 없다. 공개 검증 AP도 같은 누락 때문에 낮게(또는 틀리게) 나온다
(LVIS 공식 평가처럼 확인된 클래스만 세는 평가가 아니라 YOLOX 기본 COCO 평가다).
`--verify-rule all`에서는 원래 클래스를 여럿 합친 cup·table·towel·bed·sink가 거의 늘 확인 안 됨이 된다
(아래 표). 그만큼 그 클래스들의 배경 학습이 약해진다.

## ONNX와 manifest

- 내보내기: YOLOX 0.3.0 `tools/export_onnx.py` 그대로, opset 11, `--no-onnxsim`, **`decode_in_inference`
  끔**(공식 기본값). 프리라벨 어댑터 정책 `decoded_output: false`와 맞춘다.
- 입력 `images`: `1x3x640x640` float32, **BGR 0~255**, letterbox(비율 유지 축소, **왼쪽 위 배치**, 나머지
  **114**), 정규화 없음 (YOLOX 0.3.0 `ValTransform(legacy=False)`).
- 출력 `output`: `1x8400x(5+21)`. 앞 4개는 격자 상대값 cx·cy·w·h라 strides 8·16·32 격자로 디코드한다
  (`(xy + grid) * stride`, `exp(wh) * stride`). 이어서 objectness, 클래스 21개 (둘 다 sigmoid 적용됨).
- 클래스 순서 = `objects.yaml classes` 키 순서 (0=mop … 20=bed), manifest `onnx.classes`.
- 최근 torch(2.9 등)에는 0.3.0이 부르는 `torch.onnx._export`가 없다. `train_yolox.py`가 그 이름을
  `torch.onnx.export(..., dynamo=False)`(TorchScript 기반, 0.3.0 당시와 같은 경로)로 이어 준다.
- `model_manifest.json`: 데이터 manifest sha256, 학습 설정 전체, YOLOX 커밋, 출발 가중치 sha256, Exp 파일
  sha256, 내보낸 체크포인트와 ONNX sha256, ONNX 검사(PyTorch와 ONNX Runtime 출력 최대 차이),
  `pip freeze` 파일과 sha256, torch·CUDA·GPU, 코드 커밋·스크립트 sha256.
- `data_manifest.json`: 데이터셋 이름·판, 받은 주석 파일 9개의 URL·sha256·바이트, 클래스별 이미지·박스·확인
  이미지 수(분할별), 설정 파일 sha256과 실제 적용 설정 sha256, 코드 커밋, 내려받기 실패·버린 박스 수.

## 예상 디스크·시간

이 컨테이너(CPU 4코어, 내려받기 약 25 MB/s)에서 잰 값과 그로부터의 추정이다. Kaggle·Colab 실측은 아직 없다.

| 단계 | 근거 | 값 |
| --- | --- | --- |
| 주석 (dry-run) | 실측: OI train 박스 95초, 이미지 목록 32초, 이미지 라벨 128초, LVIS zip 받기 16초 + 파싱 약 4분 | 약 10분, 디스크 최대 약 0.4 GB(LVIS zip, 곧 지움) + 캐시 52 MB |
| 고른 이미지 수 | 실측 (지금 설정, `--verify-rule all`) | OI 17,492장 + LVIS 20,530장 = 38,022장 |
| 이미지 용량 | 표본 68장 평균: OI 236 KB(긴 변 1024), LVIS 135 KB(COCO 원본 640) | 약 7 GB (표본이 작아 오차 큼) |
| 이미지 받기 | 실측 68장 16 동시: 15장/초 | 약 40분 (플랫폼 대역폭에 따라 다름, 모름) |
| 체크포인트 | 실측 latest_ckpt.pth 72 MB | latest·best·last_epoch 약 0.2 GB (Drive) |
| ONNX | 실측 | 36 MB |
| 학습 시간 | **모른다** (GPU에서 재지 않음) | 첫 epoch 로그의 ETA를 보고 `--epochs`를 정한다 |

## 클래스별 표본 (dry-run 실측, 후보/고름/박스)

| 클래스 | Open Images V7 | LVIS v1 | 비고 |
| --- | --- | --- | --- |
| mop | – | 30 / 30 / 33 | 희귀 |
| rag | – | 17 / 17 / 31 | 희귀 (dishrag) |
| sponge | – | 92 / 91 / 126 | |
| brush | – | 139 / 134 / 150 | |
| broom | – | 111 / 111 / 158 | |
| dustpan | – | 10 / 10 / 10 | LVIS 희귀(r), 성능 기대 어려움 |
| bucket | – | 842 / 838 / 1,537 | |
| trash_bin | 1,037 / 1,037 / 1,828 | 2,236 / 2,224 / 2,996 | |
| pillow | 1,276 / 1,276 / 3,551 | 2,318 / 2,307 / 7,280 | |
| towel | 143 / 143 / 298 | 1,497 / 1,480 / 4,577 | |
| cup | 4,431 / 3,000 / 5,317 | 2,865 / 2,854 / 7,359 | 확인됨 거의 없음 |
| door | 13,510 / 3,000 / 4,275 | – | |
| drawer | 1,260 / 1,260 / 3,437 | 2,303 / 2,285 / 9,153 | |
| diaper | 114 / 114 / 151 | 80 / 79 / 99 | 적음 |
| glove | 761 / 761 / 1,249 | 2,212 / 2,211 / 5,783 | |
| sink | 1,479 / 1,479 / 1,689 | 2,553 / 2,512 / 3,150 | |
| toilet | 869 / 869 / 1,110 | 2,280 / 2,240 / 2,867 | |
| mirror | 1,161 / 1,161 / 1,588 | 2,168 / 2,147 / 3,309 | |
| table | 49,665 / 3,000 / 4,967 | 3,254 / 3,000 / 4,275 | 확인됨 거의 없음 |
| bathtub | 545 / 545 / 558 | 980 / 965 / 1,022 | |
| bed | 3,441 / 3,000 / 3,581 | 2,149 / 2,126 / 2,586 | 확인됨 적음 |

OI 박스 수는 `min_box_px` 적용 전이다 (OI는 이미지 크기를 받아야 알 수 있어 이미지를 받은 뒤 적용한다).
고른 수가 상한(3,000)을 넘지 않는 것은 희귀 클래스부터 고르고 함께 든 클래스도 세기 때문이다.
두 데이터셋에 같은 Flickr 사진이 있으면 앞 데이터셋(OI) 것만 쓰고, 받은 뒤 바이트가 같은 이미지도 하나만 둔다.

## 결과물을 저장소로 가져오기

- 가져올 것: `ckpt/export/` 의 `objects_pretrain_yolox_s.onnx`, `model_manifest.json`, `data_manifest.json`,
  `pip_freeze.txt`. **가중치(.onnx·.pth)는 저장소에 넣지 않는다.** 사내 모델 저장 위치(`data/models/` 아래,
  `make models`가 받는 곳)에 두고 manifest의 sha256으로 확인한다.
- `config/models.yaml` 등록은 이 도구 밖의 별도 작업이다 (`training_datasets: [open_images_v7, lvis_v1]`,
  출발 가중치 분류, `commercial`, sha256). 그 뒤 `dlp eval golden`으로 지금 모델과 비교한다.

## 라이선스 (확인 필요 항목 포함)

| 대상 | 라이선스 | 비고 |
| --- | --- | --- |
| Open Images 주석 | CC BY 4.0 (Google LLC) | 공식 안내 기준 |
| Open Images 이미지 | 이미지마다 CC BY 2.0으로 표기 (이미지 목록 CSV `License`·`Author`) | 검증 분할 41,620장 모두 CC BY 2.0임을 이 CSV에서 확인. Google은 이미지 라이선스를 보증하지 않는다고 적고 있다 (원문 재확인 필요) |
| LVIS 주석 | CC BY 4.0 | 공식 안내 기준 (원문 재확인 필요) |
| COCO 이미지 | Flickr 사진마다 다른 CC 라이선스 | 받은 이미지에 `by-nc-sa/2.0`이 실제로 있었다 → **비상업 조건 이미지 포함**. 가중치 분류에 반영해야 한다 (ADR 0010, `models.yaml yolox_m_coco`와 같은 근거로 review 이상) |
| YOLOX 코드·COCO 가중치 | Apache-2.0 (가중치 학습 데이터는 COCO) | 출발 가중치 분류는 review |

이미지별 라이선스·작성자·원본 주소는 COCO json `images[]`의 `license`·`author`·`source_url`에 남긴다
(OI는 작성자 표기, COCO는 라이선스 URL). 최종 분류는 `config/external_datasets.yaml`과 ADR 0060 라이선스 표에서 정한다.

## 알려진 한계

- **연합 주석**: 위 "확인된 클래스" 절. 손실 무시는 휴리스틱이고, 합친 클래스(cup·table·towel·bed·sink)는
  `all` 규칙에서 거의 확인되지 않는다. 공개 검증 AP는 참고만 한다.
- **1인칭 도메인 차이**: 공개 데이터는 대부분 3인칭 사진이다. 가슴 장착 바디캠의 가까운 거리·손 가림·움직임
  흐림·광각 왜곡이 없다. 우리 골든셋 평가 전에는 성능을 말할 수 없다.
- **희귀 범주**: dustpan 10장, rag 17장, mop 30장, diaper 약 190장. 이 클래스들은 쓸 만한 탐지를 기대하기
  어렵다 (OWLv2 경로를 남겨 둔다).
- **주석 의미 차이**: OI `Mug`에 맥주 피처 같은 것이 섞여 있다 (표본 확인). OI 클래스 계층은 쓰지 않았다.
- **재현성**: 데이터 고르기·분할은 seed로 결정적이다. 학습은 YOLOX 기본대로 seed를 두지 않아(cudnn 결정성
  끄기를 피함) 같은 결과가 나오지 않는다. 내려받기 실패(404 등)가 있으면 이미지 집합이 달라질 수 있고,
  실패 수는 manifest에 남는다.
- **환경 고정**: ADR 0060은 `scripts/pretrain-env/uv.lock`을 말하지만 Kaggle·Colab의 torch를 그대로 쓰기 위해
  `requirements.txt`(추가 패키지만 고정) + 실행마다 `pip freeze` 해시로 대신했다.
- **확인하지 못한 것**: GPU 학습(YOLOX Trainer는 CUDA 필요), Kaggle·Colab에서의 실제 실행, 학습 중 평가
  (COCOEvaluator도 CUDA 텐서를 쓴다), 전체 이미지 내려받기. 이 컨테이너에서는 주석 전체 dry-run, 작은 실제
  데이터(68장) 만들기, CPU 1회 손실·역전파, ONNX 내보내기와 ONNX Runtime 비교까지 확인했다.
