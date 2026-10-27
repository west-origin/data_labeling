# ADR 0062: 재학습 루프의 실제 학습기 (YOLOX 물건 탐지)와 AI 검수 판정 표시 보정

- 상태: 제안 (구현됨, GPU 학습 서버 실측 전)
- 날짜: 2026-10-27
- 관련: ADR 0010(상업 사용), ADR 0016·0025(재학습 루프·레지스트리), ADR 0020(원본 접근), ADR 0040(AI 1차 검수,
  I9), ADR 0057(학습 증강), ADR 0058(사내 모델 개발 용도), ADR 0060(공개 데이터 1차 학습), F-PAR-60(도구 오탐).
- 계약 변경: 없음 (라벨·DB·Alembic·내보내기·`schemas/` 그대로). 정책 `training.yaml yolox` 절과 `objects.replaces`에
  `closed_tools` 추가, `review.yaml ai_review.calibration` 절, `models.yaml yolox_s_coco_pth` 항목, 격리 환경
  `scripts/yolox-train-env/`(pyproject + uv.lock + finetune.py).

## 배경

"사람이 고친 라벨로 AI가 다시 학습해 점점 정확해지는" 루프(`dlp train run`, ADR 0016)에는 oracle-stub 학습기만
있었다 (`dlp_train.trainers`의 TODO(real-model)). 루프의 나머지(데이터셋 버전 학습·검증 분할에서만 예제 추출, 골든셋
메모리 평가, 게이트, 배포, 배포 모델 붙이기)는 이미 있다. 빠진 것은 (1) 픽셀로 실제 가중치를 학습하는 학습기와
(2) 그 산출물을 프리라벨 어댑터에 끼우는 로더다. 또 AI 1차 검수 판정(ADR 0040)은 사람 판정 기록
(`ai_review_outcomes`)이 쌓여도 그것으로 나아지는 길이 없었다. 이 컨테이너와 사내에는 아직 GPU 학습 서버가 없다.

## 결정 A: YOLOX 고정 클래스 물건 탐지기 학습기 (`trainer: yolox`, 과제 objects)

1. **과제·클래스.** 과제는 기존 `objects`(box_track)다. 출력 클래스 순서와 입력 크기는 새로 정하지 않고
   `prelabel.yaml closed_vocab_tools`(classes·input_size)를 그대로 쓴다. 로더가 끼우는 어댑터
   (`dlp_prelabel.adapters.yolox_tools.YoloxTools`)가 그 정책으로 출력 열 이름을 붙이기 때문이다. 목록 밖 클래스의
   박스는 학습 주석에서 빼고 개수만 manifest에 남긴다. 배포되면 `objects`(COCO)·`tools`(OWLv2)·`closed_tools`(공개
   데이터 YOLOX)를 대신한다 (`training.yaml objects.replaces`). 21개 클래스 밖 물건을 잃는지는 게이트가 본다: 첫 배포는
   세 기본 어댑터의 골든 예측 버전 모두(`--baseline-version`)와 비교한다. 기본 학습기는 CI가 도는 oracle-stub 그대로
   두고 `--trainer yolox`로 고른다.
2. **학습 데이터.** `dlp_train.extract`의 예제(데이터셋 버전 학습·검증 분할, 운영 라벨, 사람이 만들었거나 승인·수정·
   표본 검증한 라벨)를 `dlp_train.yolox_data.build_coco`가 COCO로 내보낸다. 이미지는 **지금 승인된 블러본**에서 box_track
   키프레임 시각(스트림 PTS)의 프레임만이다. 블러본 PTS 인덱스에서 그 정수 ms 프레임을 찾고, 없으면 건너뛰고 센다
   (프레임 번호 x 간격 계산 없음). 그 시각의 다른 트랙 박스는 키프레임 선형 보간(outside 구간 제외)으로 같이 단다.
   지운 예제(오탐)의 시각도 박스 없이 이미지로 넣어 배경으로 배운다. 블러본 공급자는 학습기가 아니라 CLI가 넘긴다
   (`dlp_cli.train_cmds.blurred_videos`: 원본 버킷 저장소 거부 `check_blurred_store`, 스트림마다 렌더 게이트
   `assert_render_current`·받은 파일 해시 `check_fetched`). 원본 버킷은 학습 이미지에 쓰지 않는다 (골든 평가는 지금처럼
   감사 원본 저장소).
3. **증강 (ADR 0057).** objects가 `augmentation.tasks`에 있으면 학습 분할 이미지마다 `augment.augment_examples`로
   `copies`개 증강본을 COCO 학습 데이터에 더한다 (박스는 같은 변환, 씨앗은 정렬 순서 표본 번호). YOLOX 내부의 무작위
   변환(모자이크·mixup·HSV·뒤집기·회전)은 끈다: 증강 정책이 하나여야 해시(`params.augmentation`)가 학습을 설명한다.
   학습기는 루프가 넣은 해시가 지금 정책 해시와 다르면 거절한다.
4. **격리 환경과 subprocess.** 본 가상 환경에 PyTorch를 넣지 않는다. `scripts/yolox-train-env/`(uv 프로젝트, 작업공간
   멤버 아님)의 `uv.lock`에 전이 의존성까지 고정하고 `uv run --locked --isolated --extra <cpu|cu128>`으로
   `finetune.py`를 돌린다 (LeRobot·리타기팅 환경과 같은 관례). PyTorch 판은 extra로 고른다: `nvidia-smi -L`이 GPU를
   보이면 `cu128`, 아니면 `cpu` (두 extra는 `conflicts`). YOLOX는 설치하지 않고 태그 0.3.0 커밋
   `419778480ab6ec0590e5d3831b3afb3b46ab2aa3`을 `training.yaml yolox.source_dir`(Git 제외 `data/cache/yolox/YOLOX`)에
   clone해 경로로 쓴다. setup.py가 요구하는 onnx-simplifier 고정판은 빌드가 무겁고 쓰지 않기 때문이다.
5. **공개 데이터 1차 학습 도구 재사용 (옮기지 않음).** `scripts/pretrain/train_yolox.py`의 `ensure_yolox`(커밋 고정
   clone)·`export_onnx`(torch.onnx 구형 경로 이음, onnxsim 없음, decode_in_inference false)·`check_onnx`(PyTorch 대
   ONNX Runtime 차이)·`environment`(패키지 목록 해시)를 `finetune.py`가 import해 그대로 쓴다. 공용 모듈로 옮기지 않은
   이유: (a) 1차 학습 묶음은 `scripts/pretrain`·`config/pretrain`만 zip으로 Kaggle·Colab에 올리는 구조라 옮기면 노트북
   절차가 바뀐다, (b) 두 경로가 같은 파일을 쓰므로 내보내기 규칙이 갈라지지 않는다 (산출물 manifest에 그 파일 sha256을
   남긴다). `yolox_ext.py`(공개 데이터 연합 주석의 배경 손실 무시)는 우리 데이터에 쓰지 않는다: 우리 검수 박스는
   클래스 목록 안에서 빠짐없이 다는 것을 전제로 한다 (아래 "한계"). YOLOX `Trainer`는 CUDA를 요구하므로 쓰지 않고 같은
   구성 요소(COCODataset·TrainTransform·get_model·get_optimizer·load_ckpt)로 짧은 학습 반복을 직접 돈다 (CPU·CUDA 공통).
6. **GPU가 없을 때.** 기본은 거절이다 (`YoloxTrainingError`). `--param allow_cpu=true`(또는 정책 `cpu.allowed`)면 "CPU
   매우 느림" 경고를 표준 오류에 내고 정책 상한(`cpu.max_epochs 1·max_iters 2·batch_size 2`)으로 자른 스모크만 돈다.
   이 모델은 성능을 기대하지 않는다. 배포 여부는 다른 모델과 같이 게이트가 정한다 (manifest `train_settings.device`).
7. **산출물.** 루프 계약(산출물 파일 하나)을 지키려고 `yolox-model.zip`(무압축, 시각 고정) 안에 `model.onnx`
   (decode_in_inference false, 입력 `images`, 출력 `output` (1, N, 5 + C))와 `manifest.json`을 둔다. manifest:
   데이터셋 버전, 예제 수(분할·변화별), 클래스 순서, 입력 크기, COCO 집계(이미지·박스·증강본·건너뛴 시각·클래스 밖
   예제), 학습 설정(장치·extra 포함)과 params·yolox 절 해시, 증강 해시·사본 수, 환경 잠금 파일·pyproject·학습 스크립트·
   공용 도구 sha256, 학습 기록(반복·손실·시간), ONNX 확인, 출발 가중치 계보, 상업 사용 분류, ONNX sha256. 루프가
   묶음의 sha256을 레지스트리(`model_versions.sha256`)에 남기고 MLflow에 올린다 (기존 그대로).
8. **로더.** `YoloxLoader`는 묶음을 풀어 형식·ONNX sha256·클래스 순서·입력 크기(지금 prelabel 정책과 같아야 함)·
   decode_in_inference·상업 사용 분류(지금 `models.yaml accept` 안)를 확인하고, 하나라도 어긋나면
   `ModelUnavailableError`다 (배포 단계는 기본 어댑터로 넘어가고, 루프 평가는 실패한다). 통과하면 `YoloxTools`의 탐지기
   자리에 `dlp_models.yolox.Yolox(학습 ONNX)`를 끼운 Predictor를 돌려준다. 이름은 `trained-objects`(라벨 ID 접두사,
   기본 `closed_tools`와 겹치지 않음)이고 추적·트랙 거르기·착용자 몸 억제(ADR 0058)는 그 어댑터 그대로다. 골든셋
   평가는 메모리에서만 하고(루프 기존 규칙), 배포 뒤 `dlp prelabel run`은 `deployed.deployed_predictors`로 붙인다.
9. **라이선스.** 출발 가중치는 `models.yaml`에 등록한 .pth만 쓴다 (`training.yolox.base_model`, 기본
   `yolox_s_coco_pth` = 공식 YOLOX-s COCO, Apache-2.0, 학습 데이터 COCO → review). `make licenses`의 사용 모델 목록에
   `training.yolox.base_model`을 넣었다 (`dlp_cli.models_cmds.used_models`). 학습기는 허용 분류 밖이거나
   `dlp_models.datasets` 위반이면 거절하고, 학습 모델은 출발 가중치 분류를 물려받는다 (우리 데이터셋 버전은 더 엄하게
   만들지 않는다, ADR 0058). 공개 데이터 1차 학습(ADR 0060) 체크포인트를 쓰려면 그 .pth를 `training_datasets`·
   `base_model: yolox_s_coco_pth`와 함께 등록하고 `base_model`을 바꾼다 (클래스 수가 같으면 머리까지 이어 받는다).
   학습 코드는 YOLOX(Apache-2.0)다 (AGPL Ultralytics 금지, ADR 0060 7절).

## 결정 B: AI 1차 검수 판정의 표시 보정 (`dlp aireview calibrate`)

VLM 자체 미세조정은 GPU가 필요해 이번 범위 밖이다 (아래 "다음 단계"). 지금은 사람 판정 기록으로 **표시를 보정**한다.

1. **학습 자료는 대조군만.** `ai_quality.load_ai_rows`의 행 중 대조군(holdout)·비교 가능(판정 있음, 사람이 봄, 재검수
   아님)·이중 작업 아님·판정 버전이 학습 버전 안인 것만 쓴다. 보인 집단은 검수자가 AI를 보고 한 결과라 지적을 따라
   고친 쪽으로 기운다 (ADR 0040 6절 3, 자동화 편향). 행 가중치(표본 묶음 모집단 / 표본 수)는 `ai_quality`와 같다.
2. **칸.** 보정 판정 값(wrong·suspect만, 정책 검증)마다 (판정), (판정, 라벨 종류), (판정, 종류, 클래스) 칸의 지적
   정밀도 P(사람이 바꿈 | 지적)와 한쪽 95% 상한(`ai_quality.weighted_rate`)을 낸다. 유효 표본 수가
   `min_effective_samples`(30)보다 적은 칸은 만들지 않는다: 더 넓은 칸, 그것도 없으면 보정 없음(지금 동작).
3. **보수적 행동.** 칸의 지적을 숨기는 것은 상한 < `hide_below_precision`(0.15), 즉 확실히 쓸모없을 때뿐이다. 확신도
   문턱은 그 아래 묶음 전체가 같은 기준으로 확실히 쓸모없는 가장 큰 관측 확신도로만 정한다 (확신도 없는 판정은 문턱으로
   숨기지 않는다). 칸마다 수축 정밀도(사전 표본 `prior_strength`)를 표시 순서 가중으로 둔다.
4. **표시만 바꾼다 (I9).** 보정은 검수 화면에 따로 표시할 의심·틀림만 거르고(노출 기록 hidden으로 간다) 같은 심각도
   안에서 정밀도 높은 칸을 먼저 보인다 (10분당 상한·CVAT 이슈 상한이 걸릴 때). AI의 ok·unsure, 판정 자체, 증가 프로필
   (우선순위 사유·표본 묶음 제외·마무리 보호)은 바꾸지 않는다. 숨긴 지적의 라벨도 표본 묶음에서 빠져 사람이 전수
   검수한다. 따라서 보정은 사람 검수를 줄이지 않는다.
5. **정책이 아니라 버전 붙은 산출물.** 결과는 `aireview-calibration-<id>.json`(정렬 JSON, `calibration_id` = 내용 해시
   12자, 판정 버전·기간·학습 설정·행 해시 포함)이다. 같은 기간·기록이면 같은 파일이라 명령은 멱등이다. review.yaml
   `ai_review.calibration.artifact`에 경로·sha256을 사람이 넣어야 쓰인다 (모델 가중치를 models.yaml에 고정하는 것과 같다).
   검수 CLI(`review_config.display_calibration`)는 sha256·내용 해시가 다르면 멈추고, 허용 판정 버전이 산출물 판정 버전에
   다 들지 않으면 쓰지 않고 알린다 (보정 없음 = 모두 표시).
6. **보고.** 명령은 보정 전후 지적 정밀도·재현율(ai_quality 정의, 행 가중)을 같은 자료와 작업 키 해시 2겹 교차로
   낸다 (같은 자료 값은 낙관적이다). 노출 기록(`ai_review_exposures`)에 어느 보정을 썼는지는 남기지 않는다 (계약
   변경 회피). 산출물이 sha256으로 review.yaml에 고정되므로 기간별 정책 이력으로 추적한다 (후속: 노출 기록 필드,
   F-AIR-35).

## 다음 단계 (설계만): VLM 판정기 LoRA 미세조정

- 사내 GPU에서만 한다 (외부 API·외부 GPU 금지: 블러본이라도 우리 데이터다, ADR 0040 I7·ADR 0060 6절).
- 학습 예제: `ai_review_outcomes`의 대조군·보인 집단 행을 판정 단위(`ai_review_units`의 입력 키: 블러본 프레임·맥락·
  온톨로지)와 묶어 "사람이 바꿨는가/무엇으로"를 정답으로 쓴다. 입력 픽셀은 `open_blurred`로만, 골든셋 세션은 빼고,
  데이터셋 버전 분할을 따른다 (작업자·장소 누출 금지).
- 평가·배포: 기존 골든 평가 `dlp aireview eval`(교란·이력, DB에 쓰지 않음)로 지금 판정 버전과 비교하고, 게이트를 통과한
  어댑터만 서버에 올린다. 판정 버전 문자열에 어댑터 해시를 넣어 재판정이 일어나게 한다 (I8). 상업 사용 분류는 기본
  모델(Qwen3-VL-8B)을 물려받는다.

## 결과

- 실행 확인 (2026-10-27, 이 컨테이너, GPU 없음): 격리 환경 `--extra cpu` 설치(약 1.1 GB, torch 2.11.0+cpu), YOLOX 0.3.0
  clone(12 MB), 합성 블러본으로 CPU 2반복 학습·ONNX 내보내기(출력 (1, 8400, 26), PyTorch 대비 최대 차이 1e-7~4e-5),
  로더가 그 ONNX로 어댑터를 돌림. 공식 yolox_s.pth에서 시작하면 텐서 462개 중 382개를 이어 받고(클래스 수가 다른 머리
  제외) 증강본이 COCO에 더해진다. 시험: `packages/training/tests/test_yolox.py`(PyTorch 없음, 손으로 부호화한 작은 ONNX로
  로더 종단), `test_yolox_isolated.py`(`isolated_env`).
- GPU 학습 서버에서의 실제 미세조정, 사람 검수 뒤 정확도 향상, 골든 게이트 통과는 아직 확인하지 않았다 (TODO(real-model)
  는 나머지 과제에 남는다).
- 한계: (1) 검수 박스가 클래스 목록 안에서 빠짐없이 달렸다고 가정한다. 사람이 안 단 물건은 배경으로 배운다. (2) 미검수
  모델 라벨은 예제가 아니므로 같은 프레임에 있어도 배경이 된다 (데이터셋 버전이 human_verified 세션이면 드물다).
  (3) 학습 이미지는 블러본이고 골든·프리라벨 추론은 원본이다 (얼굴 모자이크 영역의 분포 차이, 물건 탐지에는 작다고
  본다). (4) 학습이 루프 DB 트랜잭션 안에서 오래 돈다 (기존 루프 구조, F-ETA-18). (5) CUDA판 잠금은 x86_64 기준이다 (DGX Spark
  aarch64 휠은 확인 필요). (6) 표시 보정은 대조군이 10%라 칸이 채워지려면 기록이 많이 필요하다 (지금 기본은 보정 없음).
