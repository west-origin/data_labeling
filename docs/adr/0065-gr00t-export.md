# ADR 0065: NVIDIA Isaac GR00T N1.7 미세조정 형식 내보내기 (`dlp export gr00t`)

- 상태: 채택 (구현됨. GR00T 자체 로더로 읽어 보는 확인은 아직, 아래 "확인하지 못한 것")
- 날짜: 2026-11-04
- 관련: ADR 0018·0021·0027(내보내기·이력·가명), 0019(시각 규약), 0043(설명 번역), 0045·0053(손 3D 점·신뢰도),
  0046(세션 월드 좌표), 0058(착용자 팔)
- 계약 변경: 없음 (`dlp_schema` 그대로). 정책 `config/policies/export.yaml gr00t` 절과 `dlp_export.policy.Gr00tPolicy`를
  더했다.

## 배경

사용자는 내보낸 데이터가 NVIDIA가 공개한 요구를 따르기를 원한다. 대상은 NVIDIA Isaac GR00T N1.7 미세조정 데이터 형식
("GR00T LeRobot")이다. 근거로 읽은 것 (github.com/NVIDIA/Isaac-GR00T, main 커밋 c56f541, 2026-10-28):

- `getting_started/data_preparation.md`: 배치, parquet 열, `meta/modality.json` 형식, "annotation 접두어는 언어 주석만".
- `getting_started/data_config.md`, `finetune_new_embodiment.md`: `ActionConfig`(RELATIVE·ABSOLUTE, EEF·NON_EEF),
  "상대 행동을 쓰려면 데이터의 state·action은 절대값이어야 하고 변환은 처리기가 한다", EEF는 xyz + rot6d 9차원.
- `gr00t/data/dataset/lerobot_episode_loader.py`: info.json `chunks_size`·`data_path`·`video_path`·`features`를 읽고,
  episodes.jsonl `length`로 길이를 정하고, 주석 열 값으로 `tasks.jsonl`을 바로 찾고(`tasks_map[x]`), 영상은 행 번호로
  디코딩한다 (`get_frames_by_indices`, torchcodec).
- `gr00t/data/stats.py`: info.json에서 dtype에 float가 든 특징마다 parquet 열을 읽어 통계를 낸다. 상대 행동 통계는
  행동 길이(기본 16) 이상인 에피소드에서만 나온다.
- `tests/gr00t/data/test_data_preparation.py`, `demo_data/cube_to_bowl_5/meta/*`와 에피소드 0 parquet: 실제 예제의 열과
  meta (코덱 av1, next.reward·next.done 열 없음, 주석은 `"original_key": "task_index"`).

우리 LeRobot 내보내기(`dlp export lerobot`)는 LeRobot 0.6.1 공식 API로 쓰는 v3.0이고, `annotation.hand_state` 같은
배열 열이 "annotation 접두어는 언어 주석만" 규칙을 어긴다. GR00T는 v2를 읽는다 (v3는 NVIDIA 변환 스크립트로 바꾸라고 한다).

## 결정

1. **새 형식 `dlp export gr00t <데이터셋 버전>`** (`dlp_export.gr00t`). 에피소드는 LeRobot 형식과 같은 `build_episode`
   결과를 쓴다 (`runner.build_episodes`로 두 형식이 같은 코드를 탄다). 데이터셋 버전에서만, 블러본만, 내보내기마다 다른
   가명, 블러 라벨·원본 위치·검수자 ID 없음, 기본은 검수한 라벨만 (CLAUDE.md 규칙, `run_export`가 그대로 적용).
2. **LeRobot v2.1 배치를 pyarrow로 직접 쓴다.** LeRobot 0.6.1은 v3.0만 쓰고 PyTorch를 끌어오므로, 격리 환경 없이
   작업공간의 pyarrow·ffmpeg로 쓴다. stats.json·relative_stats.json은 GR00T가 계산하므로 쓰지 않는다.
3. **영상은 H.264(yuv420p) 고정 fps로 다시 인코딩한다.** 행 k의 영상 프레임은 LeRobot 형식과 같은 PTS 규칙
   (`PtsIndex.frame_at`, 마스터 시각 → 스트림 시각)으로 고른 블러본 프레임이다. 고른 프레임을 rawvideo로 ffmpeg에
   넣으므로 영상 프레임 수 = parquet 행 수이고, 프레임 k의 표시 시각 = k/fps = parquet `timestamp`다. VFR 블러본의 반복
   프레임도 행마다 들어간다. H.264를 고른 까닭: NVIDIA 예제는 av1이지만 torchcodec의 av1 디코딩은 FFmpeg 빌드에 달려
   있다 (`gr00t/utils/video_utils.py`가 FFmpeg 판 불일치를 흔한 실패로 적는다). 값은 `gr00t.video`(crf·preset·gop).
4. **상태·행동 (사람 1인칭 시연).**
   - `observation.state` = 손마다 손 3D 점(`lerobot.hand_points_3d`, 지금 손목·다섯 손가락 끝 6점) x, y, z + 손마다
     착용자 팔 3D 점(`lerobot.arm_points_3d`, 지금 팔꿈치). 카메라 좌표 m. 길이 2*6*3 + 2*1*3 = 42.
   - `action` = 다음 프레임 state (절대값, 마지막 프레임은 자기 자신). GR00T 문서가 상대 행동은 처리기가 만들고 데이터는
     절대값이어야 한다고 하므로 절대값을 둔다. 상대 행동은 미세조정 설정에서 `ActionRepresentation.RELATIVE`로 켠다.
   - `ActionType.EEF`는 위치 + 6D 회전 9차원을 요구한다. 우리 손 라벨은 점 위치만 있고 손 방향 라벨이 없다. 세 점으로
     회전을 만들면 검수하지 않은 값을 지어내는 것이라 하지 않는다. 그래서 `NON_EEF` + `RELATIVE`를 권한다 (아래 예).
     N1.7의 사람 영상 사전학습은 상대 EEF라고 README가 적지만, 우리 데이터로는 EEF 회전을 낼 수 없다.
   - 값이 없는 점은 0이고, 점마다 `observation.hand_valid`(float32, 1 있음·0 없음)를 둔다. annotation 접두어가 아니어서
     GR00T 규칙에 걸리지 않고, GR00T는 읽지 않는다.
5. **언어 주석.** `annotation.human.task_description` = 그 프레임 작업 문장 번호 (`task_index`와 같다),
   `annotation.human.action.task_description` = 그 프레임 행동 문장 번호. 둘 다 같은 `tasks.jsonl`을 가리킨다 (로더가
   하나만 읽는다). 문장은 `gr00t.language`(en) 설명 라벨(같은 언어 원문, 또는 원문이 같은 내보내기에 있는 번역, ADR
   0043), 없으면 온톨로지 영어 이름(작업 → 도메인, 행동은 동사)이다. 덮는 행동이 없으면 `gr00t.idle_action_text`
   ("no action") 문장을 가리킨다 (GR00T는 번호로 바로 찾으므로 -1을 쓸 수 없다). 두 손 행동 문장이 다르면
   "left hand: …; right hand: …"로 잇는다.
6. **그 밖의 라벨 열은 `labels.*`로 옮긴다.** LeRobot 형식의 annotation.* 열(손 상태·도구-표면 접촉·동사·검증 등급·
   접촉 객체 클래스·손 3D 신뢰도·방식·내부 파라미터 출처·착용자 팔)과 LeRobot 형식 `observation.state` 전체
   (`labels.dlp_state`, 2D 관절·도구 작용부 포함)를 같은 값·dtype·names로 싣는다. 어휘는 `meta/dlp_vocab.json`.
7. **확인기 `dlp export gr00t-check <디렉터리>`** (`dlp_export.gr00t_check.check_dataset`). 쓰기 코드와 따로 NVIDIA
   문서·로더 요구만으로 배치·열·dtype·modality 구간·주석 번호 범위·영상(ffprobe 코덱, 프레임 수, 프레임 시각)을 본다.
   `dlp export gr00t`는 쓴 뒤 이 확인을 돌려 문제가 있으면 올리지 않고 실패한다 (manifest `details.gr00t_check`).
   확인기는 NVIDIA 예제(cube_to_bowl_5)의 meta·parquet도 받아들인다 (2026-11-04 수동 확인, 영상은 내려받지 않음).

## 대응표

| GR00T 위치 | 우리 값 | 출처 |
| --- | --- | --- |
| `meta/info.json` `codebase_version` | `v2.1` | 고정 |
| `fps` | `export.yaml lerobot.fps` (30) | 정책 |
| `robot_type` | `lerobot.robot_type` (`human_bodycam`) | 정책 |
| `chunks_size` | `gr00t.chunks_size` (1000) | 정책 |
| `splits` | 분할별 연속 에피소드 구간 (에피소드를 분할 순으로 다시 번호 매김) | 데이터셋 버전 분할 |
| `modality.json state`·`action` | `left_hand` 0:18, `right_hand` 18:36, `left_arm` 36:39, `right_arm` 39:42 | `StateLayout` |
| `modality.json video` | `ego_view` → `observation.images.bodycam` | `gr00t.video_name`, `lerobot.video_stream` |
| `modality.json annotation` | `human.task_description`, `human.action.task_description` | 고정 |
| `observation.state` float32[42] | 손·팔 3D 점, 카메라 좌표 m, 없으면 0 | 3D 궤적 라벨 (`build_episode`) |
| `observation.hand_valid` float32[14] | 점마다 1/0 | 같은 곳 |
| `action` float32[42] | 다음 프레임 state | 같은 곳 |
| `timestamp` float32 | k/fps (초) | 고정 fps |
| `frame_index`, `episode_index`, `index`, `task_index` int64 | 행 번호, 에피소드 번호, 전체 일련번호, 작업 문장 번호 | |
| `annotation.human.task_description` int64 | 작업 문장 번호 | 작업 구간·설명·온톨로지 |
| `annotation.human.action.task_description` int64 | 행동 문장 번호 | 행동·설명·동사 |
| `next.reward` float32, `next.done` bool | 0, 마지막 프레임만 참 | 고정 |
| `labels.*` | LeRobot 형식 annotation.* 와 observation.state | `build_episode` |
| `videos/chunk-000/observation.images.bodycam/episode_XXXXXX.mp4` | H.264 yuv420p, 고정 fps, 소리·메타데이터 없음 | 승인된 블러본 |
| `meta/dlp_vocab.json`, `dlp_episodes.json`, `dlp_gr00t.json` | 어휘, 에피소드 → 세션 가명·분할, 좌표계·action 정의 | GR00T가 읽지 않는 추가 파일 |

## NVIDIA 요구 충족

| 요구 (data_preparation.md·로더) | 충족 | 근거 |
| --- | --- | --- |
| meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}, data/chunk-*, videos/chunk-* | 예 | 확인기 1번 |
| mp4, `observation.images.<이름>`, `episode_00000X` 이름 | 예 | info.json `video_path` 틀 |
| `observation.state`·`action` = float32 1차원 이어붙임, modality.json 순서 | 예 | 확인기 3·5번 |
| `timestamp` (초) | 예 | k/fps, 영상 프레임 시각과 같다 |
| 주석 열 `annotation.<source>.<type>`, 값 = tasks.jsonl 번호 | 예 | 두 열, 번호 범위 확인 |
| annotation 접두어 열은 주석만 | 예 | 다른 라벨은 `labels.*` |
| modality.json 구간 0부터, `[start:end)`, 영상 `original_key`, annotation 키 | 예 | 확인기 3번 |
| `task_index`, `episode_index`, `index`, `next.reward`, `next.done` | 예 | 확인기 5번 |
| episodes.jsonl `length` = parquet 행 수 = 영상 프레임 수 | 예 | 확인기 5·6번 |
| 영상 디코딩 (torchcodec) | H.264로 맞춤 | 실제 torchcodec 디코딩은 아래 "확인하지 못한 것" |
| stats.json·relative_stats.json | 해당 없음 | GR00T가 `gr00t/data/stats.py`로 만든다 |
| "Proprioceptive states must always be included in observation.state" | 아니오 (대체) | 아래 |

## 충족하지 못하는 것과 모르는 것

- **로봇 고유 감각이 없다.** 사람 시연이라 관절 각도·그리퍼 값이 없다. 대신 손·팔 3D 점을 state로 둔다. 이 점은 가슴에
  단 카메라 좌표라 착용자가 움직이면 손이 가만히 있어도 값이 바뀐다 (프레임 사이 상대 행동에 카메라 움직임이 섞인다).
  세션 월드 좌표 손 점(ADR 0046)은 카메라 자세가 있는 세션에만 있어 이번에는 넣지 않았다. 필요하면 별도 상태 키로 더한다.
- **N1.7 사람 데이터 미세조정은 문서에 없다.** README는 N1.7이 EgoScale 사람 영상 2만 시간으로 사전학습했다고 적고,
  사전학습 태그 `REAL_R1_PRO_SHARPA_HUMAN`이 있지만, 그 모달리티 설정은 공개 저장소에 없다 (기본 모델 처리기 설정에만
  있다). 따라서 이 데이터는 `NEW_EMBODIMENT` + 직접 쓴 모달리티 설정으로 미세조정해야 하고, 사람 손 데이터만으로 로봇
  정책이 좋아지는지는 확인하지 않았다.
- **EEF 행동이 아니다.** 손 방향(회전) 라벨이 없어 `ActionType.EEF`(xyz + rot6d)를 낼 수 없다. NON_EEF 상대 행동이다.
- **손 점 6개뿐이다.** 온톨로지 `hand_joints`가 손목과 다섯 손가락 끝만 정의한다 (21관절 3D가 아니다). 21관절 2D는
  `labels.dlp_state`에 있다.
- **값 없음 = 0.** GR00T 통계(평균·분산)와 상대 행동은 0을 실제 값으로 본다. 손이 오래 안 보이는 구간은
  `observation.hand_valid`로 걸러야 한다 (GR00T에는 상태 가림 기능이 없다).
- **짧은 에피소드.** 행동 길이 16 프레임보다 짧은 에피소드는 GR00T 상대 행동 통계에 들어가지 않는다.
- **확인하지 못한 것.** 이 환경에서 GR00T 저장소 코드(통계 생성기·에피소드 로더·torchcodec)를 직접 돌려 읽는 확인은
  하지 못했다 (외부 코드 실행 승인 필요). 확인기는 그 코드와 문서를 읽고 같은 조건을 다시 짠 것이다. 승인되면
  `gr00t/data/stats.py`와 `LeRobotEpisodeLoader`로 내보낸 결과를 읽어 보는 것이 다음 확인이다.

## 미세조정 설정 예 (GR00T 저장소에서)

```python
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

KEYS = ["left_hand", "right_hand", "left_arm", "right_arm"]  # meta/modality.json state·action 키
config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=KEYS),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=KEYS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.action.task_description"]
    ),
}
register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

## 결과

- 운영: `dlp export gr00t <버전> --target <받는 곳>` → 데이터셋 버킷 `exports/<ID>/gr00t/`. 받은 쪽은
  `dlp export gr00t-check <디렉터리>`로 다시 확인할 수 있다. 기존 형식(coco·intervals·lerobot)의 결과는 바뀌지 않는다
  (`runner.build_episodes`는 LeRobot 경로의 반복을 그대로 옮긴 것이다).
- CLI: `dlp export`가 형식마다 하위 명령을 갖게 됐다 (명령줄 모양은 같다. `--robot`은 lerobot에만 있다).
- 시험: `packages/export/tests/test_gr00t.py`(DB 없이 쓰기 → 확인 통과, 열·dtype·값·문장, 영상 프레임 = PTS로 고른
  블러본 프레임, 위반 데이터셋 거부), `test_runner.py::test_gr00t_export_end_to_end`(services, 데이터셋 버전에서 올리기까지).
