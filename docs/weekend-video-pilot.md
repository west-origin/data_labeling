# 주말 청소 영상 첫 파일럿 실행 안내

목표는 **손·객체·청소 도구의 자동 초안 → 사람 수정 → 라벨 이력 저장**이다.
자동 청소 행동 분류용 VLM과 실제 재학습기는 이번 준비 범위에 포함하지 않는다.
최신 검증 결과와 남은 문제는 [인수인계의 현재 과제](handover/known-issues.md)에 기록한다.
다른 개발 환경에서 시작한다면 먼저 [환경 준비](handover/development.md)를 따른다.
선택 큐레이션 도구 FiftyOne의 GraphQL 의존성 취약점 2건은 호환 패치가 남아 있다. 이 파일럿
절차는 해당 웹 서버를 시작하지 않는다. 외부에 제공하려면 별도로 해결해야 한다.

## 촬영과 첫 시험 범위

- 원본 파일을 보존하고, 카메라를 안정적으로 고정해 손·도구·작업면이 함께 보이게 촬영한다.
- 얼굴, 문서, 주소, 화면, 거울 속 얼굴도 프라이버시 검수 대상이다. 촬영 동의와 현장 촬영 권한을 확인한다.
- 첫 시험은 짧은 구간으로 시작한다. 이 환경은 CPU이며 OWLv2 프라이버시/도구 처리는 손 모델보다
  느리다. 긴 원본 전체를 먼저 올리기 전에 처리시간·검수 품질·여유 디스크를 확인한다.
- 현재 손 좌우 판정·장갑·가림·모션 블러의 정확도는 실촬영 정답과 비교해야 한다.
  `prelabel.yaml hands.input_is_mirrored`는 실제 바디캠의 거울상 여부에 맞춘다.
- 작업자 ID·장소 ID·동의서 버전·실제 촬영 시각을 준비한다. 영상만으로 동의를 추정하지 않는다.

## 이 클라우드 환경 시작

아래 절차는 현재 클라우드 전용이다. 다른 환경은 위 준비 안내를 통과한 뒤 `입력 준비와 등록`부터 따른다.

저장소 루트에서 실행한다. 파일럿 구성은 저장소 밖 `/workspace/.cloud-onboarding/data-labeling/`에
있으며, 환경 설정 초안에도 재사용 설치·시작 절차를 저장한다. 새 환경으로 복원하는 것은 별도 검증 사항이다.

```bash
cd /workspace/data_labeling
/workspace/.cloud-onboarding/data-labeling/start.sh
/workspace/.cloud-onboarding/data-labeling/start-cvat.sh
. /workspace/.cloud-onboarding/data-labeling/pilot/env.sh
uv run --no-sync python /workspace/.cloud-onboarding/data-labeling/pilot/bootstrap-cvat.py
uv run --no-sync python /workspace/.cloud-onboarding/data-labeling/pilot/bootstrap-label-studio.py
uv run --no-sync dlp pilot check --profile cleaning
```

CVAT는 공식 v2.78.0 서버와 UI를 사용한다. 이 작은 클라우드에서는 서버·필수 큐 작업자를 한
컨테이너에서 실행한다. 분석 대시보드·분석 이벤트 전송·품질 보고/합의 작업자는 기동하지 않는다.
Docker VFS의 이미지 복제 때문에 이 환경에서 일반 `make cvat-up`을 실행하면 디스크가 부족해질 수
있다. 위 시작 스크립트를 사용한다. 앱의 DB 감사 기록은 별도로 유지된다.

검수 계정은 `pilot-privacy`와 `pilot-labeler`로 분리했다. 계정 비밀번호는 Git과 보고서에 넣지 않고
로컬 `/workspace/.cloud-onboarding/data-labeling/pilot/cvat-accounts.json`에 권한 0600으로 저장했다.
기존 계정의 비밀번호·관리자 권한은 설치 재실행으로 바꾸지 않는다. 관리자 계정으로 일반 검수하지 않는다.
Label Studio에도 서비스 계정과 별도로 `pilot-labeler@example.invalid` 검수 계정을 준비했다.
로그인 정보는 같은 디렉터리의 `label-studio-account.json`(권한 0600)에 있다. 검수 정책 파일
(`DLP_REVIEW_POLICY_FILE`)의 `label_studio.users`에 `pilot-labeler: pilot-labeler@example.invalid`를
연결해야 한다. 연결이 없으면 Label Studio 시간 검수 작업을 만들지 않고(`dlp pilot check`의
`label-studio-account`도 실패), 수집은 이 계정이 제출한 주석만 받는다. 다른 계정(서비스 계정 포함)이
같은 작업에 낸 주석은 결과로 쓰지 않는다.

## 입력 준비와 등록

먼저 컨테이너 정보와 방향을 확인한다. 정규화 명령은 원본·기존 출력을 덮어쓰지 않는다.
지원하지 않는 입력이나 PTS가 바뀌는 변환은 출력 공개 전에 실패한다.

```bash
uv run --no-sync dlp media probe /path/to/original.mp4
uv run --no-sync dlp media prepare /path/to/original.mp4 --out /path/to/prepared.mp4
uv run --no-sync dlp pilot check --profile cleaning --video /path/to/prepared.mp4
```

입력 매니페스트에는 다음 형식을 쓰되, 꺾쇠 괄호 자리는 실제 정보로 채운다. 아래는 실행할 수 있는
예제 동의 정보가 아니라 작성 양식이다. 새 세션 ID를 사용하고 이미 검수한 세션의 영상을 교체하지 않는다.

```yaml
session_id: <새로운-세션-ID>
domain: cleaning
worker_id: <실제-작업자-ID>
site_id: <실제-장소-ID>
consent_version: <확인한-동의서-버전>
recorded_at: <시간대를-포함한-실제-촬영시각>
ontology_version: 1.0.0
streams:
  - stream_id: bodycam
    kind: bodycam
    path: /path/to/prepared.mp4
```

```bash
uv run --no-sync dlp ingest /path/to/session.yaml
uv run --no-sync dlp privacy detect SESSION_ID
uv run --no-sync dlp review create SESSION_ID --stage privacy --assignee pilot-privacy
```

검수 명령은 준비 검사와 같은 검수 정책 파일(`DLP_REVIEW_POLICY_FILE`)을 읽는다. `create`·`assign`은
원본 검수자와 CVAT·Label Studio 계정 연결을, `serve`는 두 도구의 계정 연결(웹훅 담당자 확인)을,
`collect`는 표본 판정 정책과 Label Studio 계정 연결을 이 파일에서 읽는다. 준비 검사를 통과한 셸에서 이어서 실행한다. 이 변수가 없는 셸에서는 Git의
기본 정책(원본 검수자·계정 연결 없음)을 읽어 블러 검수 작업 생성이 원본 접근 권한자가 아니라는
오류로 거부된다.

## 사람 검수에서 두 번 멈춘다

1. **프라이버시 전수 검수:** 출력된 CVAT 작업을 `pilot-privacy`로 열어 영상 전체의 누락·잘못된
   블러를 수정하고 작업을 완료한다. 탐지된 라벨이 0개여도 영상 전체를 확인한다. 실제 사람이
   완료한 뒤 해당 작업 키를 수집한다. 여러 스트림이면 모든 프라이버시 작업을 완료한다.

   ```bash
   uv run --no-sync dlp review collect cvat:TASK_ID --reviewer pilot-privacy
   uv run --no-sync dlp privacy approve SESSION_ID
   uv run --no-sync dlp privacy render SESSION_ID
   ```

2. **일반 라벨 검수:** 전신/깊이 3D는 명시적으로 제외하고 손·객체·도구 초안을 생성한다.
   필요한 모델이 없으면 기본 실행은 실패한다. 준비 검사 실패를 `--allow-missing-models`로 우회하지 않는다.

   ```bash
   uv run --no-sync dlp prelabel run SESSION_ID --skip body --skip depth3d
   uv run --no-sync dlp review create SESSION_ID --stage labeling --assignee pilot-labeler
   ```

   `pilot-labeler`로 공간 라벨의 관절·좌우 손·객체 클래스·박스를 확인하고 수정한다. 출력된 CVAT
   작업 키를 실제 완료 후 `dlp review collect cvat:TASK_ID --reviewer pilot-labeler`로 수집한다.
   이 명령은 Label Studio 시간 검수 작업도 만든다. 위 검수 이메일 계정으로 제출하고,
   `dlp review collect label_studio:TASK_ID --reviewer pilot-labeler`로 해당 작업도 수집한다.
   다른 계정으로 제출하면 "담당자 … 가 제출한 주석이 아직 없습니다 (다른 사용자 주석 N개는 받지 않음)"
   오류가 난다. 담당자 계정으로 다시 제출한다.
   마지막으로 `dlp review verify SESSION_ID`를 실행한다. 미완료 사유가 있으면 종료 코드 1과 사유를
   확인해 해결한다. 빈 결과나 자동 초안을 사람 검수 완료로 취급하지 않는다.

웹훅 자동 수집 없이도 위 수동 수집 경로로 첫 시험을 진행할 수 있다. 실패한 웹훅 사건도 같은 경로로
복구할 수 있다. 사람이 고친 라벨은 이력으로 남지만, 그 사실만으로 모델 가중치가 자동 개선되지는 않는다.

## 기록할 결과와 실패 대응

처리시간, 미검출·오검출·좌우 손 오류, 관절 위치 오차, 도구 클래스 오류, 블러 누락, 사람 수정시간을
기록한다. 샘플별 정답을 표시하기 전에는 모델 confidence나 검출 프레임 비율을 정확도로 보고하지 않는다.

- 모델/서비스 검사 실패: `[FAIL]` 항목을 해결한 뒤 재검사한다. 검증 실패를 승인 명령으로 넘기지 않는다.
- 변환 실패: 원본을 보존하고 코덱·트랙·방향·PTS 오류를 확인한다. 프레임 번호 × FPS로 시간을 대체하지 않는다.
- 검수 작업 생성 실패: DB는 롤백돼도 CVAT에 사본이 남을 수 있으므로 도구의 작업 상태를 확인한다.
- 디스크 부족: 원본이나 DB 볼륨을 지우지 않는다. 짧은 첫 구간으로 시험하고 대용량 처리 환경을 따로 준비한다.
- 파일럿 종료: CVAT에 올라간 원본 프록시는 현재 보관 명령이 자동 삭제하지 않는다. 권한자가 도구 사본도
  삭제하고 완료를 확인해야 한다. 고객 데이터 공개, 여러 고객 운영, 자동 재학습·판매 승인과는 별도 절차다.
