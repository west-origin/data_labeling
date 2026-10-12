# ADR 0038: 검수 화면 테마를 도구 밖에서 덧씌운다

- 상태: 채택
- 날짜: 2026-10-11
- 관련: WP6(검수 도구 연동), ADR 0006(무손실 왕복)·0037(온톨로지별 검수 프로젝트).
  `config/review_ui.yaml`, `dlp_review.ui_theme`, `services/review-theme/`.

## 배경

검수자가 쓰는 화면(CVAT 블러·공간 검수, Label Studio 행동 구간 검수)을 회사 디자인 목업에 맞추고
다크/화이트 모드를 전환할 수 있게 해 달라는 요청이 있었다. 목업을 그대로 만들려면 두 도구의 화면
소스(React)를 고쳐 빌드해야 한다. 그러면 몇 주가 들고, 도구 버전을 올릴 때마다 다시 맞춰야 한다.

## 결정

1. 화면 소스는 고치지 않는다. 도구가 밖에서 받아 주는 값만 바꾼다: 색, 로고, 단축키, 영상 높이,
   전역 CSS. 라벨 이름·컨트롤 이름·저장 형식·검수 흐름은 그대로라 왕복(ADR 0006)과 수집 코드는
   영향이 없다.
2. 값은 `config/review_ui.yaml` 한 곳에 둔다 (모드별 색, 라벨 색, 원시 동작 단축키, 채널 색, 화면 크기,
   로고 경로). 코드에 색을 하드코딩하지 않는다.
3. Label Studio: `label_config(ontology, theme, logo_uri=...)`가 라벨 `background`·`hotkey`, 채널
   `strokeColor`·`height`, 영상 `height`, 머리글(`className="dlp-brand"`)과 `<Style>`(두 모드 색 변수,
   로고)을 넣는다. 다크/화이트 전환은 Label Studio 1.23에 내장된 Theme 설정
   (`<html data-color-scheme>`, 사용자 메뉴)을 그대로 쓴다. 기존 프로젝트는 `dlp review create` 때
   `ensure_label_config`로 갱신된다. 단축키: 오른손 원시 동작은 숫자, 왼손은 shift+숫자.
   (추가 기록, WP17.6) 표에 없는 라벨은 Label Studio가 자동 키 후보에서 남는 키를 설정 순서대로
   붙인다. 그래서 새 명시 단축키는 후보 밖의 키로 주고(AI 결정 r·h), 동사 표를 바꾸면 자동 키가 밀림을
   검수자에게 알린다 (`ui_theme.LS_AUTO_HOTKEYS`, change-guide).
4. CVAT: 공식 `cvat/ui` 이미지 위에 테마 파일(`/dlp/` 아래 CSS·JS·로고)을 얹은 `dlp-cvat-ui:<버전>`
   이미지를 `services/cvat/cvat.sh up`이 빌드해 쓴다. 스크립트는 다크/화이트 전환 버튼(브라우저에 기억),
   회사 로고, 작업 화면의 검수 단계 배지(블러 검수=원본 영상·권한자 전용, 공간 검수=블러본)를 넣는다.
   배지는 프로젝트 이름 접두사(`PRIVACY_PROJECT`·`SPATIAL_PROJECT`)로 정하는 안내일 뿐 권한 검사가 아니다.
   색 값은 생성 파일 `services/review-theme/cvat/dlp-tokens.css`(`make ui-theme`)에서 읽고,
   `make contracts`가 설정과의 일치를 검사한다.
5. CVAT 라벨 색은 프로젝트를 만들 때 넣고, 이미 있는 프로젝트는 색이 다른 라벨만 바꾼다
   (`set_label_color`, 라벨 ID·속성 불변). 표에 없는 라벨은 이름의 해시로 팔레트에서 고정된 색을 쓴다.

## 결과

- 목업과 같은 색·로고·모드 전환·단계 배지·라벨 색·단축키는 반영된다. 목업의 "지금 이 시각" 패널,
  손별 라벨 줄(레인) 배치, 겹치는 행동 추천, CVAT 오른쪽 패널 재구성은 화면 소스를 고쳐야 해서
  하지 않았다.
- CVAT 테마 CSS는 CVAT v2.78의 클래스 이름에 기대므로 버전을 올리면 화면을 열어 확인한다
  (docs/handover/change-guide.md). Label Studio의 색 변수 이름도 1.23 기준이다.
- Label Studio의 처음 모드는 그 브라우저의 Theme 설정(기본 Auto=운영체제 설정)을 따른다.
  `default_mode`는 CVAT에만 적용된다. Label Studio 상단 메뉴의 로고는 바꾸지 않는다(작업 화면 머리글에만
  회사 로고).
