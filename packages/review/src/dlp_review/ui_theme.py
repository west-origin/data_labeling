"""검수 화면 테마 (`config/review_ui.yaml`). CVAT·Label Studio 위에 덧씌우는 모양만 정한다.

도구의 화면 소스는 고치지 않는다. 바꾸는 것은 색·로고·단축키·영상 높이처럼 도구가 밖에서 받아 주는
값뿐이고, 라벨 이름·검수 흐름·저장 형식은 그대로다 (ADR 0038).

- Label Studio: `labelstudio.label_config`가 이 테마로 라벨 색(`background`), 원시 동작 단축키
  (`hotkey`), 채널 선 색, 영상 높이, 화면 CSS(`<Style>`)를 넣는다. 다크/화이트 전환은 Label Studio에
  내장된 Theme 설정(사용자 메뉴, `<html data-color-scheme>`)을 그대로 쓰고, CSS는 두 모드의 색만
  바꾼다.
- CVAT: 공식 UI 이미지에 테마 CSS·JS·로고를 얹은 이미지(services/review-theme/cvat)를 쓴다. 색 값은
  `cvat_tokens_css`가 만든 생성 파일(`TOKENS_FILE`, `make ui-theme`)에서 읽는다. 라벨 색은 작업을
  만들 때 프로젝트 라벨에 넣는다 (`cvat_label_color`).

공개 이름:
- `Mode`: 화면 모드 (dark, light).
- `ModeColors`·`ModesTheme`·`BrandTheme`·`LabelColors`·`LayoutTheme`·`AiColors`·`AiHotkeys`·
  `AiTheme`·`UiTheme`: review_ui.yaml 절별 모델 (`ai`: AI 1차 검수 판정 색·결정 단축키, WP17.6).
- `TOKENS_FILE`: CVAT 테마 색 생성 파일 경로 (저장소 루트 기준).
- `LS_AUTO_HOTKEYS`·`LS_RESERVED_KEYS`: Label Studio(고정 1.23.2)가 단축키 없는 라벨에 스스로 붙이는
  키 후보와 내장 한 글자 단축키 (단축키 검증에 쓴다).
- `load_ui_theme`: 저장소 루트에서 테마를 읽어 검증한다.
- `ls_label_color`·`ls_hotkey`: Label Studio 라벨 이름 → 색·단축키.
- `cvat_label_color`: CVAT 라벨 이름 → 색.
- `logo_data_uri`: 로고 파일 → data URI (Label Studio CSS에 넣는다).
- `ls_style`: Label Studio 화면 CSS.
- `AI_NOTE_EMPTY`: AI 지적 없는 구간의 메모 안내 (Label Studio 입력란 자리 글과 CSS 안내가 같이
  쓴다).
- `cvat_tokens_css`: CVAT 테마 색 CSS (생성 파일 내용).
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, StringConstraints, model_validator

from dlp_schema.common import Contract
from dlp_schema.ontology import Ontology

# 화면 모드. CVAT는 <html data-dlp-mode>, Label Studio는 <html data-color-scheme>에
# 같은 이름을 쓴다.
Mode = Literal["dark", "light"]
# #rrggbb 색
HexColor = Annotated[str, StringConstraints(pattern=r"^#[0-9a-fA-F]{6}$")]

# CVAT 테마 색 생성 파일 (저장소 루트 기준). `make ui-theme`이 쓰고 테스트가 내용 일치를 검사한다.
TOKENS_FILE = "services/review-theme/cvat/dlp-tokens.css"

# Label Studio 1.23.2 화면이 단축키 없는 컨트롤(라벨·선택지·시계열)에 설정 순서대로 붙이는
# 키 후보 (프런트엔드 `Hotkey.makeComb`). 명시 단축키로 이미 쓴 키는 건너뛴다. 그래서 명시
# 단축키를 이 후보 안의 키로 새로 주면 그 뒤 라벨들의 자동 키가 한 칸씩 밀린다 (원시 동작
# pull·rotate, 사이 구간, 손 상태는 자동 키에 기댄다). AI 결정 단축키는 이 후보 밖의 키만
# 받는다 (WP17.6 검증: q·w를 쓰자 22개 라벨의 자동 키가 바뀌었다). Label Studio를 올리면 이
# 값과 실화면 범례를 다시 확인한다 (change-guide).
LS_AUTO_HOTKEYS = "1234567890qwetasdfgzxcvbyiopjklnm"
# Label Studio 1.23.2 내장 한 글자 단축키 (u: 구간 선택 해제, [·]: 도구 크기). 시간 라벨
# 화면의 나머지 내장 단축키는 조합키·화살표·Enter·Esc·Backspace라 한 글자 키와 겹치지 않는다.
LS_RESERVED_KEYS = frozenset({"u", "[", "]"})
# AI 지적이 없는 라벨 구간의 메모 안내. `labelstudio._ai_controls`의 입력란 자리 글(테마 없는
# 화면)과 `ls_style`이 입력란을 숨기고 대신 보이는 글(테마 화면)이 이 한 문장을 쓴다. 결정 칸은
# 지적 없는 구간에서도 보이고 눌리므로, 반영되지 않는다는 뒷부분을 빼면 안 된다.
AI_NOTE_EMPTY = "이 구간에는 AI 지적이 없다 (결정을 골라도 반영되지 않는다)"


class ModeColors(Contract):
    """한 모드의 화면 색."""

    bg: HexColor = Field(description="바탕")
    panel: HexColor = Field(description="패널")
    panel2: HexColor = Field(description="패널 안 칸·입력란")
    line: HexColor = Field(description="경계선")
    text: HexColor = Field(description="글자")
    muted: HexColor = Field(description="보조 글자·아이콘")
    accent: HexColor = Field(description="강조·주 버튼")
    danger: HexColor = Field(description="원본 경고·삭제")
    warn: HexColor = Field(description="주의")
    ok: HexColor = Field(description="확인")


class ModesTheme(Contract):
    """모드별 화면 색."""

    dark: ModeColors
    light: ModeColors


class BrandTheme(Contract):
    """회사 이름과 로고."""

    name: str = Field(min_length=1)
    logo: str = Field(min_length=1, description="흰색 투명 배경 PNG, 저장소 루트 기준 경로")


class LabelColors(Contract):
    """라벨 색. 표에 없는 ID는 fallback."""

    fallback: HexColor
    verbs: dict[str, HexColor] = Field(description="원시 동작 ID → 색")
    gap: HexColor
    contact_target_kinds: dict[str, HexColor]
    skill: HexColor
    task: HexColor
    substep: HexColor
    event_categories: dict[str, HexColor] = Field(description="events.yaml category → 색")
    state: HexColor
    cvat: dict[str, HexColor] = Field(description="CVAT 라벨 이름 → 색")
    palette: list[HexColor] = Field(
        min_length=1, description="cvat 표에 없는 CVAT 라벨에 이름별로 고정해 주는 색 목록"
    )


class LayoutTheme(Contract):
    """Label Studio 화면 크기 (px)."""

    video_height: int = Field(ge=120, le=1200)
    channel_height: int = Field(ge=30, le=400)


class AiColors(Contract):
    """AI 1차 검수 판정 표시 색 (Label Studio `AI:…` 구간, WP17.6)."""

    wrong: HexColor = Field(description="틀림")
    suspect: HexColor = Field(description="의심")
    unsure: HexColor = Field(description="판단 불가")
    missing: HexColor = Field(description="누락 후보 (표시는 후속, F-AIR-08)")


class AiHotkeys(Contract):
    """AI 지적 결정 단축키 (라벨 구간을 고른 뒤 누른다, 한 글자 키, 조합키 금지).

    Label Studio 자동 키 후보(`LS_AUTO_HOTKEYS`) 밖의 키여야 한다 (`UiTheme._check_hotkeys`).
    """

    apply: str = Field(min_length=1, description="제안 적용")
    dismiss: str = Field(min_length=1, description="AI 지적 기각")


class AiTheme(Contract):
    """AI 1차 검수 표시 모양 (review_ui.yaml `ai`, ADR 0040). 모양만 정하고 판정·적용 규칙은
    바꾸지 않는다."""

    colors: AiColors
    hotkeys: AiHotkeys


class UiTheme(Contract):
    """검수 화면 테마 전체 (review_ui.yaml)."""

    brand: BrandTheme
    default_mode: Mode
    modes: ModesTheme
    colors: LabelColors
    hotkeys: dict[str, str] = Field(description="원시 동작 ID → 오른손 단축키 (왼손은 shift+키)")
    channels: dict[str, HexColor] = Field(description="시계열 채널 이름 → 선 색")
    layout: LayoutTheme
    ai: AiTheme

    @model_validator(mode="after")
    def _check_hotkeys(self) -> UiTheme:
        """단축키는 한 글자 키이고 서로 달라야 하며, Label Studio 내장 한 글자 단축키가 아니다.

        - 동사 단축키(`hotkeys`): 오른손 키. 왼손은 `ls_hotkey`가 shift+를 붙여 만든다 (그래서
          조합키를 직접 쓰지 않는다).
        - AI 결정 단축키(`ai.hotkeys`): 서로 다르고 동사 단축키와 겹치지 않으며, Label Studio
          자동 키 후보(`LS_AUTO_HOTKEYS`) 밖이어야 한다. 후보 안의 키를 쓰면 단축키 없는
          라벨(pull·rotate, 사이 구간, 손 상태)의 자동 키가 한 칸씩 밀려, 검수자가 누르던 키가
          조용히 다른 라벨을 붙인다.
        대소문자는 같은 키로 본다 (Label Studio 단축키 처리와 같다). 실화면 확인 기록은
        `docs/handover/change-guide.md` "검수 화면 AI 표시 실화면 확인" 절이다.
        """
        keys = [k.lower() for k in self.hotkeys.values()]
        if any(len(k) != 1 or k == "+" for k in keys):
            raise ValueError("hotkeys: 키는 한 글자 키여야 합니다 (조합키 금지)")
        if len(keys) != len(set(keys)):
            raise ValueError("hotkeys: 두 동사에 같은 키를 줄 수 없습니다")
        reserved = sorted(set(keys) & LS_RESERVED_KEYS)
        if reserved:
            raise ValueError(f"hotkeys: Label Studio 내장 단축키입니다 {reserved}")
        ai_keys = [self.ai.hotkeys.apply.lower(), self.ai.hotkeys.dismiss.lower()]
        if any(len(k) != 1 or k == "+" for k in ai_keys) or ai_keys[0] == ai_keys[1]:
            raise ValueError("ai.hotkeys: 서로 다른 한 글자 키여야 합니다 (조합키 금지)")
        clash = sorted(k for k in ai_keys if k in keys)
        if clash:
            raise ValueError(f"ai.hotkeys: 동사 단축키와 겹칩니다 {clash}")
        auto = sorted(k for k in ai_keys if k in LS_AUTO_HOTKEYS or k in LS_RESERVED_KEYS)
        if auto:
            raise ValueError(
                f"ai.hotkeys: Label Studio 자동 키 후보·내장 단축키입니다 {auto} (후보 안의 키는 "
                "단축키 없는 라벨의 자동 키를 밀어낸다)"
            )
        return self

    def mode(self, name: Mode) -> ModeColors:
        """모드 이름 → 그 모드의 색."""
        return self.modes.dark if name == "dark" else self.modes.light


def load_ui_theme(root: Path, path: Path | None = None) -> UiTheme:
    """저장소 루트(root)에서 검수 화면 테마를 읽는다 (path가 없으면 `config/review_ui.yaml`).

    예외: 파일이 없으면 OSError, 키·값이 맞지 않으면 pydantic ValidationError.
    부작용 없음 (파일 읽기만).
    """
    source = path if path is not None else root / "config" / "review_ui.yaml"
    data: Any = yaml.safe_load(source.read_text("utf-8"))
    return UiTheme.model_validate(data)


def ls_label_color(theme: UiTheme, ontology: Ontology, name: str) -> str:
    """Label Studio 라벨 이름(`labelstudio.label_names`의 형식) → 색.

    이름 형식: `action.<손>:<동사>`, `gap.<손>:<유형>`, `hand.<손>:<종류>`, `skill:`·`task:`·
    `substep:<ID>`, `event:<유형>`, `state:<속성>=<값>`. 표에 없는 ID는 `colors.fallback`.
    """
    c = theme.colors
    prefix, _, value = name.partition(":")
    group = prefix.split(".", 1)[0]
    match group:
        case "action":
            return c.verbs.get(value, c.fallback)
        case "gap":
            return c.gap
        case "hand":
            return c.contact_target_kinds.get(value, c.fallback)
        case "skill":
            return c.skill
        case "task":
            return c.task
        case "substep":
            return c.substep
        case "event":
            event = ontology.events.get(value)
            return c.event_categories.get(event.category, c.fallback) if event else c.fallback
        case "state":
            return c.state
        case _:
            return c.fallback


def ls_hotkey(theme: UiTheme, name: str) -> str | None:
    """Label Studio 라벨 이름 → 단축키. 원시 동작만 있다 (오른손: 키, 왼손: shift+키).

    원시 동작이 아니거나 표에 없는 동사면 None.
    """
    prefix, _, verb = name.partition(":")
    key = theme.hotkeys.get(verb) if prefix in ("action.right", "action.left") else None
    if key is None:
        return None
    return key if prefix == "action.right" else f"shift+{key}"


def cvat_label_color(theme: UiTheme, name: str) -> str:
    """CVAT 라벨 이름(블러 대상 종류, 객체 클래스, `kp_<골격>`) → 색.

    `colors.cvat` 표에 있으면 그 색, 없으면 이름의 sha256으로 `colors.palette`에서 하나를 고른다
    (실행·컴퓨터와 상관없이 같은 이름은 같은 색).
    """
    if name in theme.colors.cvat:
        return theme.colors.cvat[name]
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return theme.colors.palette[int.from_bytes(digest[:4], "big") % len(theme.colors.palette)]


def logo_data_uri(theme: UiTheme, root: Path) -> str:
    """로고 PNG(`brand.logo`, 저장소 루트 기준) → `data:image/png;base64,...`.

    예외: 파일이 없으면 OSError.
    """
    data = (root / theme.brand.logo).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _rgb(color: str) -> str:
    """#rrggbb → `r g b` (Label Studio의 *-raw 변수 형식)."""
    return " ".join(str(int(color[i : i + 2], 16)) for i in (1, 3, 5))


def _ls_vars(m: ModeColors) -> str:
    """Label Studio 색 변수 덮어쓰기 (한 모드). 변수 이름은 Label Studio 1.23 main.css 기준."""
    pairs = {
        "--color-neutral-background": m.bg,
        "--color-neutral-background-raw": _rgb(m.bg),
        "--color-neutral-background-bold": m.panel,
        "--color-neutral-background-bold-raw": _rgb(m.panel),
        "--color-neutral-surface": m.panel,
        "--color-neutral-surface-inset": m.panel2,
        "--color-neutral-surface-hover": m.panel2,
        "--color-neutral-surface-active": m.panel2,
        "--color-neutral-emphasis": m.panel2,
        "--color-neutral-emphasis-subtle": m.panel2,
        "--color-neutral-border": m.line,
        "--color-neutral-border-subtle": m.line,
        "--color-neutral-border-subtler": m.line,
        "--color-neutral-border-bold": m.muted,
        "--color-neutral-content": m.text,
        "--color-neutral-content-subtle": m.muted,
        "--color-neutral-content-subtler": m.muted,
        "--color-neutral-content-subtlest": m.muted,
        "--color-neutral-icon": m.muted,
        "--color-primary-surface": m.accent,
        "--color-primary-surface-hover": m.accent,
        "--color-primary-content": m.accent,
        "--color-primary-border": m.accent,
        "--color-primary-icon": m.accent,
        "--color-negative-content": m.danger,
        "--color-negative-border": m.danger,
        "--color-dlp-warn": m.warn,
        "--color-dlp-ok": m.ok,
    }
    return ";".join(f"{k}:{v}" for k, v in pairs.items())


def ls_style(theme: UiTheme, logo_uri: str) -> str:
    """Label Studio 화면 CSS (`<Style>` 안에 넣는다).

    두 모드의 색 변수를 덮어쓰고(`data-color-scheme` 값별, 속성이 없으면 화이트), 머리글
    (`className="dlp-brand"`)에 로고를 붙인다. 화이트 모드에서는 흰 로고를 뒤집어 어둡게 보인다.
    logo_uri: `logo_data_uri` 결과.
    """
    dark = _ls_vars(theme.modes.dark)
    light = _ls_vars(theme.modes.light)
    return (
        f":root[data-color-scheme=dark]{{{dark}}}"
        f":root[data-color-scheme=light],:root:not([data-color-scheme]){{{light}}}"
        ".dlp-brand{display:flex;align-items:center;gap:14px;padding:6px 2px 10px;min-width:0;"
        "border-bottom:1px solid var(--color-neutral-border);margin-bottom:8px}"
        f".dlp-brand::before{{content:'';width:118px;height:20px;flex:none;"
        f"background:url({logo_uri}) left center/contain no-repeat}}"
        ":root[data-color-scheme=light] .dlp-brand::before,"
        ":root:not([data-color-scheme]) .dlp-brand::before{filter:invert(1)}"
        ".dlp-brand h1,.dlp-brand h2,.dlp-brand h3,.dlp-brand h4,.dlp-brand h5,.dlp-brand h6"
        "{margin:0;white-space:nowrap}"
        ".dlp-brand>:last-child{color:var(--color-neutral-content-subtle);font-weight:400;"
        "overflow:hidden;text-overflow:ellipsis}"
        ".dlp-event,.dlp-hand{margin-top:10px;padding:10px 12px;border-radius:10px;"
        "border:1px solid var(--color-neutral-border);background:var(--color-neutral-surface)}"
        # AI 1차 검수 (WP17.6): 참고용 칸은 왼쪽 띠로 구별하고, 판정 구간 범례는 누를 수 없게 해
        # 검수자가 AI 라벨로 구간을 그리지 않게 한다. AI 글(요약·메모)의 지우기 단추는 숨긴다
        # (지워도 수집에 영향은 없다)
        ".dlp-ai,.dlp-ai-region{margin-top:10px;padding:10px 12px;border-radius:10px;"
        "border:1px solid var(--color-neutral-border);background:var(--color-neutral-surface);"
        f"border-left:4px solid {theme.ai.colors.suspect}}}"
        ".dlp-ai .lsf-label{pointer-events:none;cursor:default}"
        ".dlp-ai [data-testid=htx-textbox-actions],"
        ".dlp-ai-region [data-testid=htx-textbox-actions]{display:none}"
        # AI 글 입력란은 숨기고(검수자가 쓸 칸이 아니다, 써도 수집하지 않는다) 비었으면 안내만
        # 보인다. 선택자는 Label Studio 1.23 DOM 기준 (업그레이드 때 확인 목록, change-guide)
        ".dlp-ai [data-testid=textarea-form],"
        ".dlp-ai-region [data-testid=textarea-form]{display:none}"
        ".dlp-ai [data-testid=textarea-control]:has([data-testid=textarea-form])::before,"
        ".dlp-ai-region [data-testid=textarea-control]:has([data-testid=textarea-form])::before"
        "{display:block;padding:6px 0;color:var(--color-neutral-content-subtle)}"
        ".dlp-ai [data-testid=textarea-control]:has([data-testid=textarea-form])::before"
        "{content:'AI 1차 검수: 없음'}"
        ".dlp-ai-region [data-testid=textarea-control]:has([data-testid=textarea-form])::before"
        f"{{content:{_css_string(AI_NOTE_EMPTY)}}}"
    )


def _css_string(text: str) -> str:
    """CSS 문자열 값 (작은따옴표로 감싸고 역슬래시·따옴표·줄바꿈을 이스케이프한다)."""
    escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\a ")
    return f"'{escaped}'"


def cvat_tokens_css(theme: UiTheme) -> str:
    """CVAT 테마 색 CSS (생성 파일 `TOKENS_FILE`의 내용).

    `:root`에 처음 모드(`--dlp-default-mode`)를, `:root[data-dlp-mode=<모드>]`에 그 모드의 색 변수
    (`--dlp-<이름>`)를 둔다. 화면 규칙(dlp-cvat.css)과 전환 스크립트(dlp-cvat.js)가
    이 변수를 읽는다.
    """
    lines = [
        "/* 생성 파일: make ui-theme (config/review_ui.yaml에서 만든다). 직접 고치지 않는다. */",
        f":root{{--dlp-default-mode:{theme.default_mode}}}",
    ]
    for name in ("dark", "light"):
        m = theme.mode(name)
        body = ";".join(f"--dlp-{k}:{v}" for k, v in m.model_dump().items())
        lines.append(f":root[data-dlp-mode={name}]{{{body}}}")
    return "\n".join(lines) + "\n"
