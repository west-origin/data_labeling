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
- `ModeColors`·`ModesTheme`·`BrandTheme`·`LabelColors`·`LayoutTheme`·`UiTheme`:
  review_ui.yaml 절별 모델.
- `TOKENS_FILE`: CVAT 테마 색 생성 파일 경로 (저장소 루트 기준).
- `load_ui_theme`: 저장소 루트에서 테마를 읽어 검증한다.
- `ls_label_color`·`ls_hotkey`: Label Studio 라벨 이름 → 색·단축키.
- `cvat_label_color`: CVAT 라벨 이름 → 색.
- `logo_data_uri`: 로고 파일 → data URI (Label Studio CSS에 넣는다).
- `ls_style`: Label Studio 화면 CSS.
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


class UiTheme(Contract):
    """검수 화면 테마 전체 (review_ui.yaml)."""

    brand: BrandTheme
    default_mode: Mode
    modes: ModesTheme
    colors: LabelColors
    hotkeys: dict[str, str] = Field(description="원시 동작 ID → 오른손 단축키 (왼손은 shift+키)")
    channels: dict[str, HexColor] = Field(description="시계열 채널 이름 → 선 색")
    layout: LayoutTheme

    @model_validator(mode="after")
    def _check_hotkeys(self) -> UiTheme:
        """단축키는 비어 있지 않고 서로 달라야 하며, 조합키(+)를 직접 쓰지 않는다.

        왼손 단축키는 `ls_hotkey`가 shift+를 붙여 만든다.
        """
        keys = list(self.hotkeys.values())
        if any(not k or "+" in k for k in keys):
            raise ValueError("hotkeys: 키는 비어 있지 않은 한 글자 키여야 합니다 (조합키 + 금지)")
        if len(keys) != len(set(keys)):
            raise ValueError("hotkeys: 두 동사에 같은 키를 줄 수 없습니다")
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
        ".dlp-event{margin-top:10px;padding:10px 12px;border-radius:10px;"
        "border:1px solid var(--color-neutral-border);background:var(--color-neutral-surface)}"
    )


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
