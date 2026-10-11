"""검수 화면 테마 단위 테스트 (ADR 0038). 서비스 없이 돈다.

- review_ui.yaml이 검증을 통과하고, CVAT 테마 색 생성 파일이 설정과 같다.
- 테마를 입힌 Label Studio 설정: 라벨 이름·컨트롤 이름은 테마 없을 때와 같고(왕복 형식 불변),
  라벨마다 색이 붙고, 원시 동작 단축키가 손별로 겹치지 않는다.
- CVAT 라벨 색: 표가 먼저, 표에 없으면 이름으로 고정된 색.
- CVAT 테마 이미지 파일과 스크립트의 프로젝트 이름 접두사가 코드 상수와 맞다.
정답은 저장소의 config/review_ui.yaml·온톨로지 v1과 코드 상수다.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from pydantic import ValidationError

from dlp_review.cvat import label_spec
from dlp_review.labelstudio import FROM_NAME, label_config, label_names
from dlp_review.tasks import PRIVACY_PROJECT, SPATIAL_PROJECT
from dlp_review.ui_theme import (
    TOKENS_FILE,
    UiTheme,
    cvat_label_color,
    cvat_tokens_css,
    load_ui_theme,
    logo_data_uri,
    ls_hotkey,
)
from dlp_schema import repo_root
from dlp_schema.ontology import Ontology, VerbLevel, load_ontology

ROOT = repo_root()
THEME_DIR = ROOT / "services" / "review-theme"


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    """온톨로지 v1 (라벨 이름·이벤트 분류의 정답)."""
    return load_ontology(ROOT / "config" / "ontology" / "v1")


@pytest.fixture(scope="module")
def theme() -> UiTheme:
    """저장소의 검수 화면 테마."""
    return load_ui_theme(ROOT)


def _labels(config: str) -> list[ET.Element]:
    """설정 XML의 TimeSeriesLabels 안 Label 요소들."""
    root = ET.fromstring(config)
    control = root.find(f".//TimeSeriesLabels[@name='{FROM_NAME}']")
    assert control is not None
    return list(control.iter("Label"))


def test_tokens_file_matches_config(theme: UiTheme) -> None:
    """생성 파일이 설정과 같다 (다르면 `make ui-theme`). 두 모드의 색 변수가 모두 있다."""
    text = (ROOT / TOKENS_FILE).read_text("utf-8")
    assert text == cvat_tokens_css(theme)
    for mode in ("dark", "light"):
        assert f":root[data-dlp-mode={mode}]" in text
    assert f"--dlp-default-mode:{theme.default_mode}" in text


def test_themed_label_config_keeps_names_and_adds_colors_hotkeys(
    ontology: Ontology, theme: UiTheme
) -> None:
    """테마는 모양만 바꾼다: 라벨 이름 목록은 그대로이고, 라벨마다 색, 원시 동작에는 손별 단축키."""
    themed = label_config(ontology, theme, logo_uri=logo_data_uri(theme, ROOT))
    plain = label_config(ontology)
    assert [e.get("value") for e in _labels(themed)] == label_names(ontology)
    assert [e.get("value") for e in _labels(plain)] == label_names(ontology)
    # 테마 없는 설정에는 색·단축키·CSS가 없다 (기존 동작)
    assert all(e.get("background") is None and e.get("hotkey") is None for e in _labels(plain))
    assert "<Style>" not in plain

    labels = _labels(themed)
    assert all(re.fullmatch(r"#[0-9a-fA-F]{6}", e.get("background") or "") for e in labels)
    keys = [e.get("hotkey") for e in labels if e.get("hotkey")]
    assert len(keys) == len(set(keys)), "한 컨트롤 안에서 단축키가 겹치면 안 된다"
    by_name = {e.get("value"): e for e in labels}
    for verb, key in theme.hotkeys.items():
        assert by_name[f"action.right:{verb}"].get("hotkey") == key
        assert by_name[f"action.left:{verb}"].get("hotkey") == f"shift+{key}"
    # 같은 동사는 양손이 같은 색
    for verb in theme.colors.verbs:
        if f"action.right:{verb}" in by_name:
            assert (
                by_name[f"action.right:{verb}"].get("background")
                == by_name[f"action.left:{verb}"].get("background")
                == theme.colors.verbs[verb]
            )
    # 이벤트 색은 분류(category)별
    slip = by_name["event:slip"].get("background")
    assert slip == theme.colors.event_categories[ontology.events["slip"].category]
    # 두 모드 색과 로고가 CSS에 들어 있다
    style = ET.fromstring(themed).find("Style")
    assert style is not None and style.text is not None
    assert "data-color-scheme=dark" in style.text and "data-color-scheme=light" in style.text
    assert "data:image/png;base64," in style.text


def test_hotkeys_cover_only_primitive_verbs(ontology: Ontology, theme: UiTheme) -> None:
    """단축키 동사는 원시 동작이어야 한다. 오타 동사는 설정을 만들 때 거부한다."""
    primitives = {v for v, t in ontology.verbs.items() if t.level is VerbLevel.PRIMITIVE}
    assert set(theme.hotkeys) <= primitives
    assert ls_hotkey(theme, "skill:wipe") is None
    assert ls_hotkey(theme, "gap.right:idle") is None
    bad = theme.model_copy(update={"hotkeys": {**theme.hotkeys, "grap": "0"}})
    with pytest.raises(ValueError, match="grap"):
        label_config(ontology, bad)


def test_duplicate_or_combo_hotkeys_rejected(theme: UiTheme) -> None:
    """두 동사에 같은 키, 또는 조합키(+)를 직접 쓰면 설정 검증에서 거부한다."""
    data = theme.model_dump()
    data["hotkeys"] = {"grasp": "1", "release": "1"}
    with pytest.raises(ValidationError, match="같은 키"):
        UiTheme.model_validate(data)
    data["hotkeys"] = {"grasp": "shift+1"}
    with pytest.raises(ValidationError, match="조합키"):
        UiTheme.model_validate(data)
    data["hotkeys"] = {}
    data["modes"]["dark"]["bg"] = "black"
    with pytest.raises(ValidationError):
        UiTheme.model_validate(data)


def test_cvat_label_colors(theme: UiTheme) -> None:
    """표에 있는 라벨은 그 색, 없는 라벨은 팔레트에서 이름으로 고정된 색. label_spec에 실린다."""
    assert cvat_label_color(theme, "face") == theme.colors.cvat["face"]
    other = cvat_label_color(theme, "bed_nameplate")
    assert other in theme.colors.palette
    assert cvat_label_color(theme, "bed_nameplate") == other  # 같은 이름은 항상 같은 색
    names = ["face", "bed_nameplate"]
    spec = label_spec(names, {n: cvat_label_color(theme, n) for n in names})
    assert [s["color"] for s in spec] == [theme.colors.cvat["face"], other]
    assert all("color" not in s for s in label_spec(names))  # 색을 안 주면 CVAT가 고른다


def test_cvat_theme_image_files() -> None:
    """테마 이미지가 복사하는 파일이 있고, 스크립트의 프로젝트 이름 접두사가 코드 상수와 같다."""
    dockerfile = (THEME_DIR / "cvat" / "Dockerfile").read_text("utf-8")
    for src in re.findall(r"^COPY (.+) /usr/share/nginx/html/dlp/", dockerfile, re.M):
        for path in src.split():
            assert (THEME_DIR / path).is_file(), path
    script = (THEME_DIR / "cvat" / "dlp-cvat.js").read_text("utf-8")
    prefixes = re.findall(r"prefix: '([^']+)'", script)
    assert prefixes == [PRIVACY_PROJECT, SPATIAL_PROJECT]
    logo = THEME_DIR / "logo.png"
    assert logo.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert Path(load_ui_theme(ROOT).brand.logo) == logo.relative_to(ROOT)
