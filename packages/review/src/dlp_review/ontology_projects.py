"""검수 화면 사전을 온톨로지 내용별로 격리한다. 기존 프로젝트를 덮어쓰지 않는다.

공개 이름:
- `ontology_project`: 프로젝트 이름 `<기본 이름>-v<온톨로지 버전>-<사전 해시 앞 12자>`.
- `is_ontology_project`: 이름이 이 플랫폼이 만든 그 기본 이름의 프로젝트인지 (`ontology_project`
  모양 또는 격리 전 이름 그대로). 고아 찾기(`dlp_review.orphans`)가 다른 시스템의 프로젝트를
  건드리지 않게 쓴다 (F-REV-15).
"""

import re

from dlp_schema.ontology import Ontology, ontology_digest

# 프로젝트 이름에 넣는 사전 해시 글자 수
_TAG_CHARS = 12


def ontology_project(base: str, ontology: Ontology) -> str:
    """버전과 사전 해시를 프로젝트 이름에 넣어 draft 덧붙이기도 다른 화면으로 분리한다."""
    tag = ontology_digest(ontology)[:_TAG_CHARS]
    return f"{base}-v{ontology.version}-{tag}"


def is_ontology_project(name: str, base: str) -> bool:
    """name이 `ontology_project(base, …)` 모양이거나 격리 전 이름 base 그대로면 True.

    온톨로지 버전은 `주.부.수`(`dlp_schema.common.SemVer`)다. `<base>-demo`처럼 뒤에 다른 글자가
    붙은 이름은 다른 시스템의 것으로 본다.
    """
    pattern = rf"{re.escape(base)}(-v\d+\.\d+\.\d+-[0-9a-f]{{{_TAG_CHARS}}})?"
    return re.fullmatch(pattern, name) is not None
