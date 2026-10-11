"""검수 화면 사전을 온톨로지 내용별로 격리한다. 기존 프로젝트를 덮어쓰지 않는다."""

import hashlib
import json

from dlp_schema.ontology import Ontology


def ontology_project(base: str, ontology: Ontology) -> str:
    """버전과 사전 해시를 프로젝트 이름에 넣어 draft 덧붙이기도 다른 화면으로 분리한다."""
    content = json.dumps(ontology.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    tag = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{base}-v{ontology.version}-{tag}"
