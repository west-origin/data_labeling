"""Label Studio 1.23.2 AI 1차 검수 표시 실현 가능성 (`@pytest.mark.services`, WP17.6, ADR 0040).

설계 6.3이 정한 표시 방식(같은 TimeSeries에 라벨 컨트롤 두 개, AI 구간 readonly, 구간별 결정
Choice, 작업 단위 요약 TextArea 예측)을 실제 Label Studio로 확인한다. 이 테스트가 깨지면 표시 방식을
대안(같은 `labels` 컨트롤의 `ai:` 접두사 라벨)으로 바꿔야 한다.

- 기존 작업·사람 주석이 있는 프로젝트에 AI 컨트롤을 더하는 설정 갱신이 된다 (라벨을 더하기만 한다).
- AI 결과(판정 구간 readonly, 메모, 요약, 결정)가 섞인 주석을 받아 저장한다 (readonly 유지).
- AI 영역이 든 주석이 있는 프로젝트에서도 같은 설정 갱신은 통과한다. 설정은 `review.yaml
  ai_review.mode`와 무관하게 같은 XML이라 모드를 바꿔도 갱신이 실패하지 않는다.
- AI 컨트롤을 빼는 갱신은 Label Studio가 거부한다 (400). 그래서 컨트롤을 늘 둔다 (설계 C7).
- 수집 경계: 라벨 컨트롤 결과만 라벨로 읽고(F-AIR-02), 결정은 별칭으로 읽는다.

화면 확인(구간 목록의 잠금 표시와 Backspace로 지워지지 않음, 구간을 고르면 AI 메모·결정이 보임,
결정 단축키 q·w가 동사·내장 단축키와 겹치지 않고 선택됨)은 Playwright로 실화면을 띄워 확인했다
(WP17.6 기록, 스크린샷은 인수인계 문서의 검수 화면 절). 실제 사용자·개인정보는 쓰지 않는다.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from dlp_review.clients import LabelStudioClient, ToolError
from dlp_review.labelstudio import (
    AI_DECISION,
    AI_FLAGS,
    AI_LABELS,
    AI_NOTE,
    AI_SUMMARY,
    FROM_NAME,
    TO_NAME,
    from_ls_results,
    label_config,
    ls_ai_decisions,
    to_ls_results,
)
from dlp_review.ui_theme import UiTheme, load_ui_theme
from dlp_schema import repo_root
from dlp_schema.aireview import AiDecision
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.testing import action_payload, make_label

pytestmark = pytest.mark.services
ROOT = repo_root()


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    """온톨로지 v1."""
    return load_ontology(ROOT / "config" / "ontology" / "v1")


@pytest.fixture(scope="module")
def theme() -> UiTheme:
    """저장소 검수 화면 테마 (AI 색·결정 단축키 포함)."""
    return load_ui_theme(ROOT)


@pytest.fixture
def ls() -> Iterator[LabelStudioClient]:
    """개발 Label Studio (서비스 계정). 닿지 않으면 실패한다 (services 표시 테스트)."""
    client = LabelStudioClient.from_env()
    client.service_user_id()
    yield client
    client.http.close()


def _without_ai(config: str) -> str:
    """AI 컨트롤 두 덩어리를 뺀 설정 (WP17.6 이전 프로젝트의 설정)."""
    start = config.index('  <View className="dlp-ai">')
    end = config.index('  <View visibleWhen="region-selected" className="dlp-event">')
    return config[:start] + config[end:]


def _ai_results(label_id: str) -> list[dict[str, Any]]:
    """AI 예측 결과 묶음: 판정 구간(readonly), 라벨 구간 메모, 작업 요약."""
    flag = {
        "id": "ai-0a1b2c3d", "from_name": AI_FLAGS, "to_name": TO_NAME,
        "type": "timeserieslabels", "readonly": True,
        "value": {"start": 1200, "end": 1800, "instant": False,
                  "timeserieslabels": [AI_LABELS["wrong"]]},
    }  # fmt: skip
    note = {
        "id": label_id, "from_name": AI_NOTE, "to_name": TO_NAME, "type": "textarea",
        "value": {"text": ["[틀림 · 참고] 동사 잡다→밀다 제안 · 근거 00:01.5"]},
    }  # fmt: skip
    summary = {
        "id": "ai-summary", "from_name": AI_SUMMARY, "to_name": TO_NAME, "type": "textarea",
        "value": {"text": ["AI 1차 검수(참고용·사람 검수 아님): 틀림 1 · 판정률 100%"]},
    }  # fmt: skip
    return [flag, note, summary]


def test_ai_controls_are_added_kept_and_never_removed(
    ls: LabelStudioClient, ontology: Ontology, theme: UiTheme
) -> None:
    """기존 주석이 있는 프로젝트에 AI 컨트롤을 더하고, AI 영역이 든 주석이 생긴 뒤에도 같은 설정
    갱신은 통과하며, 컨트롤을 빼는 갱신만 거부된다 (설계 C6·C7).

    정답 근거: 라벨을 더하기만 하는 설정 변경은 Label Studio가 받는다. 주석이 쓰는 컨트롤·라벨을
    지우는 변경은 400 (`LabelStudioClient.ensure_label_config` docstring).
    """
    new = label_config(ontology, theme)
    old = _without_ai(new)
    assert "$" not in new.replace("$video", "").replace("$timeseries", "")  # 새 데이터 변수 없음
    label = make_label(action_payload(), label_id=f"L{uuid.uuid4().hex[:6]}")
    data = {"video": "http://localhost/none.mp4", "timeseries": "http://localhost/none.csv",
            "session_id": "s001", "assignee": "labeler01"}  # fmt: skip
    pid = ls.create_project(f"dlp-test-ai-{uuid.uuid4().hex[:8]}", old)
    try:
        # WP17.6 이전 프로젝트: 프리라벨 예측과 사람 주석
        prelabels = to_ls_results([label])
        tid = ls.create_task(pid, data, prelabels)
        resp = ls.http.post(f"/api/tasks/{tid}/annotations", json={"result": prelabels})
        resp.raise_for_status()
        ls.ensure_label_config(pid, new)  # AI 컨트롤을 더한다
        # AI 예측이 섞인 작업과 결정이 든 사람 주석
        preds = [*to_ls_results([label]), *_ai_results(label.label_id)]
        tid2 = ls.create_task(pid, data, preds)
        assert len(ls.prediction_results(tid2)) == len(preds)
        decision = {"id": label.label_id, "from_name": AI_DECISION, "to_name": TO_NAME,
                    "type": "choices", "value": {"choices": [AiDecision.APPLY.value]}}  # fmt: skip
        resp = ls.http.post(f"/api/tasks/{tid2}/annotations", json={"result": [*preds, decision]})
        resp.raise_for_status()
        stored = resp.json()["result"]
        flags = [r for r in stored if r["from_name"] == AI_FLAGS]
        assert flags and all(r.get("readonly") is True for r in flags)
        # 같은 설정 (모드를 바꿔도 같은 XML): 통과
        ls.ensure_label_config(pid, new)
        ls.ensure_label_config(pid, label_config(ontology, theme))
        # 컨트롤을 빼면 거부 → 그래서 모드와 무관하게 늘 둔다
        with pytest.raises(ToolError, match="400"):
            ls.ensure_label_config(pid, old)
    finally:
        ls.http.delete(f"/api/projects/{pid}/")


def test_collection_reads_only_label_regions_and_ai_decisions(
    ls: LabelStudioClient, ontology: Ontology, theme: UiTheme
) -> None:
    """Label Studio가 돌려준 실제 주석으로 수집 경계를 본다: AI 구간·메모·요약은 라벨이 되지 않고,
    결정은 라벨 구간 id별로 별칭에서 읽힌다 (F-AIR-02)."""
    label = make_label(action_payload(), label_id=f"L{uuid.uuid4().hex[:6]}")
    data = {"video": "http://localhost/none.mp4", "timeseries": "http://localhost/none.csv",
            "session_id": "s001", "assignee": "labeler01"}  # fmt: skip
    pid = ls.create_project(f"dlp-test-ai-{uuid.uuid4().hex[:8]}", label_config(ontology, theme))
    try:
        preds = [*to_ls_results([label]), *_ai_results(label.label_id)]
        tid = ls.create_task(pid, data, preds)
        decision = {"id": label.label_id, "from_name": AI_DECISION, "to_name": TO_NAME,
                    "type": "choices", "value": {"choices": ["dismiss"]}}  # fmt: skip
        resp = ls.http.post(f"/api/tasks/{tid}/annotations", json={"result": [*preds, decision]})
        resp.raise_for_status()
        results: list[dict[str, Any]] = resp.json()["result"]
        [item] = from_ls_results(results, {label.label_id})
        assert item.origin_label_id == label.label_id and item.payload == label.payload
        assert ls_ai_decisions(results) == {label.label_id: AiDecision.DISMISS}
        assert {r["from_name"] for r in results} == {FROM_NAME, AI_FLAGS, AI_NOTE, AI_SUMMARY,
                                                     AI_DECISION}  # fmt: skip
    finally:
        ls.http.delete(f"/api/projects/{pid}/")


def test_feasibility_record_names_this_file() -> None:
    """테마의 단축키 설명이 이 파일을 실화면 확인 기록으로 가리킨다 (경로가 바뀌면 같이 고친다)."""
    text = (ROOT / "config" / "review_ui.yaml").read_text("utf-8")
    assert Path(__file__).relative_to(ROOT).as_posix() in text
