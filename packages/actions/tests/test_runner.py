"""`run_actions` DB 통합 테스트 (WP10, `@pytest.mark.services`, ADR 0015·0024·0026).

실행 중인 PostgreSQL(`make up`)이 필요하다. 테스트마다 일회용 DB에 합성 행동 시나리오(seed 1)의
입력 라벨(손 키포인트·손 상태)만 넣고 돌린다. 정답 행동·사이 구간은 `OracleVlm`에만 준다.
멱등성, 버전 교체, 검수 보호, 입력 변경 재실행, VLM 장애 중단, 트랙 소실, 블러본 확인을 본다.
"""

from __future__ import annotations

import itertools
import os
import uuid
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import sqlalchemy as sa

from dlp_actions.clients import OpenAICompatibleVlm, OracleVlm
from dlp_actions.policy import load_policy
from dlp_actions.runner import run_actions
from dlp_actions.translate import run_translate
from dlp_actions.translate_policy import load_policy as load_translate_policy
from dlp_actions.translators import GlossaryStubTranslator
from dlp_actions.vlm import SegmentRequest, VlmUnavailableError
from dlp_fixtures.actions import generate_action_scenario
from dlp_fixtures.video import write_video
from dlp_media.storage import LocalStore, sha256_file
from dlp_privacy.runner import RenderNotCurrentError
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    get_labels,
    insert_labels,
    insert_session,
    record_review,
    register_ontology,
    set_privacy_state,
)
from dlp_schema.episode import current_labels, retractions
from dlp_schema.labels import (
    ActionPayload,
    DescriptionPayload,
    GapPayload,
    HandStatePayload,
    KeypointTrackPayload,
    LabelRecord,
    Provenance,
    Source,
    VerificationState,
)
from dlp_schema.ontology import load_ontology
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME, make_session

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]
# 테스트 세션 ID
SID = "act-0001"
# 입력·정답을 주는 합성 행동 시나리오 (seed 1)
SCENARIO = generate_action_scenario(1, session_id=SID)


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    """일회용 DB 엔진: 마이그레이션·온톨로지 v1·세션과 입력 라벨(손 키포인트·손 상태)만 넣는다.

    테스트가 끝나면 DB를 강제로 지운다.
    """
    url = sa.make_url(
        os.environ.get(
            "DLP_DATABASE_URL", "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"
        )
    )
    name = f"dlp_test_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    test_url = url.set(database=name).render_as_string(hide_password=False)
    upgrade(test_url)
    engine = sa.create_engine(test_url)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        insert_session(conn, make_session(SID, duration_ms=SCENARIO.duration_ms))
        # 입력: 손 키포인트와 손 상태만 (정답 행동·사이 구간은 넣지 않는다)
        insert_labels(
            conn, [x for x in SCENARIO.labels if x.kind in ("keypoint_track", "hand_state")]
        )
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def _timeline(conn: sa.Connection) -> list[LabelRecord]:
    """현재 action·gap 라벨을 시작 시각 순으로."""
    labels = current_labels(get_labels(conn, SID, kinds=["action", "gap"]))
    return sorted(labels, key=lambda x: x.t_start_ms)


def test_run_is_idempotent_and_new_version_replaces_old(pg: sa.Engine) -> None:
    """같은 입력·정책이면 두 번째 실행은 건너뛰고, 결과가 0~끝을 빈틈없이 덮는지 검증한다.
    정책(merge_ms)이 바뀌면 이전 버전의 행동·사이 구간·설명을 모두 삭제 표시하고 새 버전으로 다시
    만든다.
    """
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    truth = [x for x in SCENARIO.labels if isinstance(x.payload, ActionPayload | GapPayload)]
    with pg.begin() as conn:
        first = run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        again = run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        timeline = _timeline(conn)
    counts = first.hands["right"]
    assert counts["actions"] == len(SCENARIO.actions) and counts["unknown_fallbacks"] == 0
    assert again.skipped == ["right"] and not again.hands
    assert timeline[0].t_start_ms == 0 and timeline[-1].t_end_ms == SCENARIO.duration_ms
    assert all(a.t_end_ms == b.t_start_ms for a, b in itertools.pairwise(timeline))
    assert all(x.provenance.source is Source.MODEL for x in timeline)

    # 정책이 바뀌면 이전 버전 레코드는 삭제 표시되고 새 버전으로 다시 만든다
    changed = policy.model_copy(
        update={"boundaries": policy.boundaries.model_copy(update={"merge_ms": 90})}
    )
    with pg.begin() as conn:
        second = run_actions(conn, SID, OracleVlm(truth), ontology, changed, FIXED_TIME)
        timeline2 = _timeline(conn)
        descriptions = current_labels(get_labels(conn, SID, kinds=["description"]))
    assert second.version != first.version
    assert second.retracted == len(timeline) + counts["actions"]  # 행동·사이 구간 + 설명
    assert {x.provenance.model_version for x in timeline2} == {second.version}
    assert {x.provenance.model_version for x in descriptions} == {second.version}


def test_rerun_after_reviewer_deleted_everything_is_skipped(pg: sa.Engine) -> None:
    """감사 회귀 (ADR 0015): 현재 라벨이 아니라 이력으로 멱등을 판단한다 (ID 충돌·되살림 없음).

    검수자가 모두 지운 뒤 같은 버전으로 다시 돌려도 되살리지 않고 건너뛴다.
    """
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    truth = [x for x in SCENARIO.labels if isinstance(x.payload, ActionPayload | GapPayload)]
    with pg.begin() as conn:
        run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        ours = current_labels(get_labels(conn, SID, kinds=["action", "gap", "description"]))
        insert_labels(
            conn,
            [
                x.model_copy(
                    update={
                        "label_id": f"{x.label_id}:rev",
                        "parent_label_id": x.label_id,
                        "retracted": True,
                        "provenance": Provenance(source=Source.HUMAN),
                        "confidence": None,
                    }
                )
                for x in ours
            ],
        )
        again = run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        assert current_labels(get_labels(conn, SID, kinds=["action", "gap"])) == []
    assert again.skipped == ["right"]


def test_new_version_keeps_reviewed_labels_and_fills_gaplessly(pg: sa.Engine) -> None:
    """새 버전이 승인·표본 검증된 라벨을 남기고, 그와 겹치는 새 결과를 버린 뒤 빈 곳을 미상으로 채워
    타임라인에 공백·겹침이 없는지 검증한다.
    """
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    truth = [x for x in SCENARIO.labels if isinstance(x.payload, ActionPayload | GapPayload)]
    with pg.begin() as conn:
        run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        timeline = _timeline(conn)
        approved = [x for x in timeline if isinstance(x.payload, ActionPayload)][1]
        sampled = timeline[0]
        record_review(conn, approved.label_id, VerificationState.HUMAN_APPROVED, "r1", FIXED_TIME)
        record_review(conn, sampled.label_id, VerificationState.SAMPLE_VERIFIED, "r1", FIXED_TIME)
    # 경계가 달라지는 새 버전 (병합 간격을 크게 바꾼다)
    changed = policy.model_copy(
        update={"boundaries": policy.boundaries.model_copy(update={"merge_ms": 600})}
    )
    with pg.begin() as conn:
        second = run_actions(conn, SID, OracleVlm(truth), ontology, changed, FIXED_TIME)
        timeline2 = _timeline(conn)
    ids = {x.label_id for x in timeline2}
    assert {approved.label_id, sampled.label_id} <= ids  # 검수한 라벨은 남는다
    assert second.hands["right"]["kept_reviewed"] == 2
    # 공백도 겹침도 없다
    assert timeline2[0].t_start_ms == 0 and timeline2[-1].t_end_ms == SCENARIO.duration_ms
    assert all(a.t_end_ms == b.t_start_ms for a, b in itertools.pairwise(timeline2))
    others = {x.provenance.model_version for x in timeline2} - {approved.provenance.model_version}
    assert others <= {second.version}


def test_input_change_reruns_and_reviewed_description_protects_its_action(pg: sa.Engine) -> None:
    """감사 회귀: 입력(손 상태)이 바뀌면 다시 만들고, 설명이 검수된 행동은 지우지 않는다.

    대상을 모르는 접촉 표시 ID(unresolved)는 VLM 대상 후보에 넣지 않는다.
    """
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    truth = [x for x in SCENARIO.labels if isinstance(x.payload, ActionPayload | GapPayload)]
    with pg.begin() as conn:
        first = run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        timeline = _timeline(conn)
        action = [x for x in timeline if isinstance(x.payload, ActionPayload)][1]
        assert isinstance(action.payload, ActionPayload)
        [desc] = [
            x
            for x in current_labels(get_labels(conn, SID, kinds=["description"]))
            if isinstance(x.payload, DescriptionPayload)
            and x.payload.segment_id == action.payload.action_id
        ]
        # 검수자가 설명만 고쳤다 (행동 레코드는 미검수 그대로)
        assert isinstance(desc.payload, DescriptionPayload)
        insert_labels(
            conn,
            [
                desc.model_copy(
                    update={
                        "label_id": f"{desc.label_id}:edit",
                        "parent_label_id": desc.label_id,
                        "provenance": Provenance(source=Source.HUMAN),
                        "confidence": None,
                        "payload": desc.payload.model_copy(update={"text": "걸레로 닦는다"}),
                    }
                )
            ],
        )
        # 입력 변경: 대상을 모르는 접촉 하나를 더한다 (정책·VLM은 같다)
        state = next(
            x for x in SCENARIO.labels
            if isinstance(x.payload, HandStatePayload) and x.payload.contact_target_kind != "none"
        )  # fmt: skip
        assert isinstance(state.payload, HandStatePayload)
        insert_labels(
            conn,
            [
                state.model_copy(
                    update={
                        "label_id": f"{state.label_id}-unresolved",
                        "payload": state.payload.model_copy(
                            update={
                                "contact_target_kind": "object",
                                "target_id": "unresolved",
                                "grasp_type": None,
                            }
                        ),
                    }
                )
            ],
        )
        seen: list[tuple[str, ...]] = []

        class Spy(OracleVlm):
            """요청마다 받은 대상 후보(`entities`)를 기록하는 오라클 VLM."""

            def complete(self, request: SegmentRequest, prompt: str, schema: dict[str, Any]) -> str:
                """대상 후보를 `seen`에 남기고 오라클 답을 그대로 돌려준다."""
                seen.append(request.entities)
                return super().complete(request, prompt, schema)

        second = run_actions(conn, SID, Spy(truth), ontology, policy, FIXED_TIME)
        timeline2 = _timeline(conn)
        descriptions = current_labels(get_labels(conn, SID, kinds=["description"]))
    assert second.version != first.version and second.hands  # 입력이 바뀌어 다시 돌았다
    assert seen and all("unresolved" not in e for e in seen)
    assert action.label_id in {x.label_id for x in timeline2}  # 설명이 검수된 행동은 남는다
    assert any(
        isinstance(x.payload, DescriptionPayload)
        and x.payload.segment_id == action.payload.action_id
        and x.label_id == f"{desc.label_id}:edit"
        for x in descriptions
    )
    assert all(a.t_end_ms == b.t_start_ms for a, b in itertools.pairwise(timeline2))


def test_actions_rerun_drops_english_siblings_of_dropped_descriptions(pg: sa.Engine) -> None:
    """설명의 영어 번역(ADR 0043)과 행동 단계 재실행.

    정답: 번역 뒤 설명마다 영어 형제가 있다 (원문·번역 모두 현재 라벨). 정책이 바뀌어 행동을 다시
    만들면 검수 전 행동·설명과 함께 그 영어 번역도 지운다 (삭제 수 = 행동·공백 + 설명 + 번역).
    검수자가 승인한 번역이 있는 행동은 지우지 않는다. 다시 번역하면 새 설명만 번역한다.
    """
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    tpolicy = load_translate_policy(ROOT)
    truth = [x for x in SCENARIO.labels if isinstance(x.payload, ActionPayload | GapPayload)]
    with pg.begin() as conn:
        first = run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
    with pg.begin() as conn:
        t1 = run_translate(conn, SID, GlossaryStubTranslator(), ontology, tpolicy, FIXED_TIME)
        descriptions = current_labels(get_labels(conn, SID, kinds=["description"]))
        timeline = _timeline(conn)
    n_actions = first.hands["right"]["actions"]
    assert t1.translated == n_actions and t1.pending == []
    english = [
        x for x in descriptions
        if isinstance(x.payload, DescriptionPayload) and x.payload.language == "en"
    ]  # fmt: skip
    assert len(english) == n_actions and len(descriptions) == 2 * n_actions
    # 한 행동의 영어 번역을 검수자가 승인한다 → 그 행동은 보호된다
    kept = english[0]
    assert isinstance(kept.payload, DescriptionPayload)
    with pg.begin() as conn:
        record_review(conn, kept.label_id, VerificationState.HUMAN_APPROVED, "r1", FIXED_TIME)
    changed = policy.model_copy(
        update={"boundaries": policy.boundaries.model_copy(update={"merge_ms": 90})}
    )
    with pg.begin() as conn:
        second = run_actions(conn, SID, OracleVlm(truth), ontology, changed, FIXED_TIME)
        after = current_labels(get_labels(conn, SID, kinds=["description"]))
        timeline2 = _timeline(conn)
    # 보호된 행동 1개와 그 설명·번역을 뺀 나머지가 지워졌다
    assert second.retracted == (len(timeline) - 1) + 2 * (n_actions - 1)
    assert kept.payload.segment_id in {
        x.payload.action_id for x in timeline2 if isinstance(x.payload, ActionPayload)
    }
    stale_en = [
        x for x in after
        if isinstance(x.payload, DescriptionPayload) and x.payload.language == "en"
    ]  # fmt: skip
    assert [x.label_id for x in stale_en] == [kept.label_id]  # 검수 전 번역은 행동과 함께 지워졌다
    with pg.begin() as conn:
        t2 = run_translate(conn, SID, GlossaryStubTranslator(), ontology, tpolicy, FIXED_TIME)
        final = current_labels(get_labels(conn, SID, kinds=["description"]))
    ko = [
        x for x in final if isinstance(x.payload, DescriptionPayload) and x.payload.language == "ko"
    ]
    en = [
        x for x in final if isinstance(x.payload, DescriptionPayload) and x.payload.language == "en"
    ]
    assert t2.retracted == 0 and len(en) == len(ko)
    assert all(a.t_end_ms == b.t_start_ms for a, b in itertools.pairwise(timeline2))


def test_vlm_outage_aborts_the_session_so_a_rerun_redoes_it(pg: sa.Engine) -> None:
    """감사 회귀 (4차): VLM 서버가 백오프 후에도 응답하지 않으면 미상으로 채워 쓰지 않고 실행을
    멈춘다 (트랜잭션이 되돌려진다). 같은 서버·모델이 복구되면 다시 실행해 제대로 만든다.
    """
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    fast = policy.model_copy(
        update={"vlm": policy.vlm.model_copy(update={"unavailable_backoff_s": (0.0, 0.0)})}
    )
    down = OpenAICompatibleVlm(
        "http://vlm", policy.vlm.model, frames=1, max_side=64, timeout_s=1,
        transport=httpx.MockTransport(lambda req: httpx.Response(503)),
    )  # fmt: skip
    with pytest.raises(VlmUnavailableError), pg.begin() as conn:
        run_actions(conn, SID, down, ontology, fast, FIXED_TIME)
    with pg.begin() as conn:
        assert _timeline(conn) == []  # 아무것도 쓰지 않았다
    truth = [x for x in SCENARIO.labels if isinstance(x.payload, ActionPayload | GapPayload)]
    up = OracleVlm(truth)
    up.version = down.version  # 같은 서버·모델이 복구됐다 (모델 버전이 같다)
    with pg.begin() as conn:
        again = run_actions(conn, SID, up, ontology, fast, FIXED_TIME)
        timeline = _timeline(conn)
    assert not again.skipped and again.hands["right"]["unknown_fallbacks"] == 0
    assert sum(isinstance(x.payload, ActionPayload) for x in timeline) == len(SCENARIO.actions)


def test_hand_without_track_loses_only_unreviewed_actions(pg: sa.Engine) -> None:
    """감사 회귀 (4차): 손 트랙이 사라지면(손 모델 버전 변경 등) 그 손의 검수 전 행동·공백·설명을
    지운다. 검수된 행동은 남긴다. 다시 돌려도 더 지우지 않는다.
    """
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    truth = [x for x in SCENARIO.labels if isinstance(x.payload, ActionPayload | GapPayload)]
    with pg.begin() as conn:
        run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        timeline = _timeline(conn)
        approved = next(x for x in timeline if isinstance(x.payload, ActionPayload))
        record_review(conn, approved.label_id, VerificationState.HUMAN_APPROVED, "r1", FIXED_TIME)
        tracks = current_labels(get_labels(conn, SID, kinds=["keypoint_track"]))
        # 손 모델 버전이 바뀌어 이전 트랙이 지워지고 새 버전은 이 손을 찾지 못했다
        insert_labels(conn, retractions(tracks, "hands-v2", FIXED_TIME))
        gone = run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        left = current_labels(get_labels(conn, SID, kinds=["action", "gap", "description"]))
        again = run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
    assert gone.retracted == gone.hands["right"]["retracted_without_track"] > 0
    assert [x.label_id for x in left if x.kind != "description"] == [approved.label_id]
    assert isinstance(approved.payload, ActionPayload)
    assert all(
        isinstance(x.payload, DescriptionPayload)
        and x.payload.segment_id == approved.payload.action_id
        for x in left
        if x.kind == "description"
    )
    assert again.retracted == 0 and not again.hands


def test_vlm_refuses_blurred_video_not_rendered_from_current_approval(
    pg: sa.Engine, tmp_path: Path
) -> None:
    """회귀(감사 4-2): VLM에는 지금 승인된 블러 라벨로 렌더한 블러본만 보낸다. 승인이 풀렸거나
    렌더 기록이 없거나 무효이면 프레임을 읽기 전에 멈춘다.
    """
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    truth = [x for x in SCENARIO.labels if isinstance(x.payload, ActionPayload | GapPayload)]
    labeling = LocalStore(tmp_path / "store", "dlp-labeling")
    video = tmp_path / "b.mp4"
    write_video(
        video, ((t, np.zeros((32, 48, 3), np.uint8)) for t in (0, 33, 66)), width=48, height=32
    )
    labeling.put_file(f"sessions/{SID}/blurred/bodycam.mp4", video, sha256_file(video))
    with pg.begin() as conn:
        with pytest.raises(RenderNotCurrentError, match="승인 상태"):
            run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME, labeling)
        set_privacy_state(conn, SID, PrivacyState.APPROVED)
        with pytest.raises(RenderNotCurrentError, match="렌더 기록"):
            run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME, labeling)
        assert not current_labels(get_labels(conn, SID, kinds=["action"]))


def test_same_hand_tracks_follow_the_prelabel_track_rank(pg: sa.Engine) -> None:
    """F-PAR-04 회귀: 같은 손 hand21 트랙이 여럿이면 프리라벨 접촉 단계와 같은 순위
    (`dlp_schema.episode.track_rank`: 사람·검수 → 최신 → 긴 구간 → 라벨 ID)로 하나를 고른다.

    시나리오: 정답 손 트랙(사람 출처)을 지우고, 같은 오른손 모델 트랙 둘을 넣는다. A는 정답과 같은
    움직임(hands-v1), B는 손이 첫 프레임 자리에 멈춘 미끼(hands-v2, 하루 늦게 만듦, 라벨 ID가 뒤).
    1) 검수가 없으면 더 새로 만든 B를 쓴다. 2) 검수자가 A를 승인(HUMAN_APPROVED, 라벨 ID는
    그대로)하면 고르는 트랙이 A로 바뀌므로 입력 해시(모델 버전)도 바뀌어 다시 돌고, 정답 움직임으로
    행동을 모두 찾는다. 3) 다시 돌리면 건너뛴다.
    예전에는 조회 순서(시작 시각, 라벨 ID)상 마지막 트랙(B)을 쓰고, 입력 해시가 모든 트랙의 ID라
    승인해도 다시 돌지 않았다.
    """
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    truth = [x for x in SCENARIO.labels if isinstance(x.payload, ActionPayload | GapPayload)]
    [track] = [x for x in SCENARIO.labels if isinstance(x.payload, KeypointTrackPayload)]
    assert isinstance(track.payload, KeypointTrackPayload)
    first_points = track.payload.keyframes[0].points
    still = track.payload.model_copy(
        update={
            "keyframes": tuple(
                f.model_copy(update={"points": first_points}) for f in track.payload.keyframes
            )
        }
    )
    a = track.model_copy(
        update={
            "label_id": f"{SID}-bodycam-hands-a",
            "provenance": Provenance(source=Source.MODEL, model_version="hands-v1"),
            "confidence": 0.9,
        }
    )
    b = track.model_copy(
        update={
            "label_id": f"{SID}-bodycam-hands-b",
            "provenance": Provenance(source=Source.MODEL, model_version="hands-v2"),
            "confidence": 0.9,
            "created_at": FIXED_TIME + timedelta(days=1),
            "payload": still,
        }
    )
    with pg.begin() as conn:
        insert_labels(conn, [*retractions([track], "hands-v1", FIXED_TIME), a, b])
        first = run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        record_review(conn, a.label_id, VerificationState.HUMAN_APPROVED, "r1", FIXED_TIME)
        second = run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        third = run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        timeline = _timeline(conn)
    # 1) 승인 전: 미끼(B)를 써서 손목 속도 경계가 없다 (정답 행동을 모두 찾지 못한다)
    assert first.hands["right"]["actions"] != len(SCENARIO.actions)
    # 2) 승인 후: 고른 트랙이 바뀌어 버전이 바뀌고, 승인된 A의 움직임으로 다시 만든다
    assert second.version != first.version and not second.skipped
    assert second.hands["right"]["actions"] == len(SCENARIO.actions)
    assert second.hands["right"]["unknown_fallbacks"] == 0
    assert {x.provenance.model_version for x in timeline} == {second.version}
    # 3) 같은 입력이면 건너뛴다
    assert third.skipped == ["right"] and third.version == second.version
