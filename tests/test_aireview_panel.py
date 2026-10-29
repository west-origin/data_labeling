"""판정단 종단 회귀: 두 판정 모델 실행 → 검수 운영의 불일치 사유 (ADR 0063, 서비스 불필요).

합성 세션(승인된 블러본, SQLite 파일 DB)에서 판정 실행기(`run_from_db`)를 판정단 모델마다 돌린다.

- 모델 A(`backend.model`): 가짜 OpenAI 호환 서버(`FakeOpenAIServer`, 기본 답 = 모두 ok)와
  `OpenAICompatibleJudge`.
- 모델 B(판정단 `vllm-sr/d3-turbo`, `select_model`): `OracleJudge`. 정답에서 행동 하나의 동사만
  바꿔 두어 그 행동만 wrong으로 답한다.

검수 운영(`load_ai_view`, `ai_flagged_spans`, `escalated_ids`)이 두 버전을 허용하면 그 행동에만
AI_DISAGREEMENT가 붙고 표본 제외 대상이 된다. 꺼짐·한 버전만 허용이면 붙지 않는다. 두 모델이
ok로 같은 라벨은 표본 제외도 사유도 없다 (사람 검수를 줄이지 않는다, ADR 0040 I9).
검수 화면용 표시 프로필(지금 렌더 해시·세션 온톨로지)도 그 행동에만 모델별 의견("AI 판정단
의견이 갈림: <A> 맞음 / <B> 틀림")을 낸다.

정답 근거: 바꾼 정답 라벨 ID, 가짜 서버의 기본 답(ok), 판정 버전 문자열.
"""

from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa

from dlp_aireview.backend import EndpointPool, OpenAICompatibleJudge, PoolMember
from dlp_aireview.oracle import OracleJudge
from dlp_aireview.policy import AiReviewPolicy, load_policy, select_model
from dlp_aireview.runner import RunOptions, run_from_db
from dlp_aireview.testing import FakeOpenAIServer
from dlp_fixtures.aireview import aireview_scene, scene_frame
from dlp_fixtures.video import write_video
from dlp_media.storage import LocalStore, blurred_key, sha256_file
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.rendering import RenderMeta, expected_render_hash, write_render_meta
from dlp_review.ai_display import (
    AiDisplayConfig,
    ai_flagged_spans,
    escalated_ids,
    load_ai_view,
    split_text,
)
from dlp_review.ops.policy import AiReviewDisplayPolicy
from dlp_schema import load_config, repo_root
from dlp_schema.aireview import AiVerdictValue
from dlp_schema.db.aireview_repository import list_ai_units
from dlp_schema.db.ontology_repository import get_session_ontology
from dlp_schema.db.repository import get_labels, insert_labels, insert_session, register_ontology
from dlp_schema.episode import current_labels
from dlp_schema.labels import ActionPayload, LabelRecord
from dlp_schema.ontology import load_ontology
from dlp_schema.review import ReviewReason
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.testing import sqlite_db

ROOT = repo_root()
BUCKETS = load_config(ROOT / "config/defaults.yaml").buckets
SID = "pan1"
D3 = {"model": "vllm-sr/d3-turbo", "served_name": "d3-turbo", "fingerprint": "vllm-bf16"}


def _world(tmp_path: Path) -> tuple[sa.Engine, LocalStore, list[LabelRecord]]:
    """승인된 합성 세션 하나 (블러본·렌더 기록)."""
    scene = aireview_scene(0, session_id=SID, n_units=2)
    engine = sqlite_db(f"sqlite:///{tmp_path / 'panel.db'}")
    session = scene.session.model_copy(
        update={
            "lifecycle_state": LifecycleState.PRELABELED,
            "privacy_state": PrivacyState.APPROVED,
        }
    )
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        insert_session(conn, session)
        insert_labels(conn, list(scene.labels))
    labeling = LocalStore(tmp_path / "store", BUCKETS.labeling)
    privacy = load_privacy_policy(ROOT)
    for stream in ("bodycam", "tp1"):
        video = tmp_path / f"{stream}.mp4"
        frames = ((t, scene_frame(scene, stream, t)) for t in scene.frame_times)
        write_video(video, frames, width=scene.width, height=scene.height)
        sha = sha256_file(video)
        labeling.put_file(blurred_key(SID, stream), video, sha)
        with engine.connect() as conn:
            digest = expected_render_hash(conn, SID, stream, privacy)
        write_render_meta(labeling, SID, stream, RenderMeta(digest, sha), tmp_path)
    return engine, labeling, list(scene.labels)


def _cfg(versions: tuple[str, ...], policy: AiReviewPolicy, *, enabled: bool) -> AiDisplayConfig:
    """advisory, 대조군 없음, 불일치 켬/끔. 표본 제외 판정 값은 wrong만 (불일치 효과를 가린다)."""
    display = AiReviewDisplayPolicy.model_validate({
        "mode": "advisory", "holdout_ratio": 0.0, "sampling_exclude": ["wrong"],
        "disagreement": {"enabled": enabled},
    })  # fmt: skip
    return AiDisplayConfig(display, versions, policy.target_rule)


def _ops_view(
    engine: sa.Engine, cfg: AiDisplayConfig
) -> tuple[set[str], set[str], set[str], set[str]]:
    """(갈린 라벨, AI_DISAGREEMENT 구간 라벨, 표본 제외 라벨, 판정이 있는 라벨)."""
    with engine.connect() as conn:
        current = current_labels(get_labels(conn, SID))
        view = load_ai_view(conn, SID, current, None, cfg)
    spans = ai_flagged_spans(view.escalate, current, cfg.policy, view.disagree)
    split = {i for s in spans if s.reason is ReviewReason.AI_DISAGREEMENT for i in s.label_ids}
    excluded = escalated_ids(view.escalate, cfg.policy, view.disagree)
    return set(view.disagree), split, excluded, set(view.escalate)


def test_two_panel_models_flag_only_the_split_label(tmp_path: Path) -> None:
    """A(가짜 서버, 모두 ok)와 B(판정단 d3, 오라클, 행동 하나만 wrong)의 판정이 서로 다른 버전으로
    쌓이고, 두 버전을 허용하면 그 행동에만 AI_DISAGREEMENT가 붙는다. B의 지적은 wrong이라 표본
    제외에도 들고, A·B 모두 ok인 라벨은 제외·사유가 없다. 꺼짐·한 버전만이면 불일치가 없다."""
    engine, labeling, labels = _world(tmp_path)
    base = load_policy(ROOT)
    paneled = AiReviewPolicy.model_validate(
        base.model_dump() | {"backend": base.backend.model_dump() | {"panel": [D3]}}
    )
    model_a, model_b = paneled, select_model(paneled, "vllm-sr/d3-turbo")
    assert model_a.verdict_version == base.verdict_version != model_b.verdict_version
    privacy = load_privacy_policy(ROOT)
    with FakeOpenAIServer(model_a.served_model) as server:
        judge = OpenAICompatibleJudge("spark-01", server.url, model_a)
        pool = EndpointPool([PoolMember("spark-01", judge, judge.ping)], model_a.server)
        first = run_from_db(
            engine, labeling, SID, pool, model_a, privacy, raw_bucket=BUCKETS.raw,
            options=RunOptions(workers=2),
        )  # fmt: skip
    assert first is not None and first.status == "completed"
    target = next(x for x in labels if isinstance(x.payload, ActionPayload))
    assert isinstance(target.payload, ActionPayload)
    swapped = target.payload.model_copy(
        update={"verb": "push" if target.payload.verb != "push" else "pull"}
    )
    truth = [x.model_copy(update={"payload": swapped}) if x is target else x for x in labels]
    second = run_from_db(
        engine, labeling, SID, OracleJudge(truth), model_b, privacy, raw_bucket=BUCKETS.raw,
        options=RunOptions(workers=2),
    )  # fmt: skip
    assert second is not None and second.status == "completed"
    with engine.connect() as conn:
        units = list_ai_units(conn, SID)
    assert {u.version for u in units} == {model_a.verdict_version, model_b.verdict_version}
    by_b = {v.label_id: v.verdict.value for u in units if u.version == model_b.verdict_version
            for v in u.verdicts}  # fmt: skip
    assert by_b[target.label_id] == "wrong"
    both = (model_a.verdict_version, model_b.verdict_version)
    disagree, split, excluded, judged = _ops_view(engine, _cfg(both, base, enabled=True))
    assert disagree == split == {target.label_id}
    assert excluded == {target.label_id}  # 두 모델이 ok로 같은 나머지 라벨은 아무것도 줄지 않는다
    assert len(judged) > 1
    assert _ops_view(engine, _cfg(both, base, enabled=False))[:2] == (set(), set())
    only_a = _ops_view(engine, _cfg((model_a.verdict_version,), base, enabled=True))
    assert only_a[:3] == (set(), set(), set())
    # 검수 화면 (표시 프로필: 지금 렌더 해시·세션 온톨로지): 갈린 행동에만 모델별 의견
    with engine.connect() as conn:
        current = current_labels(get_labels(conn, SID))
        hashes = {s: expected_render_hash(conn, SID, s, privacy) for s in ("bodycam", "tp1")}
        view = load_ai_view(
            conn, SID, current, hashes, _cfg(both, base, enabled=True),
            ontology=get_session_ontology(conn, SID),
        )  # fmt: skip
    opinions = ((model_a.backend.model, AiVerdictValue.OK), (D3["model"], AiVerdictValue.WRONG))
    assert view.display_split == {target.label_id: opinions}
    assert split_text(opinions) == (
        f"AI 판정단 의견이 갈림: {model_a.backend.model} 맞음 / vllm-sr/d3-turbo 틀림"
    )
