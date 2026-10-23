"""오탐 의심 힌트 하위 명령 `dlp review hints <세션>` (ADR 0055).

블러·객체·도구 라벨을 바꾸지 않고, 검수 화면에 "오탐 의심" 이슈로 달 힌트를 계산해 저장한다.
- 블러: `dlp_privacy.hints` 규칙(머리 관절 없음·손 위·짧은 깜박임) + `--tta`면 YuNet을 좌우
  뒤집기·배율 화면에 다시 돌린 지지 비율. 원본 버킷 `derived/privacy_review/<스트림>.hints.json`.
  블러 검수 작업 생성(`dlp review create --stage privacy`)이 누락 위험 구간 이슈 다음에 따로 단다
  (`review.yaml cvat.hint_issue_limit`).
- 객체·도구: `dlp_review.object_hints` 규칙(착용자 몸의 사람 박스) + `--tta`면 OWLv2 도구 트랙
  일부의 지지 비율. 라벨링 버킷 `sessions/<세션>/hints/<스트림>.objects.json`. 라벨링 CVAT 작업
  생성이 단다.

입력은 운영 라벨(`current_labels`)이다. `--tta`는 원본 영상을 감사 저장소로 받는다 (원본 접근 기록).
OWLv2 TTA는 CPU 프레임당 수 초라 `review.yaml object_hints.tta.max_tracks`로 줄인다.
`--ai`는 지금 승인된 블러본에서 박스를 잘라 사내 VLM(`DLP_VLM_URL`, 사내 주소 검사)에 묻고 no 답을
`ai_crop_no` 힌트로 싣는다 (ADR 0056, 판정은 라벨·검수 상태가 아니다. 서버 주소가 없으면 stub이 모두
unsure라 힌트가 생기지 않는다). 승인된 블러본이 없으면 그 스트림은 건너뛴다. 반환: 항상 0.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sqlalchemy as sa
from numpy.typing import NDArray

from dlp_aireview.backend import check_hosts
from dlp_aireview.crops import (
    CropAsk,
    CropVerifier,
    OracleCropVerifier,
    VlmCropVerifier,
    to_doubts,
    verify_crops,
)
from dlp_aireview.media import frames_for, open_blurred
from dlp_aireview.policy import load_policy as load_aireview_policy
from dlp_cli.raw_access import raw_store
from dlp_cli.review_config import load_review_policy
from dlp_cli.schema_cmds import database_url
from dlp_media.geometry import Box, Detector, TtaSpec, TtaTrack, tta_tracks
from dlp_media.pts import PtsIndex
from dlp_media.storage import ObjectStore, store_from_spec
from dlp_models.owlv2 import Owlv2
from dlp_models.registry import resolve
from dlp_models.vlm import OpenAICompatibleChat
from dlp_prelabel.policy import load_policy as load_prelabel_policy
from dlp_privacy import hints as privacy_hints
from dlp_privacy.detection import FrameDetector
from dlp_privacy.detectors import build_detectors
from dlp_privacy.errors import RenderNotCurrentError
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_review import object_hints
from dlp_review.ops.policy import AiCropsPolicy
from dlp_review.tasks import object_key
from dlp_schema import load_config, repo_root
from dlp_schema.db.repository import get_labels, get_session
from dlp_schema.episode import current_labels
from dlp_schema.labels import BlurTrackPayload, BoxKeyframe, BoxTrackPayload, LabelRecord
from dlp_schema.session import StreamKind

VIDEO = (StreamKind.BODYCAM, StreamKind.THIRD_PERSON)


def _pick(tracks: list[LabelRecord], spec: TtaSpec) -> list[TtaTrack]:
    """신뢰도 낮은 트랙부터 max_tracks개, 트랙마다 고르게 frames_per_track개 키프레임."""
    chosen = sorted(tracks, key=lambda x: (x.confidence or 0.0, x.label_id))[: spec.max_tracks]
    out: list[TtaTrack] = []
    for x in chosen:
        p = x.payload
        assert isinstance(p, BlurTrackPayload | BoxTrackPayload)
        frames = [k for k in p.keyframes if not k.outside]
        picks = np.linspace(0, len(frames) - 1, num=min(spec.frames_per_track, len(frames)))
        boxes: list[tuple[int, Box]] = []
        for i in picks.astype(int).tolist():
            k: BoxKeyframe = frames[int(i)]
            boxes.append((k.t_ms, (k.x, k.y, k.w, k.h)))
        key = p.target if isinstance(p, BlurTrackPayload) else p.class_id
        out.append((x.label_id, key, boxes))
    return out


def _face_detector(policy: PrivacyPolicy, root: Path) -> Detector | None:
    """얼굴 대상 탐지기 중 YuNet 하나 (없으면 None)."""
    ready, _ = build_detectors(policy, root)
    names = [n for n in policy.targets["face"].detectors if policy.detectors[n].kind == "yunet"]
    det = next((ready[n] for n in names if n in ready), None)
    if det is None:
        return None
    return face_tta_detector(det, policy.detection_threshold)


def face_tta_detector(det: FrameDetector, threshold: float) -> Detector:
    """프라이버시 프레임 탐지기 → TTA 탐지기 (얼굴 탐지만, 박스는 (x, y, w, h) 튜플).

    `Detection.box`는 `dlp_privacy.geometry.Box` 객체라 TTA의 튜플 박스로 풀어 준다.
    """

    def detect(img: NDArray[np.uint8]) -> list[tuple[str, Box]]:
        return [
            ("face", (d.box.x, d.box.y, d.box.w, d.box.h))
            for d in det.detect(img, 0, threshold)
            if d.target == "face"
        ]

    return detect


def _tool_detector(root: Path) -> Detector:
    """OWLv2 도구 탐지기 (전체 질의, 정책 min_score)."""
    policy = load_prelabel_policy(root)
    op = policy.open_vocab_objects
    model_path, _ = resolve(root, policy.models.open_vocab)
    tok_path, _ = resolve(root, policy.models.open_vocab_tokenizer)
    queries = list(op.queries)
    model = Owlv2(model_path, tok_path, queries, nms_iou=op.nms_iou)
    thresholds = [op.min_score] * len(queries)

    def detect(img: NDArray[np.uint8]) -> list[tuple[str, Box]]:
        return [(op.queries[queries[d.query_index]], d.box) for d in model.detect(img, thresholds)]

    return detect


def cmd_hints(args: argparse.Namespace) -> int:
    """`dlp review hints <세션> [--tta] [--out]`: 오탐 의심 힌트를 계산해 저장한다 (모듈 설명)."""
    root = repo_root()
    privacy, review = load_privacy_policy(root), load_review_policy(root)
    if review.object_hints is None:
        raise SystemExit("review.yaml에 object_hints 절이 없습니다")
    engine = sa.create_engine(database_url(args.url))
    with engine.connect() as conn:
        session = get_session(conn, args.session_id)
        labels = current_labels(get_labels(conn, args.session_id))
    raw = raw_store(args.store, args.url, "review.hints")
    buckets = load_config(root / "config/defaults.yaml").buckets
    labeling = store_from_spec(args.store, buckets.labeling)
    report: dict[str, Any] = {"session_id": args.session_id, "streams": {}}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for stream in (s for s in session.streams if s.kind in VIDEO):
            sid = stream.stream_id
            own = [x for x in labels if x.stream_id == sid]
            face_tta = tool_tta = None
            if args.tta:
                face_tta, tool_tta = _tta(
                    raw, stream.uri, own, privacy, review.object_hints.tta, root, work
                )
            ai_p: dict[str, float] | None = None
            ai_o: dict[str, float] | None = None
            if args.ai:
                ai_p, ai_o = _ai(
                    engine, labeling, args.session_id, sid, own, work, raw_bucket=buckets.raw
                )
            ph = privacy_hints.privacy_hints(labels, sid, privacy.hints, face_tta, ai_p)
            oh = object_hints.object_hints(labels, sid, review.object_hints, tool_tta, ai_o)
            priority = len(privacy.review_priority)
            privacy_hints.write_hint_segments(
                raw, args.session_id, sid, privacy_hints.hint_segments(ph, priority), work
            )
            object_hints.write_object_hints(labeling, args.session_id, sid, oh, work)
            report["streams"][sid] = _summary(own, ph, oh, face_tta, tool_tta)
    engine.dispose()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"리포트: {args.out}")
    for sid, s in report["streams"].items():
        print(f"{sid}: 블러 힌트 {s['privacy']}, 객체 힌트 {s['objects']}")
    return 0


def _tta(
    raw: ObjectStore,
    uri: str,
    own: list[LabelRecord],
    privacy: PrivacyPolicy,
    tool_spec: TtaSpec,
    root: Path,
    work: Path,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """원본 영상을 받아 얼굴·도구 TTA 지지 비율을 잰다 (탐지기가 없으면 그쪽은 None)."""
    video = work / "tta.mp4"
    raw.get_file(object_key(raw, uri), video)
    faces = [
        x for x in own if isinstance(x.payload, BlurTrackPayload) and x.payload.target == "face"
    ]
    tools = [
        x
        for x in own
        if isinstance(x.payload, BoxTrackPayload)
        and (x.provenance.model_version or "").startswith("owlv2")
    ]
    face_det = _face_detector(privacy, root)
    face = (
        tta_tracks(video, _pick(faces, privacy.hints.tta), face_det, privacy.hints.tta)
        if face_det and privacy.hints.tta.max_tracks
        else None
    )
    tool = (
        tta_tracks(video, _pick(tools, tool_spec), _tool_detector(root), tool_spec)
        if tools and tool_spec.max_tracks
        else None
    )
    video.unlink()
    return face, tool


def _crop_asks(own: list[LabelRecord], cfg: AiCropsPolicy, pts: PtsIndex) -> list[CropAsk]:
    """물을 박스: 블러(정책 대상)·도구 트랙을 신뢰도 낮은 것부터 max_tracks개씩."""
    spec = TtaSpec(transforms=("hflip",), iou=0.5, max_tracks=cfg.max_tracks,
                   frames_per_track=cfg.frames_per_track)  # fmt: skip
    blur = [
        x
        for x in own
        if isinstance(x.payload, BlurTrackPayload) and x.payload.target in cfg.targets
    ]
    tools = [
        x for x in own
        if cfg.tools and isinstance(x.payload, BoxTrackPayload)
        and (x.provenance.model_version or "").startswith("owlv2")
    ]  # fmt: skip
    asks: list[CropAsk] = []
    for kind, tracks in (("privacy", blur), ("object", tools)):
        for label_id, key, frames in _pick(tracks, spec):
            asks += [CropAsk(label_id, kind, key, pts.nearest(t), box) for t, box in frames]  # type: ignore[arg-type]
    return asks


def _ai(
    engine: sa.Engine,
    labeling: ObjectStore,
    session_id: str,
    stream_id: str,
    own: list[LabelRecord],
    work: Path,
    *,
    raw_bucket: str,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """승인된 블러본 조각을 사내 VLM에 물어 (블러 라벨 의심, 박스 라벨 의심). 블러본 없으면 None."""
    root = repo_root()
    cfg = load_review_policy(root).ai_crops
    if cfg is None:
        raise SystemExit("review.yaml에 ai_crops 절이 없습니다")
    try:
        media = open_blurred(
            engine, labeling, session_id, stream_id, load_privacy_policy(root), work,
            raw_bucket=raw_bucket,
        )  # fmt: skip
    except RenderNotCurrentError as exc:
        print(f"[AI 조각 확인 건너뜀] {stream_id}: {exc}")
        return None, None
    asks = _crop_asks(own, cfg, media.pts)
    verdicts = verify_crops(
        asks, frames_for(media, sorted({a.frame_index for a in asks})), _verifier(root, cfg),
        margin=cfg.margin, max_side=cfg.max_side,
    )  # fmt: skip
    media.path.unlink(missing_ok=True)
    kinds = {a.label_id: a.kind for a in asks}
    doubts = to_doubts(verdicts, cfg.min_confidence)
    return (
        {k: v for k, v in doubts.items() if kinds[k] == "privacy"},
        {k: v for k, v in doubts.items() if kinds[k] == "object"},
    )


def _verifier(root: Path, cfg: AiCropsPolicy) -> CropVerifier:
    """사내 VLM 조각 확인기 (`DLP_VLM_URL`, 사내 주소만). 주소가 없으면 stub(늘 unsure)."""
    url = os.environ.get("DLP_VLM_URL")
    if not url:
        print("[AI 조각 확인] DLP_VLM_URL이 없어 stub(모두 unsure)으로 돌립니다")
        return OracleCropVerifier()
    policy = load_aireview_policy(root)
    outside = check_hosts({"crops": url}, policy.server.internal_hosts)
    if outside:
        raise SystemExit("사내 대역 밖 VLM 서버에는 보내지 않습니다: " + "; ".join(outside))
    chat = OpenAICompatibleChat(
        url, policy.served_model, timeout_s=policy.server.timeout_s,
        api_key=os.environ.get("DLP_VLM_API_KEY"), expected_model=policy.served_model,
    )  # fmt: skip
    return VlmCropVerifier(chat, policy.backend.model, cfg.jpeg_quality)


def _summary(
    own: list[LabelRecord],
    ph: list[privacy_hints.Hint],
    oh: list[object_hints.ObjectHint],
    face_tta: dict[str, float] | None,
    tool_tta: dict[str, float] | None,
) -> dict[str, Any]:
    """스트림 하나의 힌트 요약 (이유별 트랙 수, 전체 대비 비율)."""
    faces = sum(isinstance(x.payload, BlurTrackPayload) and x.payload.target == "face" for x in own)
    people = sum(
        isinstance(x.payload, BoxTrackPayload) and x.payload.class_id == "person" for x in own
    )
    return {
        "privacy": dict(Counter(f"{h.target}:{h.reason}" for h in ph)),
        "objects": dict(Counter(f"{h.class_id}:{h.reason}" for h in oh)),
        "face_tracks": faces,
        "face_tracks_flagged": len({h.label_id for h in ph if h.target == "face"}),
        "person_tracks": people,
        "person_tracks_flagged": len({h.label_id for h in oh if h.class_id == "person"}),
        "face_tta_tracks": len(face_tta or {}),
        "face_tta_mean_support": round(float(np.mean(list(face_tta.values()))), 3)
        if face_tta
        else None,
        "tool_tta_tracks": len(tool_tta or {}),
        "tool_tta_mean_support": round(float(np.mean(list(tool_tta.values()))), 3)
        if tool_tta
        else None,
    }


def add_review_commands(rsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`review hints`를 등록한다 (`review_cmds.add_commands`가 부른다)."""
    h = rsub.add_parser("hints", help="오탐 의심 힌트 계산·저장 (라벨은 바꾸지 않음)")
    h.add_argument("session_id")
    h.add_argument(
        "--tta", action="store_true", help="YuNet·OWLv2를 변환 화면에 다시 돌린다 (느림)"
    )
    h.add_argument(
        "--ai", action="store_true", help="승인된 블러본 조각을 사내 VLM에 물어 힌트를 더한다"
    )
    h.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    h.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    h.add_argument("--out", help="요약 JSON 파일")
    h.set_defaults(func=cmd_hints)
