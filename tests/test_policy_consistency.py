"""정책 파일 사이의 값이 서로 맞는지 검사한다.

서로 다른 패키지의 정책이 한 값을 기준으로 맞물려 있을 때, 한쪽만 바꿔 조용히 깨지는 것을 막는다.
설정 주석이 코드의 저장 위치와 어긋나 운영자를 잘못 이끄는 경우(버킷 설명)도 함께 본다.
"""

from __future__ import annotations

from pathlib import Path

from dlp_retarget.policy import load_policy as load_retarget

from dlp_actions.policy import load_policy as load_actions
from dlp_aireview.policy import load_policy as load_aireview
from dlp_export.policy import load_policy as load_export
from dlp_ops.policy import load_policy as load_ops
from dlp_prelabel.policy import HAND21_JOINTS
from dlp_prelabel.policy import load_policy as load_prelabel
from dlp_review.ops.policy import load_policy as load_review
from dlp_schema import find_ontology_dir, load_ontology

ROOT = Path(__file__).resolve().parents[1]


def test_contact_mismatch_threshold_separates_single_source_contacts() -> None:
    """검수 우선순위의 접촉 불일치 문턱은 한쪽 출처(장갑만·영상만)는 잡고 융합은 놓아야 한다.

    `prelabel.yaml contact.confidence`의 출처별 신뢰도와 `review.yaml
    priority.contact_mismatch_max_confidence`를 비교한다: max(장갑, 영상) ≤ 문턱 < 융합.
    """
    conf = load_prelabel(ROOT).contact.confidence
    threshold = load_review(ROOT).priority.contact_mismatch_max_confidence
    assert max(conf.glove, conf.video) <= threshold < conf.fused


def test_aireview_policy_matches_neighbours() -> None:
    """AI 1차 검수 정책과 이웃 정책 (WP17, ADR 0040).

    - 대상을 모르는 접촉 ID는 행동 단계와 같다 (`actions.yaml unresolved_entity_ids`).
    - 판정 모델은 행동 분류 VLM과 다르다 (같으면 행동 라벨 판정에 상관 오류, 사용자 결정).
    - 장갑 Parquet을 읽는 원본 접근 용도 `aireview.run`이 서비스 용도로 등록돼 있다 (월간 감사가
      이상 접근으로 표시하지 않게).
    """
    aireview, actions = load_aireview(ROOT), load_actions(ROOT)
    assert set(aireview.unresolved_entity_ids) == set(actions.unresolved_entity_ids)
    assert aireview.backend.model != actions.vlm.model
    assert "aireview.run" in load_ops(ROOT).audit.service_purposes


def test_ok_miss_thresholds_match_sampling_defect_ratio() -> None:
    """AI ok 놓침률 기준 세 곳은 같은 값이다 (각 정책 주석의 "같은 값으로 둔다", WP17.8).

    - `ops.yaml alerts.ai_ok_miss_upper95_max`: 주간 경고 (대조군 ok 놓침률 95% 상한).
    - `aireview.yaml gate.max_ok_miss_rate`: 골든 게이트 (`dlp aireview eval --gate`).
    - `review.yaml sampling.max_defect_ratio`: 표본 합격 묶음의 결함 비율 상한 (`dlp review quality
      --ai`가 기준으로 보인다).
    AI의 맞음이 표본 허용 결함률보다 나쁘면 믿을 근거가 없다. 한 값만 바꾸면 경고·게이트·품질 보고의
    기준이 서로 달라진다.
    """
    defect = load_review(ROOT).sampling.max_defect_ratio
    assert load_ops(ROOT).alerts.ai_ok_miss_upper95_max == defect
    assert load_aireview(ROOT).gate.max_ok_miss_rate == defect


def _key_comment(text: str, key: str) -> str:
    """YAML 텍스트에서 `  <key>: 값  # 주석` 줄의 주석 부분 (그 키 줄이 정확히 하나여야 한다)."""
    lines = [ln for ln in text.splitlines() if ln.startswith(f"  {key}: ")]
    assert len(lines) == 1, (key, lines)
    return lines[0].partition("#")[2]


def test_bucket_comments_keep_proxy_in_raw_bucket() -> None:
    """F-PMS-04: 버킷 설명 주석은 프록시(블러 전 영상)를 원본 버킷에 둔다고 적어야 한다.

    수집(`dlp_media.ingest`)은 프록시를 원본 저장소의 `sessions/<세션>/derived/`에 쓰고, 블러 검수
    (`dlp_review.tasks.proxy_key`)도 원본 버킷에서 읽는다. 주석을 믿고 프록시를 라벨링 버킷에 두면
    일반 라벨러 자격 증명으로 블러 전 영상이 열린다. 정답: defaults.yaml `buckets.raw` 줄 주석에는
    "프록시"가 있고 `buckets.labeling` 줄 주석에는 없다.
    """
    from dlp_review.tasks import proxy_key

    assert proxy_key("s1", "bodycam") == "sessions/s1/derived/bodycam.proxy.mp4"
    text = (ROOT / "config" / "defaults.yaml").read_text(encoding="utf-8")
    assert "프록시" in _key_comment(text, "raw")
    assert "프록시" not in _key_comment(text, "labeling")


# 바디캠으로 받는 가장 느린 프레임률 (fps, 영화 24 fps NTSC판). 3D 표본 간격의 여유를 정한다
SLOWEST_FPS = 23.976


def test_3d_interpolation_limit_covers_the_depth_sample_stride() -> None:
    """LeRobot 3D 보간 한도는 3D 표본 간격(깊이 간격 + 최대 한 프레임)보다 커야 한다.

    3D 표본은 `prelabel.yaml depth.frame_stride_ms`가 지난 뒤 첫 영상 프레임에 찍힌다 (월드 좌표
    표본도 같은 시각). 한도가 깊이 간격과 같으면 29.97·24 fps·VFR에서 손 3D·로봇 목표가 대부분
    빠졌다 (F-COE-50, ADR 0053).
    """
    stride = load_prelabel(ROOT).depth.frame_stride_ms
    limit = load_export(ROOT).lerobot.interp_max_gap_3d_ms
    assert limit >= stride + 1000 / SLOWEST_FPS


def test_exported_and_retargeted_hand_points_are_produced_by_prelabel() -> None:
    """내보내기 손 3D 점(`export.yaml lerobot.hand_points_3d`)과 리타기팅 손 부위(`retarget.yaml
    hand_parts`)는 프리라벨이 만드는 부위(`prelabel.yaml depth.hand_points`) 안이고 온톨로지
    hand_joints 안이다 (한쪽만 바꾸면 그 열이 아무 경고 없이 비었다, F-COE-54)."""
    made = {HAND21_JOINTS[i] for i in load_prelabel(ROOT).depth.hand_points}
    exported = set(load_export(ROOT).lerobot.hand_points_3d)
    retargeted = set(load_retarget(ROOT).hand_parts.model_dump().values())
    ontology = load_ontology(find_ontology_dir(ROOT / "config" / "ontology", "1.0.0"))
    assert exported <= made and retargeted <= made
    assert made <= set(ontology.hand_joints)
