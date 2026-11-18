"""촉각장갑 채널을 에피소드 프레임 시각에 맞춰 싣는다 (`dlp export lerobot|gr00t`, ADR 0069).

세션의 장갑 스트림(glove_left·glove_right)마다 정규화 Parquet(`dlp_media.glove`, 시각은 장갑 기기
시계)을 읽어 샘플 시각을 그 스트림의 동기화 결과(`Stream.to_master_ms`)로 마스터 시각으로 옮기고,
에피소드 프레임 시각(`Episode.times_ms`, 마스터 시각)의 채널 값을 앞뒤 샘플 사이 선형 보간으로
구한다. 프레임 시각은 `build_episode`가 정한 것을 그대로 받는다 (여기서 새로 계산하지 않는다).

규칙
- 동기화되지 않은(`sync_method` unsynced) 장갑은 시간축을 알 수 없어 싣지 않는다. 판정 기준은 접촉
  프리라벨(`dlp_prelabel.runner`)·AI 1차 검수 입력과 같다. 싣지 않은 사실은 세션 기록에 남는다.
- 값을 지어내지 않는다. 장갑 기록 범위 밖(외삽 없음)과 앞뒤 샘플 간격이 `interp_max_gap_ms`보다
  큰 프레임은 "없음"이다: 값 0과 표시 0 (손 3D 점과 같은 표현). 실은 장갑이 없는 세션·손도 같다.
- 세션마다 같은 손의 채널 구성이 다르면(장갑 기종이 다름) 한 데이터셋의 열 모양이 달라지므로
  맞춰 넣지 않고 `ExportError`로 거부한다.
- 장갑 Parquet은 원본 버킷에 있다. 감사 저장소(`dlp_media.audit.AuditedStore`, 만드는 곳은
  `dlp_cli.raw_access.raw_store`)로만 받아 접근이 기록된다. 결과물에는 채널 값·채널 이름·장갑
  기종·동기화 방법만 넣고 원본 위치(버킷·키·URI)는 넣지 않는다.
- 값은 기기가 낸 그대로다 (단위 모름, 세션 계약의 기종별 보정값 `glove_calibration` 미적용).

싣는 곳 (손마다, 그 손을 실은 세션이 하나라도 있을 때만 생긴다)
- LeRobot: 특징 `observation.tactile.<손>` (float32, 채널 값 + 끝에 present).
- GR00T: `observation.state` 뒤쪽의 `<손>_tactile` 구간(`meta/modality.json` state)과
  `observation.tactile_valid`(손마다 1/0). action에는 넣지 않는다 (관측이지 행동이 아니다).
- 두 형식 모두 `meta/dlp_tactile.json`(`TactileData.meta`): 채널 이름, 장갑 기종, 단위 모름, 시간
  맞춤 방식, 세션별 상태. manifest `details.tactile`도 같은 내용이다.

흐름: `runner.build_episodes`가 세션마다 `TactileCollector.add`를 부르고, 끝에 `result()`의
`TactileData`를 `lerobot.write_package`·`gr00t.write_gr00t`에 넘긴다. 장갑 스트림이 있는 세션이
하나도 없으면 `result()`는 None이고 결과물은 촉각 기능이 없던 때와 같다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from dlp_export.policy import TactilePolicy
from dlp_export.source import ExportError
from dlp_media.audit import AuditedStore
from dlp_media.glove import GloveData, read_glove
from dlp_schema.labels import Hand
from dlp_schema.session import Session, Stream, StreamKind, SyncMethod

# meta/dlp_tactile.json 형식 이름과 판. 판 규칙은 구간 JSON(`dlp_schema.export`)과 같다: 필드를
# 더하면 부 버전을, 받는 쪽 리더가 깨지는 변경이면 주 버전을 올리고 ADR에 적는다.
FORMAT = "dlp-tactile"
FORMAT_VERSION = "1.0"
# 특징 배열의 손 순서 (왼손이 앞, `dlp_export.lerobot.HANDS`와 같다)
HANDS = (Hand.LEFT, Hand.RIGHT)
# 장갑 스트림 종류 → 손
GLOVE_HANDS = {StreamKind.GLOVE_LEFT: Hand.LEFT, StreamKind.GLOVE_RIGHT: Hand.RIGHT}
# LeRobot 특징 끝의 유효 표시 이름 (채널 이름으로 쓸 수 없다)
PRESENT = "present"
# GR00T 유효 표시 열 (손마다 1/0)
VALID_COLUMN = "observation.tactile_valid"

# meta/dlp_tactile.json 설명 (구매자용 영어, 계약의 의미라 정책 값이 아니다)
_SOURCE = "tactile gloves worn by the human demonstrator (not a robot sensor)"
_TIME_ALIGNMENT = (
    "Frame times are the episode's master-timeline times (bodycam clock). Each glove sample time "
    "(device clock) is mapped to the master timeline with that stream's synchronisation result "
    "(offset, clock scale, manual adjustment); the value at a frame time is linearly interpolated "
    "between the two neighbouring samples."
)
_MISSING = (
    "Value 0 with the present/valid flag 0 when the frame is outside the glove recording (no "
    "extrapolation), when the two neighbouring samples are more than interp_max_gap_ms apart, or "
    "when the episode has no exported glove for that hand (see sessions[].hands[].status). "
    "0 can also be a real reading, so use the flag."
)
_UNITS = "unknown: raw device values as recorded, no per-model calibration applied"


def feature_key(hand: Hand) -> str:
    """LeRobot 특징 키 (예: observation.tactile.left)."""
    return f"observation.tactile.{hand.value}"


def npz_key(hand: Hand) -> str:
    """LeRobot 묶음 npz의 배열 키."""
    return f"tactile_{hand.value}"


def channel_names(hand: Hand, channels: tuple[str, ...]) -> list[str]:
    """채널의 특징 이름 (예: left.tactile.pressure_0). LeRobot 특징과 GR00T state가 같이 쓴다."""
    return [f"{hand.value}.tactile.{c}" for c in channels]


def present_name(hand: Hand) -> str:
    """LeRobot 특징 끝 유효 표시의 이름 (예: left.tactile.present)."""
    return f"{hand.value}.tactile.{PRESENT}"


def state_key(hand: Hand) -> str:
    """GR00T modality.json state 키 (예: left_tactile)."""
    return f"{hand.value}_tactile"


def valid_name(hand: Hand) -> str:
    """GR00T `observation.tactile_valid` 안 이름 (LeRobot 표시 이름에서 `.present`를 뗀 것)."""
    return f"{hand.value}.tactile"


def frame_values(
    sample_ms: NDArray[np.float64],
    values: NDArray[np.float64],
    frame_ms: NDArray[np.float64],
    max_gap_ms: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """프레임 시각마다의 값과 유효 표시 (순수 함수).

    `dlp_export.lerobot.interp_series`와 같은 규칙을 배열로 계산한다: 샘플 시각과 같으면 그 값,
    두 샘플 사이면 선형 보간, 첫 샘플 앞·마지막 샘플 뒤(외삽)와 앞뒤 샘플 간격이 `max_gap_ms`보다
    큰 곳은 없음이다.

    Args:
        sample_ms: 샘플 시각 (m,), 프레임 시각과 같은 시계(마스터 ms), 오름차순.
        values: 샘플 값 (m, 채널 수).
        frame_ms: 값을 읽을 프레임 시각 (n,).
        max_gap_ms: 보간을 허용하는 최대 샘플 간격 (ms).

    Returns:
        (값 (n, 채널 수), 유효 (n,)). 유효하지 않은 행의 값은 0이다.
    """
    m = sample_ms.size
    if m == 0:
        return np.zeros((frame_ms.size, values.shape[1])), np.zeros(frame_ms.size, dtype=bool)
    # hi: 시각이 프레임 시각 이상인 첫 샘플의 위치 (m이면 마지막 샘플 뒤)
    hi = np.searchsorted(sample_ms, frame_ms, side="left")
    at = np.minimum(hi, m - 1)
    lo = np.maximum(hi - 1, 0)
    exact = (hi < m) & (sample_ms[at] == frame_ms)
    between = (hi > 0) & (hi < m)
    gap = sample_ms[at] - sample_ms[lo]
    valid = exact | (between & (gap <= max_gap_ms))
    w = np.zeros(frame_ms.size)
    np.divide(frame_ms - sample_ms[lo], gap, out=w, where=between & (gap > 0))
    out = values[lo] * (1 - w)[:, None] + values[at] * w[:, None]
    out[exact] = values[at[exact]]
    out[~valid] = 0.0
    return out, valid


def master_times(stream: Stream, t_ms: NDArray[np.float64]) -> NDArray[np.float64]:
    """장갑 샘플 시각(장갑 기기 시계 ms) → 마스터 시각 ms.

    계약의 `Stream.to_master_ms`를 샘플마다 부른다 (식을 여기 다시 적지 않는다). 배율이 양수이고
    샘플 시각이 순증가(`GloveData` 검증)라 결과도 오름차순이다.
    """
    return np.array([stream.to_master_ms(float(t)) for t in t_ms], dtype=np.float64)


def _natural(name: str) -> list[tuple[int, int | str]]:
    """자연 정렬 키: 숫자 조각은 수로 비교한다 (pressure_2가 pressure_10 앞)."""
    return [(0, int(p)) if p.isdigit() else (1, p) for p in re.split(r"(\d+)", name) if p]


def select_channels(names: tuple[str, ...], prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """실을 채널 (자연 정렬). 접두사가 비면 전부, 있으면 그중 하나로 시작하는 채널만.

    파일의 열 순서가 아니라 이름순으로 정해, 같은 채널을 가진 세션은 늘 같은 열 순서가 된다.
    """
    chosen = [n for n in names if not prefixes or n.startswith(prefixes)]
    return tuple(sorted(chosen, key=_natural))


def glove_streams(session: Session) -> dict[Hand, list[Stream]]:
    """손 → 그 손의 장갑 스트림 (세션에 적힌 순서). 장갑이 없는 손은 키가 없다."""
    out: dict[Hand, list[Stream]] = {}
    for s in session.streams:
        if s.kind in GLOVE_HANDS:
            out.setdefault(GLOVE_HANDS[s.kind], []).append(s)
    return out


@dataclass(frozen=True)
class HandSeries:
    """에피소드 하나에서 한 손의 프레임별 촉각 값."""

    values: NDArray[np.float32]  # (프레임 수, 채널 수). 없음은 0
    valid: NDArray[np.bool_]  # (프레임 수,)


@dataclass(frozen=True)
class TactileData:
    """한 내보내기의 촉각 값 전체 (`TactileCollector.result`).

    Attributes:
        hands: 실은 손 → 채널 이름 (`HANDS` 순). 실은 장갑이 하나도 없으면 비어 있고, 이때는
            특징·열을 더하지 않고 세션 기록만 남긴다.
        episodes: 에피소드마다(`BuiltEpisodes.items`와 같은 순서) 실은 손 → 값.
        frames: 에피소드마다 프레임 수 (실은 장갑이 없는 손을 없음으로 채울 때 쓴다).
        sessions: 세션 기록 (가명 세션 ID, 장갑 기종, 손마다 상태·동기화 방법·채움).
        policy: 적용한 정책 (`meta`에 보간 한도와 채널 선택을 적는다).
    """

    hands: dict[Hand, tuple[str, ...]]
    episodes: list[dict[Hand, HandSeries]]
    frames: list[int]
    sessions: list[dict[str, Any]]
    policy: TactilePolicy

    def _series(self, i: int, hand: Hand) -> HandSeries:
        """에피소드 i의 그 손 값. 실은 장갑이 없으면 전부 없음(값 0, 표시 0)이다."""
        got = self.episodes[i].get(hand)
        if got is not None:
            return got
        n = self.frames[i]
        return HandSeries(np.zeros((n, len(self.hands[hand])), np.float32), np.zeros(n, bool))

    # LeRobot ---------------------------------------------------------------------------------

    def lerobot_features(self) -> dict[str, Any]:
        """LeRobot `features`에 더할 정의: 손마다 float32 [채널 값…, present]."""
        return {
            feature_key(h): {
                "dtype": "float32",
                "shape": [len(c) + 1],
                "names": [*channel_names(h, c), present_name(h)],
            }
            for h, c in self.hands.items()
        }

    def lerobot_frame_arrays(self) -> dict[str, str]:
        """LeRobot 특징 → npz 배열 키 (`lerobot.FRAME_ARRAYS`에 더한다)."""
        return {feature_key(h): npz_key(h) for h in self.hands}

    def lerobot_arrays(self, i: int) -> dict[str, NDArray[np.float32]]:
        """에피소드 i의 npz 배열: 손마다 (프레임 수, 채널 수 + 1), 마지막 열이 present."""
        out: dict[str, NDArray[np.float32]] = {}
        for h in self.hands:
            s = self._series(i, h)
            out[npz_key(h)] = np.hstack([s.values, s.valid[:, None].astype(np.float32)])
        return out

    # GR00T -----------------------------------------------------------------------------------

    def gr00t_names(self) -> list[str]:
        """GR00T `observation.state` 끝에 붙는 열 이름 (손 순서대로 채널)."""
        return [n for h, c in self.hands.items() for n in channel_names(h, c)]

    def gr00t_modality(self, start: int) -> dict[str, dict[str, int]]:
        """modality.json state에 더할 구간 (`{<손>_tactile: {start, end}}`, `start`부터 잇는다)."""
        out: dict[str, dict[str, int]] = {}
        for h, c in self.hands.items():
            out[state_key(h)] = {"start": start, "end": start + len(c)}
            start += len(c)
        return out

    def gr00t_valid_names(self) -> list[str]:
        """`observation.tactile_valid` 열 이름 (손마다 하나)."""
        return [valid_name(h) for h in self.hands]

    def gr00t_arrays(self, i: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """에피소드 i의 (state에 붙일 값 (프레임 수, 채널 합), 유효 표시 (프레임 수, 손 수))."""
        series = [self._series(i, h) for h in self.hands]
        values = np.hstack([s.values for s in series]).astype(np.float32)
        valid = np.stack([s.valid for s in series], axis=1).astype(np.float32)
        return np.ascontiguousarray(values), np.ascontiguousarray(valid)

    # 기록 ------------------------------------------------------------------------------------

    def meta(self, fmt: Literal["lerobot", "gr00t"]) -> dict[str, Any]:
        """meta/dlp_tactile.json·manifest `details.tactile` 내용 (원본 위치·내부 ID 없음).

        Args:
            fmt: 형식. 손마다 값이 실린 위치를 그 형식의 이름으로 적는다 (LeRobot `feature`·
                `present`, GR00T `state_key`·`valid_column`·`valid_name`).
        """
        hands: dict[str, Any] = {}
        for h, c in self.hands.items():
            where = (
                {"feature": feature_key(h), "present": present_name(h)}
                if fmt == "lerobot"
                else {
                    "state_key": state_key(h),
                    "valid_column": VALID_COLUMN,
                    "valid_name": valid_name(h),
                }
            )
            hands[h.value] = {"channels": list(c), **where}
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "source": _SOURCE,
            "time_alignment": _TIME_ALIGNMENT,
            "missing": _MISSING,
            "units": _UNITS,
            "interp_max_gap_ms": self.policy.interp_max_gap_ms,
            "channel_prefixes": list(self.policy.channel_prefixes),
            "hands": hands,
            "glove_models": self.glove_models(),
            "sessions": self.sessions,
        }

    def glove_models(self) -> list[str | None]:
        """장갑을 실은 세션들의 장갑 기종 (중복 없이 정렬, 모르는 세션이 있으면 끝에 None)."""
        models = {
            s["glove_model"]
            for s in self.sessions
            if any(h["status"] == "exported" for h in s["hands"].values())
        }
        named = sorted(m for m in models if m is not None)
        return [*named, None] if None in models else [*named]


@dataclass
class TactileCollector:
    """세션마다 장갑을 읽어 프레임 값을 모은다 (`runner.build_episodes`).

    Attributes:
        raw: 원본 버킷 감사 저장소. 장갑 Parquet 읽기마다 접근 기록이 남는다. None이면 읽을 장갑이
            나왔을 때 실패한다 (장갑이 없거나 동기화되지 않은 세션뿐이면 읽지 않으므로 필요 없다).
        policy: export.yaml `tactile` 절.
        work: 작업 디렉터리 (`work/tactile/<세션>/<스트림>.parquet`에 받는다).
        warnings: 내보내기 경고 (manifest details.warnings, 세션은 가명으로 적는다).
    """

    raw: AuditedStore | None
    policy: TactilePolicy
    work: Path
    warnings: list[str] = field(default_factory=list[str])
    _layout: dict[Hand, tuple[str, ...]] = field(default_factory=dict[Hand, tuple[str, ...]])
    # 손 → 채널 구성을 처음 정한 세션 (내부 ID, 오류 메시지용)과 그 장갑 기종
    _first: dict[Hand, tuple[str, str | None]] = field(
        default_factory=dict[Hand, tuple[str, str | None]]
    )
    _episodes: list[dict[Hand, HandSeries]] = field(default_factory=list[dict[Hand, HandSeries]])
    _frames: list[int] = field(default_factory=list[int])
    _sessions: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    _with_glove: int = 0  # 장갑 스트림이 하나라도 있는 세션 수

    def add(self, session: Session, sid: str, frame_ms: NDArray[np.float64]) -> None:
        """세션 하나(에피소드 하나)의 장갑을 읽어 더한다. 내보낼 순서대로 부른다.

        Args:
            session: 세션 (데이터셋 버전 스냅샷에 고정된 스트림·동기화 값).
            sid: 결과물에 적을 세션 ID (가명).
            frame_ms: 에피소드 프레임 시각 (`Episode.times_ms`, 마스터 ms).

        세션 기록의 손별 상태(`status`)는 넷 중 하나다: exported(실음), no_glove(장갑 스트림 없음),
        not_synced(동기화 안 됨), multiple_streams(같은 손 장갑 스트림이 여럿).

        Raises:
            ExportError: 읽을 장갑이 있는데 감사 저장소가 없을 때, 장갑 Parquet이 없거나 읽을 수
                없을 때, 고른 채널이 없을 때, 같은 손의 채널 구성이 앞 세션과 다를 때.
        """
        streams = glove_streams(session)
        hands: dict[Hand, HandSeries] = {}
        report: dict[str, Any] = {}
        for hand in HANDS:
            series, report[hand.value] = self._hand(session, sid, hand, streams.get(hand, []))
            if series is not None:
                hands[hand] = HandSeries(*self._at_frames(series, frame_ms, report[hand.value]))
                self._coverage_warning(sid, hand, report[hand.value])
        self._with_glove += bool(streams)
        self._episodes.append(hands)
        self._frames.append(int(frame_ms.size))
        self._sessions.append(
            {"session_id": sid, "glove_model": session.calibration.glove_model, "hands": report}
        )

    def result(self) -> TactileData | None:
        """모은 값. 장갑 스트림이 있는 세션이 하나도 없으면 None (결과물이 예전과 같다).

        모든 세션을 더한 뒤 한 번 부른다 (장갑 기종이 섞였으면 여기서 경고를 하나 더한다).
        """
        if self._with_glove == 0:
            return None
        data = TactileData(
            {h: self._layout[h] for h in HANDS if h in self._layout},
            self._episodes,
            self._frames,
            self._sessions,
            self.policy,
        )
        models = data.glove_models()
        if len(models) > 1:
            self.warnings.append(
                f"촉각: 장갑 기종이 섞여 있음 ({', '.join(str(m) for m in models)}). 채널 이름은 "
                "같아도 값의 척도가 다를 수 있다 (보정하지 않은 기기 값, meta/dlp_tactile.json "
                "sessions[].glove_model)"
            )
        return data

    def _hand(
        self, session: Session, sid: str, hand: Hand, streams: list[Stream]
    ) -> tuple[tuple[Stream, GloveData, tuple[str, ...]] | None, dict[str, Any]]:
        """한 손의 (실을 장갑 (스트림, 데이터, 채널) 또는 None, 그 손의 기록)."""
        if not streams:
            return None, {"status": "no_glove"}
        if len(streams) > 1:
            self.warnings.append(
                f"{sid}: {hand.value} 장갑 스트림이 {len(streams)}개라 어느 것이 그 손의 값인지 "
                "정할 수 없어 촉각 값을 싣지 않음"
            )
            return None, {"status": "multiple_streams"}
        stream = streams[0]
        if stream.sync_method is SyncMethod.UNSYNCED:
            self.warnings.append(
                f"{sid}: {hand.value} 장갑이 동기화되지 않아 촉각 값을 싣지 않음 (dlp sync run "
                "또는 dlp sync adjust 뒤 데이터셋 버전을 다시 만든다)"
            )
            return None, {"status": "not_synced"}
        glove = self._read(session, stream)
        where = f"{session.session_id}/{stream.stream_id}"
        channels = self._channels(glove, where)
        self._check_layout(session, hand, channels)
        return (stream, glove, channels), {
            "status": "exported",
            "sync_method": stream.sync_method.value,
            "sync_confidence": stream.sync_confidence,
            "sample_rate_hz": round(glove.sample_rate_hz, 3),
        }

    def _at_frames(
        self,
        series: tuple[Stream, GloveData, tuple[str, ...]],
        frame_ms: NDArray[np.float64],
        report: dict[str, Any],
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """장갑 값을 프레임 시각에서 읽고, 채움 수를 기록에 적는다."""
        stream, glove, channels = series
        master = master_times(stream, glove.t_ms)
        samples = np.stack([glove.channels[c] for c in channels], axis=1)
        values, valid = frame_values(master, samples, frame_ms, self.policy.interp_max_gap_ms)
        report["frames"] = int(frame_ms.size)
        # 장갑 기록 구간(첫 샘플~마지막 샘플) 안 프레임 수와 그중 값이 있는 수
        report["frames_in_recording"] = int(
            ((frame_ms >= master[0]) & (frame_ms <= master[-1])).sum()
        )
        report["frames_valid"] = int(valid.sum())
        return values.astype(np.float32), valid

    def _coverage_warning(self, sid: str, hand: Hand, report: dict[str, Any]) -> None:
        """장갑 기록이 에피소드와 겹치지 않거나 기록 구간 안 채움 비율이 낮으면 경고를 남긴다."""
        span, valid = report["frames_in_recording"], report["frames_valid"]
        if span == 0:
            self.warnings.append(
                f"{sid}: {hand.value} 장갑 기록이 에피소드 구간과 겹치지 않아 촉각 값이 모두 없음 "
                "(동기화 오프셋 확인)"
            )
        elif valid / span < self.policy.min_coverage:
            self.warnings.append(
                f"{sid}: {hand.value} 장갑 기록 구간 안 프레임의 {valid / span:.0%}만 촉각 값이 "
                f"있음 (기준 {self.policy.min_coverage:.0%}, 장갑 샘플 간격이 "
                f"tactile.interp_max_gap_ms {self.policy.interp_max_gap_ms} ms보다 긴지 확인)"
            )

    def _read(self, session: Session, stream: Stream) -> GloveData:
        """장갑 정규화 Parquet을 감사 저장소로 받아 읽는다 (원본 접근 기록이 남는다).

        Raises:
            ExportError: 감사 저장소를 받지 못했을 때, 스트림 위치가 이 원본 저장소가 아닐 때,
                객체가 없을 때, 형식이 틀릴 때.
        """
        where = f"{session.session_id}/{stream.stream_id}"
        off = "촉각 값 없이 내보내려면 config/policies/export.yaml tactile.enabled를 false로 둡니다"
        if self.raw is None:
            raise ExportError(
                f"{where}: 동기화된 장갑이 있는데 장갑 Parquet을 읽을 원본 감사 저장소가 없습니다 "
                f"(run_export의 raw). {off}"
            )
        prefix = self.raw.uri("")
        if not stream.uri.startswith(prefix):
            raise ExportError(f"{where}: 장갑 스트림이 원본 저장소({self.raw.bucket})에 없습니다")
        key = stream.uri.removeprefix(prefix)
        # 존재 확인은 내용을 읽지 않아 감사 기록 대상이 아니다 (AuditedStore.head)
        if self.raw.head(key) is None:
            raise ExportError(f"{where}: 장갑 정규화 Parquet이 원본 버킷에 없습니다. {off}")
        path = self.work / "tactile" / session.session_id / f"{stream.stream_id}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw.get_file(key, path)
        try:
            return read_glove(path)
        except ValueError as exc:
            raise ExportError(f"{where}: 장갑 Parquet을 읽지 못했습니다: {exc}") from exc

    def _channels(self, glove: GloveData, where: str) -> tuple[str, ...]:
        """정책이 고른 채널.

        Raises:
            ExportError: 고른 채널이 없거나, 채널 이름이 유효 표시 이름과 겹칠 때.
        """
        channels = select_channels(tuple(glove.channels), self.policy.channel_prefixes)
        if not channels:
            raise ExportError(
                f"{where}: 접두사 {list(self.policy.channel_prefixes)}로 시작하는 장갑 채널이 "
                f"없습니다 (파일의 채널: {', '.join(sorted(glove.channels))}). "
                "config/policies/export.yaml tactile.channel_prefixes를 이 장갑에 맞게 고치세요"
            )
        if PRESENT in channels:
            raise ExportError(
                f"{where}: 장갑 채널 이름 {PRESENT!r}는 유효 표시 이름과 겹쳐 실을 수 없습니다 "
                "(tactile.channel_prefixes로 빼거나 수집 때 채널 이름을 바꾼다)"
            )
        return channels

    def _check_layout(self, session: Session, hand: Hand, channels: tuple[str, ...]) -> None:
        """같은 손의 채널 구성이 앞 세션과 같은지. 처음이면 이 세션이 구성을 정한다.

        Raises:
            ExportError: 구성이 다를 때 (다른 채널을 0으로 채우거나 버려서 맞추지 않는다).
        """
        model = session.calibration.glove_model
        if hand not in self._layout:
            self._layout[hand] = channels
            self._first[hand] = (session.session_id, model)
            return
        if self._layout[hand] == channels:
            return
        first_sid, first_model = self._first[hand]
        expected = self._layout[hand]
        differ = sorted(set(expected) ^ set(channels), key=_natural)
        raise ExportError(
            f"{session.session_id}: {hand.value} 장갑의 채널 구성이 {first_sid}와 다릅니다 "
            f"({len(channels)}개 vs {len(expected)}개, 한쪽에만 있는 채널 {differ[:8]}, 장갑 기종 "
            f"{model!r} vs {first_model!r}). 한 데이터셋 안에서 촉각 열 모양이 달라지므로 맞춰 "
            "넣지 않습니다. 장갑 기종별로 나눠 내보내거나, config/policies/export.yaml "
            "tactile.channel_prefixes로 공통 채널만 고르거나, tactile.enabled를 false로 둡니다"
        )
