"""블러 렌더. 원본 PTS를 그대로 두고 오디오는 넣지 않는다.

블러는 강한 모자이크(박스 짧은 변을 blocks_per_box개 이하 블록으로 평균) 또는 단색 채움이다.
약한 가우시안 블러는 쓰지 않는다. 박스는 라벨 키프레임 시각(정수 ms)과 프레임 시각을 반올림해
맞춘다. 키프레임이 없는 프레임은 검수 화면(CVAT)이 보여 준 것과 같게 정한다.
- 직전 키프레임이 화면 밖(outside)이면 블러 없음.
- 직전·다음 키프레임이 모두 보이면 두 박스를 **프레임 번호** 비율로 선형 보간한다. CVAT는
  프레임 번호로 보간하므로 가변 프레임레이트(VFR) 영상에서 시각 비율로 보간하면 검수 화면과
  블러본의 박스가 달라진다. 프레임 번호는 렌더할 때 그 영상의 PTS 프레임 시각 순서로만 쓰고
  저장하지 않는다 (ADR 0024).
- 다음 키프레임이 화면 밖이거나 없으면 직전 박스를 그대로 유지한다.
검수자는 CVAT에서 키프레임 몇 개만 고치므로 (CVAT는 키프레임만 돌려준다) 보간하지 않으면
검수 화면과 블러본의 박스가 달라진다.

WP5. `runner.render_session`(`dlp privacy render`)이 승인된 세션의 운영 블러 라벨로 부른다.
입력은 로컬 원본 영상, 출력은 로컬 블러본 파일이다 (라벨링 버킷 업로드는 runner가 한다).
정책 값: privacy.yaml `render`(min_block_px, blocks_per_box, encoder_rate, crf),
defaults.yaml `privacy.render_mode`. 이 값과 라벨 집합이 `runner.render_hash`를 정한다.

- `FramePositions`: PTS 시각 → 프레임 순서 위치 (보간용).
- `blur_tracks`: 라벨 → 렌더용 트랙 (`_Track.box_at`으로 임의 프레임 박스).
- `apply_blur`: 이미지 한 장의 박스 영역을 모자이크/단색으로 덮는다 (제자리 수정).
- `render_blurred`: 영상 전체를 렌더한다.
"""

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
import numpy as np

from dlp_media.probe import to_fraction
from dlp_media.pts import build_pts_index
from dlp_privacy.detection import Image
from dlp_privacy.geometry import Box
from dlp_schema.labels import BlurTrackPayload, BoxKeyframe, LabelRecord


class FramePositions:
    """영상 프레임 시각(PTS, 정수 ms) → 프레임 순서 위치.

    프레임 사이 시각은 이웃 프레임 사이 비율로 둔다.
    """

    def __init__(self, frame_times: list[int]) -> None:
        """frame_times: 그 영상의 모든 프레임 시각 (정수 ms, 순서 무관 — 정렬해 쓴다)."""
        self.times = sorted(frame_times)
        self.index = {t: i for i, t in enumerate(self.times)}

    def at(self, t_ms: int) -> float:
        """시각 → 프레임 위치(실수).

        - 프레임 시각과 같으면 그 프레임 번호.
        - 두 프레임 사이면 (앞 프레임 번호 + 두 프레임 사이 시각 비율).
        - 첫 프레임 앞·마지막 프레임 뒤는 1초를 프레임 1개로 친 근사값 (키프레임이 영상 밖일 때만
          쓰인다).
        - 프레임이 하나도 없으면 t_ms 그대로.
        """
        i = self.index.get(t_ms)
        if i is not None:
            return float(i)
        ts = self.times
        j = bisect.bisect_left(ts, t_ms)
        if not ts:
            return float(t_ms)
        if j == 0:
            return (t_ms - ts[0]) / 1000.0  # 첫 프레임 앞 (보간에는 쓰이지 않는다)
        if j >= len(ts):
            return len(ts) - 1 + (t_ms - ts[-1]) / 1000.0
        return j - 1 + (t_ms - ts[j - 1]) / (ts[j] - ts[j - 1])


@dataclass(frozen=True)
class _Track:
    """렌더용 블러 트랙 (라벨 하나의 키프레임)."""

    # 키프레임 시각 (frames와 같은 순서, bisect용)
    times: list[int]
    # 키프레임 (시각 오름차순·중복 없음: LabelRecord 검증이 보장한다)
    frames: list[BoxKeyframe]
    # 이 영상의 프레임 위치 (모든 트랙이 같은 인스턴스를 공유)
    positions: FramePositions

    def box_at(self, t_ms: int) -> Box | None:
        """프레임 시각 t_ms에 그릴 박스. 블러가 없으면 None.

        규칙은 모듈 docstring 참고 (직전 키프레임 기준, CVAT와 같은 프레임 번호 보간).
        첫 키프레임 앞이면 None, 마지막 키프레임 뒤면(outside가 아니면) 그 박스를 유지한다.
        """
        i = bisect.bisect_right(self.times, t_ms) - 1  # t_ms 이하의 마지막 키프레임
        if i < 0:
            return None
        k = self.frames[i]
        if k.outside:
            return None
        box = Box(k.x, k.y, k.w, k.h)
        if k.t_ms == t_ms or i + 1 >= len(self.frames):
            return box
        nxt = self.frames[i + 1]
        if nxt.outside:
            return box
        # CVAT와 같이 프레임 번호로 보간한다 (VFR에서 시각 비율과 다르다)
        p0, p1 = self.positions.at(k.t_ms), self.positions.at(nxt.t_ms)
        if p1 <= p0:
            return box
        s = (self.positions.at(t_ms) - p0) / (p1 - p0)
        return box.lerp(Box(nxt.x, nxt.y, nxt.w, nxt.h), min(max(s, 0.0), 1.0))


def blur_tracks(labels: list[LabelRecord], frame_times: list[int]) -> list[_Track]:
    """frame_times: 블러를 입힐 영상의 프레임 시각 (PTS 인덱스).

    Args:
        labels: 블러 라벨 (blur_track 외 종류는 무시한다). 호출자가 운영 블러만 골라 넘긴다
            (`runner.operational_blur`).
        frame_times: 그 영상의 프레임 시각 (정수 ms).

    Returns:
        라벨마다 `_Track` 하나.
    """
    positions = FramePositions(frame_times)
    out: list[_Track] = []
    for label in labels:
        p = label.payload
        if isinstance(p, BlurTrackPayload):
            out.append(_Track([k.t_ms for k in p.keyframes], list(p.keyframes), positions))
    return out


def apply_blur(
    img: Image, box: Box, mode: Literal["mosaic", "solid"], min_block: int, blocks: int
) -> None:
    """img의 박스 영역을 제자리에서 블러한다 (부작용: img를 바꾼다).

    박스는 화면 안으로 자른 뒤 바깥쪽으로 정수화한다 (부분 픽셀도 덮는다).

    Args:
        img: (H, W, 3) RGB uint8 이미지.
        box: 픽셀 박스. 화면 밖이면 아무것도 하지 않는다.
        mode: "mosaic"이면 블록 평균, "solid"면 회색(128) 채움.
        min_block: 블록 한 변 최소 픽셀 (privacy.yaml render.min_block_px).
        blocks: 박스 짧은 변을 나눌 최대 블록 수 (render.blocks_per_box).
    """
    h, w = img.shape[:2]
    clipped = box.clipped(w, h)
    if clipped is None:
        return
    x1, y1 = int(np.floor(clipped.x)), int(np.floor(clipped.y))
    x2, y2 = int(np.ceil(clipped.x + clipped.w)), int(np.ceil(clipped.y + clipped.h))
    region = img[y1:y2, x1:x2]  # 뷰이므로 region을 고치면 img가 바뀐다
    if mode == "solid":
        region[:] = 128
        return
    # 블록 크기 = max(최소 블록, ceil(짧은 변 / 블록 수)) → 짧은 변에는 blocks개 이하 블록
    block = max(min_block, int(np.ceil(min(region.shape[0], region.shape[1]) / blocks)))
    for by in range(0, region.shape[0], block):
        for bx in range(0, region.shape[1], block):
            cell = region[by : by + block, bx : bx + block]
            cell[:] = cell.reshape(-1, 3).mean(axis=0).astype(np.uint8)


def render_blurred(
    src: Path,
    dst: Path,
    labels: list[LabelRecord],
    *,
    mode: Literal["mosaic", "solid"],
    min_block_px: int,
    blocks_per_box: int,
    encoder_rate: int,
    crf: int,
) -> int:
    """블러본을 쓰고 블러를 적용한 (프레임, 박스) 수를 돌려준다.

    encoder_rate·crf는 privacy.yaml render에서 온다.

    원본의 첫 비디오 트랙만 libx264(yuv420p)로 다시 인코딩한다. 각 프레임의 PTS·time_base는 원본
    그대로 두어 VFR과 마스터 타임라인이 유지된다. 오디오·데이터 트랙은 쓰지 않는다
    (defaults.yaml `privacy.strip_audio_in_release`, runner가 확인).

    Args:
        src: 로컬 원본 영상.
        dst: 쓸 블러본 경로 (mp4).
        labels: 운영 블러 라벨.
        mode, min_block_px, blocks_per_box: `apply_blur` 참고.
        encoder_rate: 인코더 명목 프레임레이트 (PTS에는 영향 없음).
        crf: libx264 CRF.

    Returns:
        블러를 그린 (프레임, 트랙) 쌍 수 (박스가 화면 밖으로 잘려 실제로 칠하지 않은 경우도 센다).
    """
    # 프레임 위치(보간용)는 패킷 PTS 인덱스로 만든다. 아래 디코딩 루프의 t_ms와 같은 반올림이다.
    tracks = blur_tracks(labels, [round(t) for t in build_pts_index(src).ms])
    applied = 0
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        vin = inp.streams.video[0]
        tb = to_fraction(vin.time_base)
        vout = out.add_stream("libx264", rate=encoder_rate)
        assert isinstance(vout, av.VideoStream)
        vout.width, vout.height = vin.codec_context.width, vin.codec_context.height
        vout.pix_fmt = "yuv420p"
        # 출력 time_base를 원본과 같게 두어 PTS 값을 그대로 옮긴다
        vout.time_base = tb
        vout.codec_context.time_base = tb
        vout.options = {"crf": str(crf), "preset": "veryfast", "threads": "1"}
        for frame in inp.decode(vin):
            if frame.pts is None:
                continue
            img = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            t_ms = round(float(frame.pts * tb * 1000))  # pipeline.detect_video와 같은 식
            for track in tracks:
                box = track.box_at(t_ms)
                if box is not None:
                    apply_blur(img, box, mode, min_block_px, blocks_per_box)
                    applied += 1
            new = av.VideoFrame.from_ndarray(img, format="rgb24")
            new.pts, new.time_base = frame.pts, tb
            out.mux(vout.encode(new))
        out.mux(vout.encode(None))  # 인코더에 남은 프레임을 비운다
    return applied
