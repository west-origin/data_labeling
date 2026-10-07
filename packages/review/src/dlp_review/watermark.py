"""라벨러 ID 워터마크. 화면 캡처가 유출되면 누가 본 영상인지 알 수 있게 한다.

반투명 글씨를 화면 전체에 비스듬히 반복해 넣는다. 원본 PTS를 유지하고 오디오는 넣지 않는다.

파이프라인 위치: `dlp_review.tasks.create_labeling_tasks`(작업 라벨 검수 작업 만들기)가
블러본에 `"<담당자> <세션>"` 글씨를 입혀
`sessions/<세션>/review/<담당자>/<스트림>.mp4`(라벨링 버킷)로 올린다.
관련: WP6, ADR 0006.

공개 함수:
- `watermark_layer`: 워터마크 글씨 마스크(0~255) 이미지.
- `burn_watermark`: 영상 프레임마다 마스크를 섞어 새 영상으로 인코딩한다.

주의점:
- 프레임 PTS와 time_base를 원본 그대로 두므로 공간 라벨 키프레임 시각(스트림 PTS ms, ADR 0019)이
  워터마크 영상에서도 같은 프레임을 가리킨다. 테스트가 PTS 일치를 검사한다.
- 불투명도·화질·명목 프레임레이트는 `config/policies/review.yaml` `media` 절에서 온다.
"""

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from pathlib import Path

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_media.probe import to_fraction


def watermark_layer(text: str, width: int, height: int) -> NDArray[np.uint8]:
    """흰 글씨 마스크 (0~255). 30도 기울인 줄로 화면 전체에 반복한다.

    인자: text(넣을 글씨), width·height(영상 화소 크기).
    반환: (height, width) uint8 배열. 255가 글씨, 0이 배경이다.

    동작: 영상 대각선 길이의 정사각형 캔버스에 글씨를 줄마다 반 칸씩 엇갈려 반복해 쓰고,
    가운데를 기준으로 30도 회전한 뒤 영상 크기만큼 가운데를 잘라 낸다(회전 후 모서리가 비지 않게
    캔버스를 대각선 크기로 잡는다).
    """
    size = int(np.hypot(width, height)) + 1
    canvas = np.zeros((size, size), dtype=np.uint8)
    # 글씨 크기: 너비 900 px 기준 1.0, 작은 영상에서도 0.4 아래로 줄이지 않는다
    scale = max(0.4, width / 900)
    # 글씨 사이 가로·세로 간격 (화소). 글씨 크기에 비례한다
    step_x, step_y = int(260 * scale * 2), int(70 * scale * 2)
    for row, y in enumerate(range(0, size, step_y)):
        # 홀수 줄은 반 칸 왼쪽에서 시작해 벽돌 쌓기처럼 엇갈린다
        for x in range(-(row % 2) * step_x // 2, size, step_x):
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, cv2.LINE_AA)
    rot = cv2.getRotationMatrix2D((size / 2, size / 2), 30, 1.0)
    rotated = cv2.warpAffine(canvas, rot, (size, size))
    # 회전한 캔버스 가운데에서 영상 크기만큼 잘라 낸다
    oy, ox = (size - height) // 2, (size - width) // 2
    return np.ascontiguousarray(rotated[oy : oy + height, ox : ox + width], dtype=np.uint8)


def burn_watermark(
    src: Path, dst: Path, text: str, *, opacity: float, crf: int, encoder_rate: int
) -> None:
    """opacity·crf·encoder_rate는 config/policies/review.yaml media에서 온다 (PTS는 원본 그대로).

    인자:
    - src: 입력 영상 (블러본). 첫 영상 스트림만 쓰고 오디오는 버린다.
    - dst: 출력 MP4 경로 (덮어쓴다).
    - text: 워터마크 글씨 (보통 `"<담당자> <세션>"`).
    - opacity: 글씨 불투명도 (0, 1]. `media.watermark_opacity`.
    - crf: libx264 화질 (0~51, 낮을수록 고화질). `media.watermark_crf`.
    - encoder_rate: 인코더 명목 프레임레이트. 프레임 PTS를 원본 그대로 쓰므로 실제 재생 시각에는
      영향이 없다 (`media.encoder_rate`).

    부작용: dst 파일 쓰기. PTS가 없는 프레임은 건너뛴다.
    """
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        vin = inp.streams.video[0]
        tb = to_fraction(vin.time_base)
        w, h = vin.codec_context.width, vin.codec_context.height
        # 알파 마스크 (h, w, 1), 0~opacity. RGB 세 채널에 같은 값으로 퍼진다
        mask = watermark_layer(text, w, h).astype(np.float32)[:, :, None] / 255 * opacity
        vout = out.add_stream("libx264", rate=encoder_rate)
        assert isinstance(vout, av.VideoStream)
        vout.width, vout.height, vout.pix_fmt = w, h, "yuv420p"
        # 출력 time_base를 입력과 같게 두어 PTS 값을 그대로 옮길 수 있게 한다 (VFR 유지)
        vout.time_base = tb
        vout.codec_context.time_base = tb
        vout.options = {"crf": str(crf), "preset": "veryfast", "threads": "1"}
        for frame in inp.decode(vin):
            if frame.pts is None:
                continue
            img = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.float32)
            # 알파 합성: 글씨 자리를 흰색(255) 쪽으로 opacity만큼 섞는다
            marked = (img * (1 - mask) + 255 * mask).astype(np.uint8)
            new = av.VideoFrame.from_ndarray(marked, format="rgb24")
            new.pts, new.time_base = frame.pts, tb
            out.mux(vout.encode(new))
        # 인코더 버퍼 비우기 (남은 프레임을 내보낸다)
        out.mux(vout.encode(None))
