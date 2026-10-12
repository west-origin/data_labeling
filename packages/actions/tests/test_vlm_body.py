"""행동 VLM 요청 본문 고정 테스트 (WP17.1, ADR 0040).

행동 단계의 버전 문자열 `vlm-<model>`에는 요청 형식이 들어가지 않는다. 그래서 공통 전송
(`dlp_models.vlm`)으로 옮기면서 요청 바이트가 조금이라도 바뀌면, 버전은 그대로인데 결과만 조용히
달라진다. `golden/vlm_body.json`은 리팩토링 **전** 코드(`OpenAICompatibleVlm.complete`)가 보낸
본문을 그대로 저장한 파일이다. 지금 코드가 같은 입력에서 바이트까지 같은 본문을 보내야 한다.

프레임은 고정 바이트로 바꿔 넣는다 (`sample_frames`는 바꾸지 않았고 기존 테스트가 따로 본다).
4xx는 이제 설정 오류(`VlmRequestError`)로 즉시 올라가고(재시도 없음), 명령은 종료 5로 끝난다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import httpx
import pytest

import dlp_actions.clients as clients
from dlp_actions.clients import OpenAICompatibleVlm
from dlp_actions.vlm import SegmentRequest, VlmRequestError, VlmUnavailableError, classify
from dlp_schema.labels import Hand
from dlp_schema.ontology import Ontology, load_ontology

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = Path(__file__).parent / "golden" / "vlm_body.json"
FRAMES = [b"\xff\xd8\xff\xe0frame-0", b"\xff\xd8\xff\xe0frame-1\x00\xff"]
SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["label", "description"],
    "properties": {
        "label": {"enum": ["action", "gap"]},
        "verb": {"enum": ["grasp", "rub", None]},
        "description": {"type": ["string", "null"], "maxLength": 200},
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    },
}
PROMPT = (
    "1인칭 돌봄·청소 영상의 100~900 ms 구간 프레임이다.\n"
    '"따옴표"와 역슬래시 \\ 와 탭\t도 그대로.\nJSON 하나만 출력한다.'
)


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    """저장소의 온톨로지 v1."""
    return load_ontology(ROOT / "config/ontology/v1")


def _request() -> SegmentRequest:
    """고정 구간 요청 (영상 없음: 프레임은 sample_frames 대체로 넣는다)."""
    return SegmentRequest(
        "s001", "bodycam", None, Hand.RIGHT, 100, 900, True, ("cup_01", "sink_01")
    )


def test_request_body_is_byte_identical_to_pre_refactor(monkeypatch: pytest.MonkeyPatch) -> None:
    """공통 전송으로 옮긴 뒤에도 요청 본문 바이트와 버전 문자열이 리팩토링 전과 같다."""

    def frames(request: SegmentRequest, n: int, max_side: int) -> list[bytes]:
        return list(FRAMES)

    monkeypatch.setattr(clients, "sample_frames", frames)
    seen: list[bytes] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    client = OpenAICompatibleVlm(
        "http://vlm", "Qwen/Qwen2.5-VL-7B-Instruct", frames=2, max_side=448, timeout_s=5,
        transport=httpx.MockTransport(handler),
    )  # fmt: skip
    assert client.version == "vlm-Qwen/Qwen2.5-VL-7B-Instruct"
    assert client.complete(_request(), PROMPT, SCHEMA) == "{}"
    assert seen == [GOLDEN.read_bytes()]


def test_api_key_goes_to_header_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """서버 API 키(`DLP_VLM_API_KEY`)는 Authorization 헤더로만 가고 본문은 골든과 같다.
    키가 없으면 Authorization 헤더를 보내지 않는다 (리팩토링 전과 같음)."""

    def frames(request: SegmentRequest, n: int, max_side: int) -> list[bytes]:
        return list(FRAMES)

    monkeypatch.setattr(clients, "sample_frames", frames)
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    for key in ("k-test", None):
        client = OpenAICompatibleVlm(
            "http://vlm", "Qwen/Qwen2.5-VL-7B-Instruct", frames=2, max_side=448, timeout_s=5,
            api_key=key, transport=httpx.MockTransport(handler),
        )  # fmt: skip
        client.complete(_request(), PROMPT, SCHEMA)
    with_key, without = seen
    assert with_key.headers["authorization"] == "Bearer k-test"
    assert "authorization" not in without.headers
    assert with_key.content == without.content == GOLDEN.read_bytes()


def test_4xx_is_config_error_without_retry(ontology: Ontology) -> None:
    """4xx는 장애가 아니라 설정 오류다: 백오프 재시도 없이 바로 `VlmRequestError` (A-08 4xx 부분).
    5xx는 그대로 장애로 다시 묻는다."""
    calls: list[int] = []

    def bad_request(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="unauthorized")

    client = OpenAICompatibleVlm(
        "http://vlm", "m", frames=1, max_side=64, timeout_s=1,
        transport=httpx.MockTransport(bad_request),
    )  # fmt: skip
    waits: list[float] = []
    with pytest.raises(VlmRequestError):
        classify(
            client, _request(), ontology, max_retries=2, backoff_s=(1.0, 5.0), sleep=waits.append
        )
    assert calls == [1] and waits == []
    assert not issubclass(VlmRequestError, VlmUnavailableError)


def test_cli_exits_5_on_config_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`dlp actions run`은 설정 오류(4xx)면 종료 5로 끝난다 (트랜잭션이 되돌려진다)."""
    from dlp_cli import actions_cmds

    def boom(*args: Any, **kw: Any) -> Any:
        raise VlmRequestError("HTTP 401: unauthorized", status=401)

    class _Engine:
        def begin(self) -> Any:
            return _Tx()

        def dispose(self) -> None:
            return None

    class _Tx:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(actions_cmds, "run_actions", boom)

    def nothing(*args: Any) -> Any:
        return None

    def engine(*args: Any) -> _Engine:
        return _Engine()

    def url(*args: Any) -> str:
        return "sqlite://"

    monkeypatch.setattr(actions_cmds, "session_ontology", nothing)
    monkeypatch.setattr(actions_cmds.sa, "create_engine", engine)
    monkeypatch.setattr(actions_cmds, "database_url", url)
    args = argparse.Namespace(
        session_id="s001", vlm_url="http://vlm", store=f"local:{tmp_path}", url=None
    )
    assert actions_cmds.cmd_run(args) == 5
