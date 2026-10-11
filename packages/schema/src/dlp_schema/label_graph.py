"""현재 라벨의 상위 구간 연결 검사. DB 저장과 에피소드 계약이 같은 규칙을 사용한다."""

from collections.abc import Sequence

from dlp_schema.labels import ActionPayload, LabelRecord, SegmentLevel, SegmentPayload


def check_segment_links(labels: Sequence[LabelRecord]) -> list[str]:  # noqa: C901
    """자식 ID·중복·시간 포함·계층 순서·순환을 검사한다. 입력은 현재 라벨이어야 한다.

    ID는 label_id가 아니라 action_id / segment_id다. 같은 수준이나 상위 수준을 자식으로
    묶을 수 없다. 하위 단계가 없는 파일럿도 표현하도록 수준을 건너뛰는 연결은 허용한다.
    """
    problems: list[str] = []
    nodes: dict[str, LabelRecord] = {}
    ranks = {SegmentLevel.SKILL: 1, SegmentLevel.SUBSTEP: 2, SegmentLevel.TASK: 3}
    for label in labels:
        payload = label.payload
        if not isinstance(payload, (ActionPayload, SegmentPayload)):
            continue
        ref = payload.action_id if isinstance(payload, ActionPayload) else payload.segment_id
        if ref in nodes:
            problems.append(f"구간·행동 ID 중복: {ref}")
        nodes[ref] = label
    edges: dict[str, tuple[str, ...]] = {}
    for ref, label in nodes.items():
        p = label.payload
        if not isinstance(p, SegmentPayload):
            continue
        edges[ref] = p.child_ids
        if len(set(p.child_ids)) != len(p.child_ids):
            problems.append(f"{ref}: 자식 ID 중복")
        for child_id in p.child_ids:
            child = nodes.get(child_id)
            if child is None:
                problems.append(f"{ref}: 알 수 없는 자식 구간·행동 {child_id}")
                continue
            cp = child.payload
            if isinstance(cp, SegmentPayload) and ranks[cp.level] >= ranks[p.level]:
                problems.append(f"{ref}: 자식 {child_id}의 계층 순서 위반")
            if child.t_start_ms < label.t_start_ms or child.t_end_ms > label.t_end_ms:
                problems.append(f"{ref}: 자식 {child_id}이 상위 구간 밖에 있습니다")
    # 반복 DFS로 긴 계층에서도 Python 재귀 한도를 피한다.
    finished: set[str] = set()
    for root in edges:
        visiting: set[str] = set()
        stack = [(root, False)]
        while stack:
            node, leaving = stack.pop()
            if leaving:
                visiting.discard(node)
                finished.add(node)
            elif node in visiting:
                problems.append(f"구간 참조 순환: {node}")
            elif node not in finished:
                visiting.add(node)
                stack.append((node, True))
                stack.extend((child, False) for child in edges.get(node, ()) if child in edges)
    return problems
